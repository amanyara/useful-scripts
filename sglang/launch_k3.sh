#!/usr/bin/env bash
# ==============================================================================
# Kimi-K3 launch — 4 nodes x 8 GPUs = 32 ranks, TP32 / EP32, SM100 + CUDA 12.9.
#
# Run this ON EVERY NODE. --node-rank is derived from the node's own IP position
# in /root/paddlejob/workspace/hostfile, so the command line is identical
# everywhere; there is nothing per-node to edit.
#
#   bash launch_k3.sh --all           # fan out to all 4 nodes over ssh, wait for ready
#   bash launch_k3.sh --stop          # stop every rank on every node
#   bash launch_k3.sh                 # foreground, this node only
#   MOE_PATH=marlin bash launch_k3.sh --all    # fallback path (see below)
#
# Two MoE paths:
#   deepep (default) — --moe-a2a-backend deepep --moe-runner-backend deep_gemm.
#       The DeepGEMM runner is the ONLY runner on this platform that registers
#       the deepep_ll / deepep_normal permutes (moe_runner/deep_gemm.py). This
#       needs DeepEP + NVSHMEM over IB, and it is verified working on this
#       cluster: 32 ranks reached "The server is fired up and ready to roll!"
#       and served real completions. Note IBGDA came up despite
#       /proc/driver/nvidia/params having no PeerMappingOverride=1.
#   marlin (fallback) — --moe-a2a-backend none --moe-runner-backend marlin.
#       No NVSHMEM at all, plain TP32 all-reduce MoE. Slower, but it is what the
#       official cookbook uses for the 4-node H100 cell and marlin handles SiTU.
#       Only reach for it if a future image breaks the NVSHMEM transport.
#
# --moe-runner-backend is passed EXPLICITLY on purpose. The K3 override hook
# (arg_groups/overrides.py:_kimi_k3_moe_runner_overrides) only fires when the
# value is "auto", and because the trtllm-gen cubin pool is installed it would
# then silently pick flashinfer_mxfp4 — which registers no DeepEP permutes.
# ==============================================================================
set -euo pipefail

K3_ROOT="${K3_ROOT:-/root/paddlejob/gpfsspace/k3-env}"
V="$K3_ROOT/venv"
MODEL="${MODEL:-/root/paddlejob/workspace/env_run/moonshotai/Kimi-K3}"
HOSTFILE="${HOSTFILE:-/root/paddlejob/workspace/hostfile}"
LOG_DIR="${LOG_DIR:-$K3_ROOT/logs}"

SERVE_HOST="${SERVE_HOST:-0.0.0.0}"
SERVE_PORT="${SERVE_PORT:-30000}"
DIST_PORT="${DIST_PORT:-29500}"
NET_IFACE="${NET_IFACE:-bond0}"          # matches /etc/nccl.conf NCCL_SOCKET_IFNAME
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MOE_PATH="${MOE_PATH:-deepep}"           # deepep | marlin
MEM_FRACTION="${MEM_FRACTION:-0.85}"

# --------------------------------- stop mode ----------------------------------
# `pkill -f "sglang serve"` alone is not enough: the 8 scheduler processes are
# spawned children named `sglang::scheduler_TPn`, and if they are killed while
# holding CUDA contexts their workers can survive re-parented to PID 1, keeping
# every GPU at 100% util. So kill by both patterns and then sweep whatever still
# holds a GPU, by process group.
#
# The patterns are bracketed ("[s]glang") because pkill -f matches against full
# command lines, and the remote `ssh host 'pkill -f "sglang serve" ...'` wrapper
# shell has that string in its own cmdline — unbracketed, the first pkill kills
# the shell running it and every node reports UNREACHABLE while nothing is
# actually stopped. The bracketed regex still matches the real processes.
if [ "${1:-}" = "--stop" ]; then
  [ -f "$HOSTFILE" ] || { echo "ERROR: $HOSTFILE not found" >&2; exit 1; }
  while read -r host _; do
    [ -n "$host" ] || continue
    printf '%-16s ' "$host"
    ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$host" '
      pkill -f "[s]glang serve" 2>/dev/null || true
      pkill -f "[s]glang::"     2>/dev/null || true
      sleep 2
      pkill -9 -f "[s]glang::"  2>/dev/null || true
      PGS=$(for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
              ps -o pgid= -p $p 2>/dev/null; done | tr -d " " | sort -u)
      for pg in $PGS; do kill -KILL -- "-$pg" 2>/dev/null || true; done
      sleep 2
      echo "gpus_busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" \
           "mem/GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)MiB"
    ' || echo "UNREACHABLE"
  done < "$HOSTFILE"
  exit 0
fi

# ------------------------------- fan-out mode ---------------------------------
if [ "${1:-}" = "--all" ]; then
  mkdir -p "$LOG_DIR"
  STAMP=$(date +%m%d_%H%M%S)
  # ssh does not forward the environment, so every tunable is passed explicitly.
  FWD=$(printf '%s=%q ' \
    K3_ROOT "$K3_ROOT" MODEL "$MODEL" HOSTFILE "$HOSTFILE" \
    MOE_PATH "$MOE_PATH" MEM_FRACTION "$MEM_FRACTION" \
    SERVE_HOST "$SERVE_HOST" SERVE_PORT "$SERVE_PORT" DIST_PORT "$DIST_PORT" \
    NET_IFACE "$NET_IFACE" GPUS_PER_NODE "$GPUS_PER_NODE" \
    KILL_STRESS "${KILL_STRESS:-1}")
  i=0
  while read -r host _; do
    [ -n "$host" ] || continue
    log="$LOG_DIR/k3_rank${i}_${STAMP}.log"
    echo "== rank $i -> $host  ($log)"
    ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$host" \
      "$FWD nohup bash '$K3_ROOT/launch_k3.sh' > '$log' 2>&1 &" \
      || { echo "ERROR: could not start rank $i on $host" >&2; exit 1; }
    i=$((i + 1))
  done < "$HOSTFILE"
  MASTER_IP=$(awk 'NR==1{print $1}' "$HOSTFILE")
  echo
  echo "All $i ranks launched. Waiting for readiness (weights load + CUDA graph"
  echo "capture take ~4-5 min for 1.42 TiB over 32 ranks; timeout ${READY_TIMEOUT:-1800}s)."
  echo "  follow rank0 : tail -f $LOG_DIR/k3_rank0_${STAMP}.log"

  # $http_proxy is exported in the interactive PDC shell and would send this
  # cluster-internal probe to the site proxy, which answers 503 while curl still
  # exits 0 — an easy way to believe a dead server is healthy. Bypass it, and
  # judge on the HTTP status code only.
  DEADLINE=$(( $(date +%s) + ${READY_TIMEOUT:-1800} ))
  while :; do
    code=$(no_proxy='*' NO_PROXY='*' curl -s -o /dev/null -w '%{http_code}' \
             --max-time 5 "http://$MASTER_IP:$SERVE_PORT/health" 2>/dev/null || true)
    [ "$code" = "200" ] && break
    # A rank that dies takes the whole job down; surface it instead of waiting.
    if grep -lq -e "Segfault encountered" -e "scheduler died" -e "^ERROR:" \
         "$LOG_DIR"/k3_rank*_"$STAMP".log 2>/dev/null; then
      echo
      echo "ERROR: a rank failed during startup. First failing log:" >&2
      grep -l -e "Segfault encountered" -e "scheduler died" -e "^ERROR:" \
        "$LOG_DIR"/k3_rank*_"$STAMP".log | head -1 >&2
      exit 1
    fi
    [ "$(date +%s)" -lt "$DEADLINE" ] || {
      echo; echo "ERROR: not ready within ${READY_TIMEOUT:-1800}s; inspect $LOG_DIR/k3_rank0_${STAMP}.log" >&2
      exit 1; }
    printf '.'; sleep 10
  done
  echo
  echo "Server ready on http://$MASTER_IP:$SERVE_PORT"
  echo "  models  : no_proxy='*' curl -s http://$MASTER_IP:$SERVE_PORT/v1/models"
  echo "  stop all: bash $K3_ROOT/launch_k3.sh --stop"
  exit 0
fi

# ------------------------------ node identity ---------------------------------
[ -f "$HOSTFILE" ] || { echo "ERROR: $HOSTFILE not found" >&2; exit 1; }
mapfile -t HOSTS < <(awk 'NF{print $1}' "$HOSTFILE")
NNODES="${#HOSTS[@]}"
[ "$NNODES" -ge 1 ] || { echo "ERROR: no hosts in $HOSTFILE" >&2; exit 1; }
MASTER="${HOSTS[0]}"

# The IP this node owns on the cluster fabric; also what sglang advertises.
LOCAL_IP=$(ip -o -4 addr show "$NET_IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)
[ -n "$LOCAL_IP" ] || { echo "ERROR: no IPv4 on $NET_IFACE" >&2; exit 1; }

NODE_RANK=-1
for i in "${!HOSTS[@]}"; do
  [ "${HOSTS[$i]}" = "$LOCAL_IP" ] && NODE_RANK="$i"
done
[ "$NODE_RANK" -ge 0 ] || {
  echo "ERROR: local $NET_IFACE address $LOCAL_IP is not in $HOSTFILE (${HOSTS[*]})" >&2
  exit 1
}
TP_SIZE=$((NNODES * GPUS_PER_NODE))
printf '== node_rank %d/%d  ip %s  tp/ep %d  moe_path %s\n' \
  "$NODE_RANK" "$NNODES" "$LOCAL_IP" "$TP_SIZE" "$MOE_PATH"

# ------------------------------- preflight ------------------------------------
[ -x "$V/bin/sglang" ] || { echo "ERROR: $V/bin/sglang missing — run setup_k3_env.sh first" >&2; exit 1; }
[ -f "$MODEL/config.json" ] || { echo "ERROR: model not found at $MODEL" >&2; exit 1; }

# The GPUs must be free: K3 needs ~44 GiB/GPU of weights plus KV cache, and the
# GPU stress test that may still be running holds memory and all SM time.
# KILL_STRESS=0 turns the kill into a hard error instead.
#
# The kill targets the PROCESS GROUP, not the matched pid. gpu_burn.py is a
# multiprocessing launcher with one worker per GPU, and the workers' cmdline is
# `python -c from multiprocessing.spawn import spawn_main; ...` — it does not
# match the pattern. Killing only the parent re-parents all 8 workers to PID 1,
# where they keep every GPU at 100% util and 1 GiB allocated; the pattern then
# matches nothing, this check "passed", and the next check aborted the launch:
#     ERROR: 8 process(es) still hold GPUs on 10.51.201.87
# The workers stay in the launcher's process group, so -- -PGID reaches them.
STRESS=$(pgrep -f 'gpu_burn\.py|gpu_stress\.py' | tr '\n' ' ' || true)
if [ -n "$STRESS" ]; then
  if [ "${KILL_STRESS:-1}" != "1" ]; then
    echo "ERROR: GPU stress processes running on $LOCAL_IP ($STRESS) and KILL_STRESS=0" >&2
    exit 1
  fi
  echo "== stopping GPU stress processes on $LOCAL_IP: $STRESS"
  MY_PGID=$(ps -o pgid= -p $$ | tr -d ' ')
  PGIDS=$(ps -o pgid= -p $STRESS 2>/dev/null | tr -d ' ' | sort -u | grep -v "^${MY_PGID}$" || true)
  for pg in $PGIDS; do kill -TERM -- "-$pg" 2>/dev/null || true; done
  # Wait for the GPUs themselves to drain — the pattern disappearing proves
  # nothing, since the workers that hold the memory never matched it.
  for _ in $(seq 1 25); do
    [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ] && break
    sleep 1
  done
  if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -ne 0 ]; then
    for pg in $PGIDS; do kill -KILL -- "-$pg" 2>/dev/null || true; done
    for _ in $(seq 1 15); do
      [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ] && break
      sleep 1
    done
  fi
fi
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$BUSY" -eq 0 ] || {
  echo "ERROR: $BUSY process(es) still hold GPUs on $LOCAL_IP:" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
  exit 1
}
GPU_N=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[ "$GPU_N" -eq "$GPUS_PER_NODE" ] || {
  echo "ERROR: expected $GPUS_PER_NODE GPUs, found $GPU_N" >&2; exit 1; }

# ------------------------------- environment ----------------------------------
# NCCL's own settings (IB HCA list, GID index, timeouts) come from
# /etc/nccl.conf, which libnccl parses itself; they are not duplicated here.
export NCCL_SOCKET_IFNAME="$NET_IFACE"
export GLOO_SOCKET_IFNAME="$NET_IFACE"

# NVSHMEM, however, does NOT read /etc/nccl.conf — verified: after a full
# ncclCommInitRank the NVSHMEM_* keys from that file are still absent from the
# process environment. The interactive PDC shell happens to export them, but
# --all launches over non-interactive ssh, which inherits nothing. So NVSHMEM
# has to be configured here or it silently falls back to its own defaults.
#
# The IFNAME below is the one that must not be left to the default. NVSHMEM's
# UID bootstrap picks the first non-loopback interface, which here is eth2 on a
# per-node 33.10.x/25 subnet that is NOT routable between nodes:
#     from 10.51.201.87 -> 33.10.218.17:18003  timed out
#     from 10.51.201.87 -> 10.51.195.14:29500  connected
# The root then published its 33.10.x address, the other 24 ranks could not
# reach it, and each node's first scheduler died:
#     socketStartConnect: exceeded timeouts (3)
#     Bootstrap plugin init failed for 'nvshmem_bootstrap_uid.so.3'
#     !!!!!!! Segfault encountered !!!!!!!  (in dlsym on the failed handle)
#     RuntimeError: Rank 0 scheduler died during initialization (exit code: -11)
# Rank0, being the bootstrap root, never connects out — it just blocked in
# accept() forever, so the node looked alive at 0% GPU util with port 30000
# closed. Pin the bootstrap to the routable fabric interface instead.
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME="$NET_IFACE"
export NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET

# Re-export the site's NVSHMEM fabric settings from /etc/nccl.conf. That file is
# per-node here (each node lists its own NVSHMEM_HCA_LIST ordering), so it is
# read locally on every node rather than forwarded from rank0. Keys are filtered
# to NVSHMEM_* with shell-safe values; anything else in the file is ignored.
if [ -r /etc/nccl.conf ]; then
  while IFS='=' read -r key val; do
    case "$key" in NVSHMEM_[A-Z0-9_]*) ;; *) continue ;; esac
    case "$val" in ''|*[!A-Za-z0-9_,:.=-]*) continue ;; esac
    # An explicit setting in the environment wins over the file.
    [ -n "${!key:-}" ] || export "$key=$val"
  done < /etc/nccl.conf
fi
export NCCL_CUMEM_ENABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
export SGLANG_HOST_IP="$LOCAL_IP"
export SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL="$K3_ROOT/cubin/trtllm_gen_moe_cubin_pool"

# JIT caches must be NODE-LOCAL, never on shared GPFS.
#
# tvm-ffi serialises its ninja builds with fcntl.flock on <build_dir>/lock, but
# flock is NOT honoured across nodes on this GPFS mount — verified directly: two
# nodes both took LOCK_EX on the same file at the same time. With the cache on
# GPFS all 4 nodes build into one directory, so one node's `c++` links a
# cuda_0.o that another node's nvcc is still writing:
#     cuda_0.o: file not recognized: file format not recognized
# That killed all 8 ranks of one node ~6 min into startup, and the surviving 24
# ranks then spun in the NCCL barrier at 100% GPU until dist-timeout.
# Intra-node flock does work, so a per-node dir means one compile per node and
# 7 cache hits. deep_gemm JITs through tvm-ffi as well, so this covers it;
# FlashInfer already caches under $HOME/.cache, which is node-local here.
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-/root/.cache/k3-tvm-ffi}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/.cache/k3-triton}"
mkdir -p "$TVM_FFI_CACHE_DIR" "$TRITON_CACHE_DIR"

# No external network is needed at serve time: the FlashInfer cubin/jit-cache
# wheels are pre-installed and the remaining kernels JIT-compile locally with
# nvcc. Drop the proxy so (a) the long-lived server process does not carry proxy
# credentials in its environment, and (b) rank0 and the remotes behave
# identically — the remotes have no proxy set, so a hidden download would
# otherwise succeed on rank0 and fail on the other three.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# The weights are complete on local NVMe; never let a missing file turn into a
# 1.5 TiB hub download. A genuinely missing file should fail loudly instead.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# ---------------------------------- flags -------------------------------------
# Shape from the cookbook's 4-node "unified" cell (tp32/ep32, mem-fraction 0.85,
# dist-timeout 3600, kimi_k3 parsers). Attention backends are deliberately NOT
# passed: the K3 SM100 hook resolves both prefill and decode to trtllm_mla and
# sets page_size=64 (verified by a ServerArgs dry-run). --trust-remote-code is
# required by K3's TikTokenTokenizer auto_map.
ARGS=(
  --model-path "$MODEL"
  --trust-remote-code
  --tp-size "$TP_SIZE"
  --ep-size "$TP_SIZE"
  --nnodes "$NNODES"
  --node-rank "$NODE_RANK"
  --dist-init-addr "$MASTER:$DIST_PORT"
  --dist-timeout 3600
  --watchdog-timeout 3600
  --mem-fraction-static "$MEM_FRACTION"
  --model-loader-extra-config '{"enable_multithread_load": true}'
  --reasoning-parser kimi_k3
  --tool-call-parser kimi_k3
  --host "$SERVE_HOST"
  --port "$SERVE_PORT"
)

case "$MOE_PATH" in
  deepep)
    # deepep-mode auto = normal dispatch for prefill, low_latency for decode.
    ARGS+=( --moe-a2a-backend deepep --moe-runner-backend deep_gemm --deepep-mode auto )
    ;;
  marlin)
    ARGS+=( --moe-a2a-backend none --moe-runner-backend marlin )
    ;;
  *)
    echo "ERROR: MOE_PATH must be 'deepep' or 'marlin', got '$MOE_PATH'" >&2; exit 1 ;;
esac

# This server has NO authentication: anyone who can reach $SERVE_HOST:$SERVE_PORT
# can send inference requests. Keep it on a private interface, front it with an
# authenticating proxy, or pass --api-key before exposing it beyond the cluster.
if [ "$NODE_RANK" -eq 0 ] && [ "$SERVE_HOST" = "0.0.0.0" ]; then
  echo "== NOTE: listening on 0.0.0.0:$SERVE_PORT with no auth (add --api-key to require a token)"
fi

echo "== exec: sglang serve ${ARGS[*]}"
exec "$V/bin/sglang" serve "${ARGS[@]}"
