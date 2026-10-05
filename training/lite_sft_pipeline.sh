#!/bin/bash
# =============================================================================
# lite_sft_pipeline.sh —— Lite SFT 一条龙：挂 AFS → 拉模型/数据 → 生成 env → 起训练
# -----------------------------------------------------------------------------
# 为什么这么写（都是踩过的坑，改之前先看一眼）：
#   1. 用 orterun 而不是 /bin/mpirun —— /bin/mpirun 是包装器，已内置 --hostfile，
#      再传一次会报 "More than one hostfile was passed"。
#   2. 权重/数据一律放【每节点本地 NVMe】而不是 GPFS —— GPFS 配额会被平台改小
#      （实测被降到 512G），一个 ckpt 577G 根本放不下；本地盘每台 3T+ 且读最快。
#   3. 拉权重优先用内网 HTTP（实测 4.4 GB/s），AFS 挂载点只有 ~152 MB/s，差 30 倍。
#   4. 训练必须从 clone 的 examples_custom 目录启动 —— train 脚本用三层 dirname
#      反推 PYTHONPATH，从 env_run 根目录启动会指向不存在的路径。
#   5. torch 2.11+cu130 跑在 12.4 驱动上，必须挂 /usr/local/cuda-13.0/compat，
#      否则 torch.cuda.is_available()=False。
#   6. 起训前必须停占卡程序，训练结束后再拉起来。
#
# 用法：
#   bash lite_sft_pipeline.sh sync        # 把本脚本推到全部节点（其余 stage 会自动做）
#   bash lite_sft_pipeline.sh mount       # 16 节点挂 AFS（需 AFS_UGI/AFS_URI/AFS_MP）
#   bash lite_sft_pipeline.sh model       # 16 节点拉底座 ckpt 到本地
#   bash lite_sft_pipeline.sh data        # 16 节点拉训练数据并补齐 train/test/validation
#   bash lite_sft_pipeline.sh env         # 生成 .env（自动算 TOTAL_TRAIN_STEPS）并分发
#   bash lite_sft_pipeline.sh preflight   # 起训前 16 节点自检
#   bash lite_sft_pipeline.sh train       # 停占卡 → 后台起训练
#   bash lite_sft_pipeline.sh status      # 看 loss / 步数 / 报错
#   bash lite_sft_pipeline.sh fake-on|fake-off
#   bash lite_sft_pipeline.sh all         # model → data → env → preflight → train
#
# 常用覆盖（都可以用环境变量传）：
#   EXP=v18 CKPT_URL=http://10.95.252.148:8078/xxx-distckpt/ \
#   DATA_URL=http://10.95.252.148:8000/xxx.32768.final/ \
#   bash lite_sft_pipeline.sh all
# =============================================================================
set -uo pipefail

# ---------------- 可配置 ----------------
EXP=${EXP:-v18}                                     # 实验名，用于 env/日志/输出目录
WORKDIR=/root/paddlejob/workspace/env_run
LOCAL_ROOT=${LOCAL_ROOT:-/root/paddlejob/workspace}  # 本地盘根（模型/数据/日志都落这里）
SELF_PATH=$LOCAL_ROOT/lite_sft_pipeline.sh
HOSTFILE=${HOSTFILE:-$LOCAL_ROOT/hostfile}
IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
BRIDGE=$WORKDIR/baidu/amfp/Megatron-Bridge           # clone 出来的代码库
CONFDIR=$BRIDGE/examples_custom/recipes/mimo_v25/conf

# 数据源（HTTP 优先；留空则跳过对应 stage）
CKPT_URL=${CKPT_URL:-http://10.95.252.148:8078/Lite-post-train-V2.5-Base-bf16-with-kv-distckpt/}
DATA_URL=${DATA_URL:-http://10.95.252.148:8000/v17_basev16_format_filter_v2_lines_filter.lite.format.in32k.shuf.jsonl.32768.final/}
CKPT_ITER=${CKPT_ITER:-iter_0000258}                 # 底座里的 iter 目录名
MODEL_DIR=$LOCAL_ROOT/models/base_$CKPT_ITER          # 底座落地路径（每节点本地）
DATA_DIR=$LOCAL_ROOT/data/$EXP                        # 数据落地路径（每节点本地）

# AFS 挂载（可选，只有 mount stage 用；不填就别跑 mount）
AFS_UGI=${AFS_UGI:-}                                  # 形如 user,passwd
AFS_URI=${AFS_URI:-}                                  # 形如 afs://aries.afs.baidu.com:9902/user/xxx
AFS_MP=${AFS_MP:-$WORKDIR/new_afs_$EXP}               # 挂载点必须在 env_run 之下

# 训练超参（128 卡口径，改之前先算 EP = TP*CP*DP）
TEMPLATE=${TEMPLATE:-$CONFDIR/lite.v17.basev16.32k.bf16.16node.env}
ENVFILE=$CONFDIR/lite.$EXP.32k.bf16.16node.env
TP=${TP:-4}; PP=${PP:-4}; CP=${CP:-2}; EP=${EP:-32}
GBS=${GBS:-64}; SEQ=${SEQ:-32768}; SAVE_INTERVAL=${SAVE_INTERVAL:-48}
MOUNT_CKPT_PATH=${MOUNT_CKPT_PATH:-$WORKDIR/new_afs_lightning/zhangyuhan08/output_path/model_save/}
LOGDIR=$LOCAL_ROOT/log
LAUNCHER=run.v3.sp_v2.mimov25.tools.sh                # H800/H200/B200 用这个；B300 换 .b300.sh

export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}

log(){ echo "[$(date +%H:%M:%S)] $*"; }
die(){ echo "[ERROR] $*" >&2; exit 1; }
# 全节点执行（注意用 orterun，不能用 /bin/mpirun）
ort(){ orterun --allow-run-as-root --hostfile "$HOSTFILE" --mca btl_tcp_if_include "$IFNAME" \
       -pernode --bind-to none "$@"; }
NODES=$(grep -cve '^[[:space:]]*$' "$HOSTFILE" 2>/dev/null || echo 0)

# ---------------- 把自己推到全部节点 ----------------
sync_self(){
  [ -f "$HOSTFILE" ] || die "hostfile 不存在: $HOSTFILE"
  local n=0
  for h in $(awk '{print $1}' "$HOSTFILE"); do
    tar czf - -C "$(dirname "$SELF_PATH")" "$(basename "$SELF_PATH")" 2>/dev/null | \
      timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
      "mkdir -p $LOCAL_ROOT && tar xzf - -C $LOCAL_ROOT" >/dev/null 2>&1 && n=$((n+1))
  done
  log "脚本已同步到 $n/$NODES 个节点"
}
# 在全部节点跑本脚本的某个 _node 子命令
fan(){ ort bash "$SELF_PATH" "_node_$1" 2>&1 | sed 's/.*<stdout>://' | grep -E '^(tjzj|\[)' | sort; }

# ============ 以下 _node_* 只在各节点内部执行 ============
_node_mount(){
  [ -n "$AFS_UGI" ] && [ -n "$AFS_URI" ] || { echo "$(hostname -s) SKIP 未配置 AFS_UGI/AFS_URI"; exit 0; }
  if grep -q " $AFS_MP fuse" /proc/mounts; then echo "$(hostname -s) SKIP 已挂载"; exit 0; fi
  cd "$WORKDIR" || { echo "$(hostname -s) NO_WORKDIR"; exit 1; }
  sh ./tools/afs_mount.sh "${AFS_UGI%%,*}" "${AFS_UGI#*,}" "$AFS_MP" "$AFS_URI" >/dev/null 2>&1
  sleep 6
  grep -q " $AFS_MP fuse" /proc/mounts && echo "$(hostname -s) OK $(ls "$AFS_MP" | wc -l) 项" \
                                       || echo "$(hostname -s) FAIL"
}

# HTTP 并行下载一个目录（含一层子目录），比走 AFS 快约 30 倍
_http_pull(){
  local base=$1 dst=$2 par=${3:-8}
  mkdir -p "$dst"
  curl -sS -m 120 "$base" | grep -oE 'href="[^"]+"' | sed 's/href="//;s/"//' | grep -vE '^(\.\.|/)$' > "$dst/.all"
  grep -v '/$' "$dst/.all" > "$dst/.files"; grep '/$' "$dst/.all" > "$dst/.dirs" || true
  xargs -a "$dst/.files" -d '\n' -P "$par" -I{} curl -sS -f --retry 3 -o "$dst/{}" "$base{}"
  while IFS= read -r d; do
    [ -z "$d" ] && continue
    mkdir -p "$dst/$d"
    curl -sS -m 60 "$base$d" | grep -oE 'href="[^"]+"' | sed 's/href="//;s/"//' | grep -vE '^(\.\.|/)$' > "$dst/$d.f"
    xargs -a "$dst/$d.f" -d '\n' -P 4 -I{} curl -sS -f --retry 3 -o "$dst/$d{}" "$base$d{}"
    rm -f "$dst/$d.f"
  done < "$dst/.dirs"
  rm -f "$dst/.all" "$dst/.files" "$dst/.dirs"
}

_node_model(){
  # 底座结构必须是：<父目录>/<iter_xxx>/ + latest_checkpointed_iteration.txt
  # 少了 tracker，Megatron 会当空目录随机初始化，step1 loss 直接炸（手册说的 iterxxxx.txt）
  if [ -f "$MODEL_DIR/latest_checkpointed_iteration.txt" ] && [ -d "$MODEL_DIR/$CKPT_ITER" ]; then
    echo "$(hostname -s) SKIP 已存在 $(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)"; exit 0
  fi
  mkdir -p "$MODEL_DIR"
  _http_pull "$CKPT_URL" "$MODEL_DIR" 8
  local it; it=$(cat "$MODEL_DIR/latest_checkpointed_iteration.txt" 2>/dev/null)
  echo "$(hostname -s) files=$(find "$MODEL_DIR" -type f | wc -l) size=$(du -sh "$MODEL_DIR" | cut -f1) tracker=$it"
}

_node_data(){
  # 训练要求 <root>/{train,test,validation} 三个 split；源目录通常是扁平的 HF save_to_disk
  if [ -d "$DATA_DIR/train" ] && [ -d "$DATA_DIR/validation" ] && [ -d "$DATA_DIR/test" ]; then
    echo "$(hostname -s) SKIP 三个 split 已就绪"; exit 0
  fi
  _http_pull "$DATA_URL" "$DATA_DIR/train" 10
  for s in test validation; do [ -d "$DATA_DIR/$s" ] || cp -r "$DATA_DIR/train" "$DATA_DIR/$s"; done
  echo "$(hostname -s) train=$(ls "$DATA_DIR/train" | wc -l) test=$(ls "$DATA_DIR/test" | wc -l) validation=$(ls "$DATA_DIR/validation" | wc -l)"
}

_node_preflight(){
  local ok=1 msg=""
  [ -d "$BRIDGE/src" ] && [ -d "$BRIDGE/../megatron/megatron/core" ] || { ok=0; msg="$msg 代码库缺失;"; }
  [ -f "$ENVFILE" ] || { ok=0; msg="$msg env 缺失;"; }
  [ -f "$MODEL_DIR/latest_checkpointed_iteration.txt" ] || { ok=0; msg="$msg 底座 tracker 缺失;"; }
  [ -d "$DATA_DIR/train" ] || { ok=0; msg="$msg 数据缺失;"; }
  [ -w "$MOUNT_CKPT_PATH" ] || { ok=0; msg="$msg 上传目录不可写;"; }
  python -c "import cutlass" >/dev/null 2>&1 || { ok=0; msg="$msg cutlass 不可用(装 nvidia-cutlass-dsl-libs);"; }
  local gpu; gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr -d ' MiB' | awk '{s+=$1} END{print s}')
  [ "$gpu" -eq 0 ] || { ok=0; msg="$msg GPU 未清空(${gpu}MiB);"; }
  local dsk; dsk=$(df -BG "$LOCAL_ROOT" | awk 'NR==2{gsub("G","",$4); print $4}')
  [ "$dsk" -gt 800 ] || { ok=0; msg="$msg 本地盘不足(${dsk}G);"; }
  [ "$ok" = 1 ] && echo "$(hostname -s) READY env_md5=$(md5sum "$ENVFILE" | cut -c1-8)" \
                || echo "$(hostname -s) NOT_READY$msg"
}

# ============ driver 侧 stage ============
gen_env(){
  [ -f "$TEMPLATE" ] || die "模板 env 不存在: $TEMPLATE"
  local packs steps
  packs=$(python3 -c "
import json;d=json.load(open('$DATA_DIR/train/dataset_info.json'));print(list(d['splits'].values())[0]['num_examples'])" 2>/dev/null)
  [ -n "$packs" ] || die "读不到 packs 数（先跑 data stage）"
  steps=$(( (packs + GBS - 1) / GBS ))                      # 向上取整 = 1 个 epoch
  local world=$((NODES*8)) dp
  dp=$(( world / (TP*PP*CP) ))
  [ $(( (TP*CP*dp) % EP )) -eq 0 ] || die "EP=$EP 不能整除 TP*CP*DP=$((TP*CP*dp))，改 EP 或 CP"
  [ $steps -gt $((2*2)) ] || die "TOTAL_TRAIN_STEPS 必须 > 2*WARMUP_STEPS"
  sed -e "s|^JOB_NAME=.*|JOB_NAME='sft_${EXP}_32k_bf16_${NODES}node'|" \
      -e "s|^TB_NAME=.*|TB_NAME=\"lite.${EXP}.in32k.${NODES}node\"|" \
      -e "s|^NODES=.*|NODES='$NODES'|" \
      -e "s|^TP=.*|TP=$TP|" -e "s|^PP=.*|PP=$PP|" -e "s|^CP=.*|CP=$CP|" -e "s|^EP=.*|EP=$EP|" \
      -e "s|^GLOBAL_BATCH_SIZE=.*|GLOBAL_BATCH_SIZE=$GBS|" \
      -e "s|^SEQ_LENGTH=.*|SEQ_LENGTH=$SEQ|" -e "s|^PADDING_LENGTH=.*|PADDING_LENGTH=$SEQ|" \
      -e "s|^SAVE_INTERVAL=.*|SAVE_INTERVAL=$SAVE_INTERVAL|" \
      -e "s|^TOTAL_TRAIN_STEPS=.*|TOTAL_TRAIN_STEPS=$steps|" \
      -e "s|^LOAD_CHECKPOINT_PATH_PRE=.*|LOAD_CHECKPOINT_PATH_PRE=\"$MODEL_DIR\"|" \
      -e "s|^OUTPUT_BASE_PATH=.*|OUTPUT_BASE_PATH=\"$LOCAL_ROOT/ckpt/$EXP\"|" \
      -e "s|^MOUNT_CKPT_PATH=.*|MOUNT_CKPT_PATH=\"$MOUNT_CKPT_PATH\"|" \
      -e "s|^resume_steps=.*|resume_steps=\"$CKPT_ITER\"|" \
      "$TEMPLATE" > "$ENVFILE"
  python3 - "$ENVFILE" "$DATA_DIR" <<'PY'
import re,sys
p,d=sys.argv[1],sys.argv[2]
s=open(p).read()
s=re.sub(r'LOCAL_DATA_DIRS=\([^)]*\)', 'LOCAL_DATA_DIRS=(\n    "%s/train/"\n    "%s/validation/"\n)'%(d,d), s, flags=re.S)
open(p,'w').write(s)
PY
  bash -n "$ENVFILE" || die "生成的 env 语法错误"
  log "env 已生成: $ENVFILE"
  log "  packs=$packs GBS=$GBS -> TOTAL_TRAIN_STEPS=$steps | world=$world DP=$dp EP=$EP"
  cp -f "$ENVFILE" "$WORKDIR/recipes/mimo_v25/conf/" 2>/dev/null || true
  for h in $(awk '{print $1}' "$HOSTFILE"); do
    tar czf - -C "$CONFDIR" "$(basename "$ENVFILE")" 2>/dev/null | \
      timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "tar xzf - -C $CONFDIR" >/dev/null 2>&1
  done
  log "env 已分发到 $NODES 个节点"
}

start_train(){
  [ -f "$ENVFILE" ] || die "env 不存在，先跑 env stage"
  log "停占卡程序（起真训前必须停，否则拖慢甚至 OOM）"
  HOSTFILE=$HOSTFILE bash "$WORKDIR/fake_util_stop.sh" --no-verify >/dev/null 2>&1 || true
  sleep 5
  mkdir -p "$LOGDIR"
  local ts log
  ts=$(date +%m%d%H%M); log=$LOGDIR/train.$EXP.$ts.log
  [ "${IS_STANDALONE:-0}" = "1" ] && log "警告: IS_STANDALONE=1 会退化成单机 8 卡"
  log "从 clone 的 examples_custom 启动（三层 dirname 反推 PYTHONPATH 的要求）"
  ( cd "$BRIDGE/examples_custom" && \
    nohup bash "$LAUNCHER" mimo "recipes/mimo_v25/conf/$(basename "$ENVFILE")" > "$log" 2>&1 & )
  echo "$log" > "$LOCAL_ROOT/.current_train_log"
  log "训练已后台启动，日志: $log"
  log "看进度: bash $SELF_PATH status"
}

show_status(){
  local L; L=$(cat "$LOCAL_ROOT/.current_train_log" 2>/dev/null)
  [ -n "$L" ] && [ -f "$L" ] || die "找不到当前训练日志"
  echo "日志: $L"
  grep -oE 'successfully loaded checkpoint[^[]*' "$L" | tail -1
  grep -oE 'iteration +[0-9]+/ *[0-9]+.*lm loss: [0-9.E+-]+' "$L" | tail -3 | \
    sed -E 's/.*iteration +([0-9]+)\/ *([0-9]+).*elapsed time per iteration \(ms\): ([0-9.]+).*lm loss: ([0-9.E+-]+).*/  step \1\/\2  \3ms  lm_loss=\4/'
  local err; err=$(grep -cE 'CUDA out of memory|Traceback|No space left' "$L")
  echo "  报错计数: $err"
  [ "$err" -gt 0 ] && grep -oE '(RuntimeError|ValueError|OutOfMemoryError|AssertionError): .{0,100}' "$L" | sort -u | head -3
  echo "  step1 loss 应 < 1，否则说明底座没加载成功（检查 tracker 与 iter 目录名是否一致）"
}

case "${1:-}" in
  sync)       sync_self ;;
  mount)      sync_self; log "16 节点挂载 $AFS_URI -> $AFS_MP"; fan mount ;;
  model)      sync_self; log "拉底座到本地: $CKPT_URL -> $MODEL_DIR"; fan model ;;
  data)       sync_self; log "拉数据到本地: $DATA_URL -> $DATA_DIR"; fan data ;;
  env)        gen_env ;;
  preflight)  sync_self; fan preflight ;;
  train)      start_train ;;
  status)     show_status ;;
  fake-on)    HOSTFILE=$HOSTFILE bash "$WORKDIR/fake_util_start.sh" --no-verify ;;
  fake-off)   HOSTFILE=$HOSTFILE bash "$WORKDIR/fake_util_stop.sh" ;;
  all)        sync_self; fan model; fan data; gen_env; fan preflight
              echo; read -r -p "预检结果如上，回车开始训练，Ctrl-C 取消 " _; start_train ;;
  _node_*)    "${1}" ;;                       # 节点内部调用
  *)          sed -n '2,40p' "$0"; exit 1 ;;
esac
