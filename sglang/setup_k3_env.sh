#!/usr/bin/env bash
# ==============================================================================
# Kimi-K3 (1.42 TiB / mxfp4-pack-quantized) serving environment — SGLang native
# build for 4 nodes x 8 GPUs (SM100, CUDA 12.9), no Docker.
#
# Reproduces the official recipe in sglang/docker/kimi_k3/kimi_k3_cu12.Dockerfile
# natively, because this cluster is k8s and cannot run docker in-container. The
# cookbook explicitly sanctions this: "If you do not want to use a Docker image,
# reproduce the dependency installation steps from the CUDA 13 / CUDA 12
# Dockerfile."
#
# Everything lands on shared GPFS ($K3_ROOT), so this script runs ONCE on rank0
# and all 4 nodes see the identical venv, the rebuilt DeepEP .so and the patched
# DeepGEMM headers. Only the model weights are per-node (local NVMe).
#
# Idempotent: every step is guarded and safe to re-run.
#
# Usage:   bash setup_k3_env.sh              # build + verify on all 4 nodes
#          VERIFY_ONLY=1 bash setup_k3_env.sh
#
# External downloads go through the proxy already exported in this shell
# ($http_proxy / $https_proxy). This script never hard-codes those credentials.
# ==============================================================================
set -euo pipefail

# ---------------------------------- config ------------------------------------
K3_ROOT="${K3_ROOT:-/root/paddlejob/gpfsspace/k3-env}"
HOSTFILE="${HOSTFILE:-/root/paddlejob/workspace/hostfile}"
PY_SYS="${PY_SYS:-/usr/bin/python3.12}"

# Recipe commit: branch tip deleted docker/kimi_k3/, so pin the commit that has it.
SGLANG_REPO="https://github.com/sgl-project/sglang.git"
SGLANG_COMMIT="bd51cab0c01075c9b73cc1dfae9d27b7b0c95619"
SGLANG_VERSION="0.5.16"          # depth-1 clone has 0 tags -> setuptools-scm needs this

# DeepEP lineage pinned by the K3 Dockerfile (origin/hybrid-ep).
DEEPEP_REPO="https://github.com/deepseek-ai/DeepEP.git"
DEEPEP_COMMIT="d28bd676c2120573c9f1425f0c16c39faa4117e6"
DEEPEP_LOCAL_MIRROR="/root/paddlejob/DeepEP"     # optional fast path
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0a}"   # SM100 only on this cluster

FLASHINFER_VERSION="0.6.15.post1"
DEEP_GEMM_WHL="sgl_deep_gemm-0.1.5+cu129-py3-none-manylinux2014_x86_64.whl"
DEEP_GEMM_URL="https://github.com/sgl-project/whl/releases/download/v0.1.5/sgl_deep_gemm-0.1.5%2Bcu129-py3-none-manylinux2014_x86_64.whl"
DEEP_GEMM_SHA="6ee67b2cc19b3227a376f6cd2b4f8c81bbdbfad1a9c39c984ba4fd6333e8f7c2"
SGL_KERNEL_WHL="sglang_kernel-0.4.5+cu129-cp310-abi3-manylinux2014_x86_64.whl"
SGL_KERNEL_URL="https://github.com/sgl-project/whl/releases/download/v0.4.5/sglang_kernel-0.4.5%2Bcu129-cp310-abi3-manylinux2014_x86_64.whl"
SGL_KERNEL_SHA="aa09af4599121558f813e7089caa9d4ff43e7a31b3794c0e48523085484484f6"

CUBIN_URL="https://github.com/sgl-project/whl/releases/download/trtllm_gen_moe_cubin_20260617/trtllm_gen_moe_cubin_pool_20260617_v0613rc1.zip"
CUBIN_SHA="4900501cbe782a76b08a5858f9f07152287b97cb68114466dac286366b66c192"
CUBIN_ROOT="trtllm_gen_moe_cubin_pool_20260617_v0613rc1"
CUBIN_COUNT=1696
SG="$K3_ROOT/sglang.gh"
V="$K3_ROOT/venv"
PIP="$V/bin/pip"
PY="$V/bin/python"
DEEPEP_DIR="$K3_ROOT/DeepEP"
WHEELS="$K3_ROOT/wheels"
CUBIN_POOL="$K3_ROOT/cubin/trtllm_gen_moe_cubin_pool"

log()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32mOK\033[0m %s\n' "$*"; }
skip() { printf '   -- %s (already done)\n' "$*"; }
die()  { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ------------------------------- 0. preflight ---------------------------------
log "[0/12] preflight"
[ -x "$PY_SYS" ] || die "system python $PY_SYS not found"
# nvcc is needed both here (DeepEP build) and at serve time (tvm-ffi / deep_gemm
# JIT). It is NOT on the non-interactive ssh PATH on these nodes; every consumer
# falls back to $CUDA_HOME/$CUDA_PATH and then to /usr/local/cuda, which is why
# the remote ranks can still JIT. Accept either.
if command -v nvcc >/dev/null; then
  nvcc --version | tail -2 | head -1
elif [ -x /usr/local/cuda/bin/nvcc ]; then
  export PATH="/usr/local/cuda/bin:$PATH"
  echo "   nvcc not on PATH; using /usr/local/cuda/bin/nvcc"
  nvcc --version | tail -2 | head -1
else
  die "nvcc found neither on PATH nor at /usr/local/cuda/bin/nvcc"
fi
CUDART=$("$PY_SYS" -c 'import torch;print(torch.version.cuda)')
echo "   torch $("$PY_SYS" -c 'import torch;print(torch.__version__)') / CUDA $CUDART"
case "$CUDART" in
  12.*) ;;
  *) die "this script is the CUDA 12 recipe; found CUDA $CUDART" ;;
esac
[ -f "$HOSTFILE" ] || die "hostfile $HOSTFILE not found"
mkdir -p "$K3_ROOT" "$WHEELS" "$K3_ROOT/cubin"
[ -n "${http_proxy:-}" ] || echo "   WARN: \$http_proxy unset — external downloads may fail"

# Passwordless ssh to every node is a hard requirement: step [12] verifies over
# ssh and launch_k3.sh --all starts the other 24 ranks the same way. Check it
# now rather than 40 minutes into the build.
NODES=$(awk 'NF{print $1}' "$HOSTFILE" | wc -l)
echo "   hostfile: $NODES node(s)"
while read -r h _; do
  [ -n "$h" ] || continue
  ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=8 -o BatchMode=yes "$h" true 2>/dev/null \
    || die "passwordless ssh to $h failed"
done < <(tail -n +2 "$HOSTFILE")
ok "ssh reachable on all $NODES nodes"

# uv is only used to create the venv; install it if missing.
if ! command -v uv >/dev/null 2>&1; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
UV="$(command -v uv)"
echo "   uv: $UV ($($UV --version))"

if [ "${VERIFY_ONLY:-0}" = "1" ]; then
  log "VERIFY_ONLY=1 — skipping to verification"
else

# --------------------------- 1. official sglang clone -------------------------
log "[1/12] official sglang @ $SGLANG_COMMIT"
if [ -d "$SG/.git" ] && [ "$(git -C "$SG" rev-parse HEAD)" = "$SGLANG_COMMIT" ]; then
  skip "clone present at pinned commit"
else
  [ -e "$SG" ] && die "$SG exists but is not the pinned clone; move it aside first"
  git clone --filter=blob:none --no-checkout "$SGLANG_REPO" "$SG"
  git -C "$SG" fetch --depth 1 origin "$SGLANG_COMMIT"
  git -C "$SG" checkout --detach "$SGLANG_COMMIT"
fi
for f in kimi_k3_cu12.Dockerfile apply_deepep_k3_patch.sh apply_deepgemm_situ_patch.py \
         flashinfer-perkz-dcp-0.6.15.txt; do
  [ -f "$SG/docker/kimi_k3/$f" ] || die "recipe file missing: docker/kimi_k3/$f"
done
ok "recipe files present"

# -------------------------------- 2. uv venv ----------------------------------
# --system-site-packages: torch 2.11.0+cu129, nvshmem, triton and the CUDA libs
# come from the image's dist-packages; we only layer the K3 deltas on top.
# NOTE: use $VENV/bin/pip, never `uv pip` — uv does not count system-site
# packages as satisfying a requirement and would pull a second torch.
log "[2/12] uv venv (system-site-packages) at $V"
if [ -x "$PY" ]; then
  skip "venv exists ($("$PY" -V 2>&1))"
else
  "$UV" venv --seed --python "$PY_SYS" --system-site-packages "$V"
fi
"$PY" -c 'import torch;assert torch.cuda.is_available();print("   torch",torch.__version__,"| gpus",torch.cuda.device_count())'

# ------------------------ 3. sglang editable install --------------------------
# SGLANG_BUILD_RUST_EXTS=none: no cargo/rustc on this box, and the Rust _core
# modules are only imported by the gRPC server and an opt-in preprocessor.
log "[3/12] sglang editable install"
if "$PY" -c 'import sglang,pathlib,sys; sys.exit(0 if str(pathlib.Path(sglang.__file__)).startswith("'"$SG"'") else 1)' 2>/dev/null; then
  skip "sglang already editable from $SG"
else
  "$PIP" install --no-deps "setuptools-scm==10.2.1" "setuptools-rust==1.13.0"
  SGLANG_BUILD_RUST_EXTS=none SETUPTOOLS_SCM_PRETEND_VERSION="$SGLANG_VERSION" \
    "$PIP" install -e "$SG/python" --no-deps --no-build-isolation
fi
ok "sglang $("$PY" -c 'import importlib.metadata as m;print(m.version("sglang"))')"
# --------------------- 4. +cu129 kernel wheels (GitHub only) -------------------
# PyPI / the internal mirror only carry the plain CUDA-13-linked builds of these
# two, which fail at import with `libcudart.so.13: cannot open shared object
# file`. The +cu129 variants exist ONLY as GitHub release assets.
log "[4/12] sgl-deep-gemm 0.1.5+cu129 and sglang-kernel 0.4.5+cu129"
fetch_verify() {   # $1=url $2=dest $3=sha256
  if [ -f "$2" ] && echo "$3  $2" | sha256sum --check --status -; then
    skip "$(basename "$2") cached"
  else
    curl -fL --retry 3 --retry-delay 5 -o "$2" "$1"
    echo "$3  $2" | sha256sum --check --strict -
  fi
}
fetch_verify "$DEEP_GEMM_URL"  "$WHEELS/$DEEP_GEMM_WHL"  "$DEEP_GEMM_SHA"
fetch_verify "$SGL_KERNEL_URL" "$WHEELS/$SGL_KERNEL_WHL" "$SGL_KERNEL_SHA"
need_wheel() { "$PY" - "$1" "$2" <<'EOF'
import importlib.metadata as m, sys
try:    sys.exit(0 if m.version(sys.argv[1]) != sys.argv[2] else 1)
except m.PackageNotFoundError: sys.exit(0)
EOF
}
if need_wheel sgl-deep-gemm "0.1.5+cu129"; then
  "$PIP" install --no-deps --force-reinstall "$WHEELS/$DEEP_GEMM_WHL"
else skip "sgl-deep-gemm 0.1.5+cu129"; fi
if need_wheel sglang-kernel "0.4.5+cu129"; then
  "$PIP" install --no-deps --force-reinstall "$WHEELS/$SGL_KERNEL_WHL"
else skip "sglang-kernel 0.4.5+cu129"; fi
# NB: the `sglang-kernel` distribution imports as `sgl_kernel`.
"$PY" -c 'import deep_gemm, sgl_kernel' || die "deep_gemm / sgl_kernel import failed"
ok "kernel wheels import clean"

# ------------------------- 5. python-side dependencies ------------------------
# Deliberately --no-deps and package-by-package: a plain `pip install -r` of the
# recipe's pins resolves an entire CUDA-13 stack (cuda-python 13.x, nvidia-*13.x,
# nvidia-nvshmem-cu13, nvidia-cutlass-dsl-libs-cu13) and would shadow the
# working CUDA 12.9 system libraries.
#
# nvidia-cutlass-dsl 4.6.0 is metadata + a .pth only; its content lives in the
# three -libs- packages. Omitting libs-core leaves `nvidia_cutlass_dsl_packages
# .pth` raising ModuleNotFoundError at every interpreter start.
log "[5/12] python dependencies (--no-deps, CUDA-12-safe)"
DEPS=(
  "nvidia-cutlass-dsl==4.6.0"
  "nvidia-cutlass-dsl-libs-base==4.6.0"
  "nvidia-cutlass-dsl-libs-cu12==4.6.0"
  "nvidia-cutlass-dsl-libs-core==4.6.0"
  "nvidia-mathdx==25.6.0"
  "apache-tvm-ffi==0.1.11"
  "llvmlite==0.47.0" "numba==0.65.1"
  "helion==0.2.6" "tilelang==0.1.11" "quack-kernels==0.6.1"
  "tokenspeed-mla==0.1.8" "tokenspeed-triton==3.7.10.post20260505"
  "xgrammar==0.2.1" "humming-kernels==0.1.10"
  "transformers==5.12.1" "flash-attn-4==4.0.0b19"
)
MISSING=()
for spec in "${DEPS[@]}"; do
  name="${spec%%==*}"; want="${spec##*==}"
  if need_wheel "$name" "$want"; then MISSING+=("$spec"); fi
done
if [ "${#MISSING[@]}" -eq 0 ]; then
  skip "all ${#DEPS[@]} dependencies at pinned versions"
else
  echo "   installing: ${MISSING[*]}"
  "$PIP" install --no-deps "${MISSING[@]}"
fi
"$PY" -c 'import cutlass, cutlass.cute, pathlib
p = pathlib.Path(cutlass.__file__)
assert "'"$V"'" in str(p), f"cutlass resolves outside the venv: {p}"
print("   cutlass", cutlass.__version__ if hasattr(cutlass,"__version__") else "", p)'
ok "cutlass-dsl 4.6.0 resolves inside the venv"
# ------------------------------ 6. FlashInfer trio ----------------------------
# All three must be the same version: a mixed Python/cubin/JIT-cache install
# fails at import. flashinfer-python and -cubin are CUDA-independent; the
# jit-cache wheel is CUDA-specific (cu129, ~2.1 GB — hence the long timeout).
log "[6/12] FlashInfer trio $FLASHINFER_VERSION (+cu129 jit-cache)"
fi_ok() { "$PY" - <<EOF
import importlib.metadata as m, sys
e = "$FLASHINFER_VERSION"
try:
    ok = (m.version("flashinfer-python").split("+")[0] == e
          and m.version("flashinfer-cubin").split("+")[0] == e
          and m.version("flashinfer-jit-cache").startswith(e + "+cu129"))
except m.PackageNotFoundError:
    ok = False
sys.exit(0 if ok else 1)
EOF
}
if fi_ok; then
  skip "FlashInfer trio at $FLASHINFER_VERSION"
else
  "$PIP" uninstall -y flashinfer-python flashinfer-cubin flashinfer-jit-cache || true
  rm -rf /root/.cache/flashinfer
  "$PIP" install --no-deps "flashinfer-python==$FLASHINFER_VERSION"
  "$PIP" install --no-deps "flashinfer-cubin==$FLASHINFER_VERSION" \
      --index-url https://flashinfer.ai/whl
  "$PIP" install --no-deps "flashinfer-jit-cache==$FLASHINFER_VERSION" \
      --index-url https://flashinfer.ai/whl/cu129
  fi_ok || die "FlashInfer trio versions do not agree"
fi
ok "flashinfer $("$PY" -c 'import importlib.metadata as m;print(m.version("flashinfer-python"),"/ jit-cache",m.version("flashinfer-jit-cache"))')"

# --------------- 7. FlashInfer CuTeDSL MLA DCP runtime patch ------------------
# 7 runtime files; the tail of the patch touches tests/ that the wheel does not
# ship, so cut it. --forward makes a re-run a no-op instead of a failure.
log "[7/12] FlashInfer MLA decode-context-parallel patch"
FI_PATCH="$SG/docker/kimi_k3/flashinfer-perkz-dcp-0.6.15.txt"
FI_SP=$("$PY" -c 'from pathlib import Path; import flashinfer; print(Path(flashinfer.__file__).resolve().parent.parent)')
if sed '/^diff --git a\/tests\//,$d' "$FI_PATCH" | \
     patch --dry-run --batch --forward --strip=1 --directory="$FI_SP" >/tmp/k3_fi_dry.log 2>&1; then
  sed '/^diff --git a\/tests\//,$d' "$FI_PATCH" | \
    patch --batch --forward --strip=1 --directory="$FI_SP"
  rm -rf /root/.cache/flashinfer
  ok "DCP patch applied"
elif grep -qi "previously applied\|Reversed" /tmp/k3_fi_dry.log; then
  skip "DCP patch already applied"
else
  cat /tmp/k3_fi_dry.log; die "FlashInfer DCP patch does not apply cleanly"
fi
"$PY" -c 'import flashinfer' || die "flashinfer import broken after patch"

# ------------------ 8. DeepGEMM mega-MoE SiTU header patch --------------------
# K3 uses SiTU (beta 4.0 / linear_beta 25.0) instead of SwiGLU. DeepGEMM
# JIT-compiles from headers, so this is a source patch with no rebuild. The
# upstream script hard-codes the system dist-packages path; redirect it at the
# venv copy. The anchor only exists in 0.1.5 -> must run AFTER step 4.
log "[8/12] DeepGEMM SiTU patch (sm100_fp8_fp4_mega_moe.cuh)"
DG=$("$PY" -c 'import deep_gemm,pathlib;print(pathlib.Path(deep_gemm.__file__).parent)')
DG_HDR="$DG/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh"
[ -f "$DG_HDR" ] || die "DeepGEMM header not found: $DG_HDR"
sed "s|^P = .*|P = \"$DG_HDR\"|" "$SG/docker/kimi_k3/apply_deepgemm_situ_patch.py" > /tmp/k3_situ_venv.py
"$PY" /tmp/k3_situ_venv.py
grep -q "kUseSitu" "$DG_HDR" || die "SiTU sentinel missing after patch"
ok "kUseSitu present ($(grep -c kUseSitu "$DG_HDR") sites)"

# ------------------- 9. trtllm-gen MoE cubin pool (sha256) -------------------
# Needed by the flashinfer_mxfp4 runner. We do not use that runner in the
# primary launch config, but its presence is what makes `--moe-runner-backend
# auto` select it — see the launch script's explicit --moe-runner-backend.
log "[9/12] trtllm-gen MoE cubin pool ($CUBIN_COUNT cubins)"
have=$(find "$CUBIN_POOL" -type f -name '*.cubin' 2>/dev/null | wc -l)
if [ "$have" -eq "$CUBIN_COUNT" ]; then
  skip "cubin pool complete ($have)"
else
  ZIP="$K3_ROOT/cubin/pool.zip"
  fetch_verify "$CUBIN_URL" "$ZIP" "$CUBIN_SHA"
  rm -rf "$K3_ROOT/cubin/extract" "$CUBIN_POOL"
  mkdir -p "$K3_ROOT/cubin/extract"
  unzip -q "$ZIP" -d "$K3_ROOT/cubin/extract"
  mv "$K3_ROOT/cubin/extract/$CUBIN_ROOT" "$CUBIN_POOL"
  rm -rf "$K3_ROOT/cubin/extract" "$ZIP"
  have=$(find "$CUBIN_POOL" -type f -name '*.cubin' | wc -l)
  [ "$have" -eq "$CUBIN_COUNT" ] || die "expected $CUBIN_COUNT cubins, found $have"
fi
ok "$CUBIN_POOL"
# ---------------------- 10. DeepEP: source at the pinned commit ---------------
# The K3 Dockerfile pins d28bd67 (origin/hybrid-ep), which builds BOTH
# deep_ep_cpp and hybrid_ep_cpp. The commonly-installed 92fe2de lineage ships
# deep_ep_cpp only, and its cubins are sm_90 — unusable on this SM100 cluster.
log "[10/12] DeepEP source @ $DEEPEP_COMMIT"
if [ -f "$DEEPEP_DIR/csrc/kernels/internode_ll.cu" ]; then
  skip "DeepEP source tree present"
else
  mkdir -p "$DEEPEP_DIR"
  if [ -d "$DEEPEP_LOCAL_MIRROR/.git" ] && \
     git -C "$DEEPEP_LOCAL_MIRROR" cat-file -e "$DEEPEP_COMMIT^{commit}" 2>/dev/null; then
    echo "   using local mirror $DEEPEP_LOCAL_MIRROR"
    git -C "$DEEPEP_LOCAL_MIRROR" archive "$DEEPEP_COMMIT" | tar -x -C "$DEEPEP_DIR"
  else
    git clone "$DEEPEP_REPO" "$DEEPEP_DIR.git-tmp"
    git -C "$DEEPEP_DIR.git-tmp" checkout --detach "$DEEPEP_COMMIT"
    git -C "$DEEPEP_DIR.git-tmp" archive "$DEEPEP_COMMIT" | tar -x -C "$DEEPEP_DIR"
    rm -rf "$DEEPEP_DIR.git-tmp"
  fi
fi

# ----------------- 11. DeepEP: K3 patches + CUDA-12 guard + build -------------
# Upstream's apply_deepep_k3_patch.sh does steps 1..5b then builds and verifies
# against the SYSTEM dist-packages path, which does not apply here (venv install,
# egg-style layout). So: run only the source patches, then build/install/verify
# ourselves.
#
# Extra local change (not upstream): d28bd67's SETUP_OVERLAP_LAUNCH_CONFIG uses
# cudaLaunchAttributeNvlinkUtilCentricScheduling, a CUDA 13 addition, so the
# stock tree does not compile under CUDA 12.9. The guard below sets the
# attribute only on CUDA >= 13. Dropping it on CUDA 12 loses an NVLink
# *scheduling hint*, not correctness; the macro is reachable only from
# internode_ll.cu's topk_weights!=nullptr branch, i.e. SBO combine-overlap.
log "[11/12] DeepEP K3 patches + rebuild for TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
LAUNCH_CUH="$DEEPEP_DIR/csrc/kernels/launch.cuh"
if grep -q "K3_SET_NVLINK_UTIL_CENTRIC" "$LAUNCH_CUH"; then
  skip "CUDA-12 NVLink-attribute guard already present"
else
  "$PY" - "$LAUNCH_CUH" <<'EOF'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
guard = '''// K3/CUDA12: `cudaLaunchAttributeNvlinkUtilCentricScheduling` is a CUDA 13
// addition. Under CUDA 12.x the attribute simply is not set (the kernel keeps
// the default NVLink scheduling policy); under CUDA 13+ behaviour is unchanged.
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
#define K3_SET_NVLINK_UTIL_CENTRIC(cfg, attr) do { \\
    (attr)[1].id = cudaLaunchAttributeNvlinkUtilCentricScheduling; \\
    (attr)[1].val.nvlinkUtilCentricScheduling = 1; \\
    (cfg).numAttrs = 2; \\
} while (0)
#else
#define K3_SET_NVLINK_UTIL_CENTRIC(cfg, attr) do { } while (0)
#endif

#ifndef SETUP_OVERLAP_LAUNCH_CONFIG'''
anchor = "#ifndef SETUP_OVERLAP_LAUNCH_CONFIG"
assert s.count(anchor) == 1, f"unexpected anchor count: {s.count(anchor)}"
s = s.replace(anchor, guard, 1)
old_tail = """    cfg.attrs = attr; \\
    attr[1].id = cudaLaunchAttributeNvlinkUtilCentricScheduling; \\
    attr[1].val.nvlinkUtilCentricScheduling = 1; \\
    cfg.numAttrs = 2;"""
new_tail = """    cfg.attrs = attr; \\
    cfg.numAttrs = 1; \\
    K3_SET_NVLINK_UTIL_CENTRIC(cfg, attr);"""
assert s.count(old_tail) == 1, f"unexpected overlap-config tail: {s.count(old_tail)}"
p.write_text(s.replace(old_tail, new_tail, 1))
print("   guard inserted")
EOF
fi

# Steps 1..5b of the official patcher (topk 9/11->16, SWITCH_HIDDEN += 3584,
# timeouts 100s->1000s, nvfp4 test tolerance, SourceMeta 4-byte alignment for
# EP>8, setup.py cccl include). All are grep/count-guarded upstream. Step 6 is
# cut off because it builds and then verifies against the system dist-packages
# path, which does not exist in a venv install.
grep -q '^echo "== \[6/6\] rebuild' "$SG/docker/kimi_k3/apply_deepep_k3_patch.sh" \
  || die "apply_deepep_k3_patch.sh layout changed: step [6/6] marker not found"
awk '/^echo "== \[6\/6\] rebuild/{exit} {print}' \
    "$SG/docker/kimi_k3/apply_deepep_k3_patch.sh" > /tmp/k3_deepep_patch_only.sh
DEEPEP_DIR="$DEEPEP_DIR" TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  bash /tmp/k3_deepep_patch_only.sh

SM="sm_${TORCH_CUDA_ARCH_LIST//./}"
deepep_built_for_arch() {
  local so
  so=$("$PY" -c 'import importlib.util as u;s=u.find_spec("deep_ep_cpp");print(s.origin if s else "")' 2>/dev/null) || return 1
  [ -n "$so" ] || return 1
  cuobjdump --list-elf "$so" 2>/dev/null | grep -Eq "(^|[^[:alnum:]_])${SM}([^[:alnum:]_]|\$)"
}
if deepep_built_for_arch; then
  skip "deep_ep already installed with $SM cubins"
else
  ( cd "$DEEPEP_DIR" && rm -rf build dist && \
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" "$PY" setup.py bdist_wheel )
  "$PIP" install --force-reinstall --no-deps "$DEEPEP_DIR"/dist/deep_ep-*.whl
  deepep_built_for_arch || die "installed deep_ep_cpp has no $SM cubin"
fi
for mod in deep_ep_cpp hybrid_ep_cpp; do
  "$PY" -c "import importlib.util as u,sys;sys.exit(0 if u.find_spec('$mod') else 1)" \
    || die "$mod not importable (wrong DeepEP lineage?)"
done
ok "deep_ep $("$PY" -c 'import importlib.metadata as m;print(m.version("deep_ep"))') with $SM cubins, both extensions present"

fi   # end of build phase (VERIFY_ONLY)
# --------------------------- 12. verify on all 4 nodes ------------------------
# Everything above lives on shared GPFS, so the remotes get it for free. This
# proves it: same venv, same .so, same patched headers, from every node.
log "[12/12] verifying on every node in $HOSTFILE"
cat > "$K3_ROOT/k3_verify.py" <<'EOF'
import importlib.metadata as md, importlib.util, pathlib, socket
import deep_ep, flashinfer, deep_gemm, sglang

print(socket.gethostname())
print("  sglang       ", sglang.__file__)
print("  deep_ep_cpp  ", importlib.util.find_spec("deep_ep_cpp").origin.split("/")[-1])
print("  hybrid_ep_cpp", importlib.util.find_spec("hybrid_ep_cpp").origin.split("/")[-1])
print("  flashinfer   ", md.version("flashinfer-python"),
      "| jit-cache", md.version("flashinfer-jit-cache"))
print("  deep_gemm    ", md.version("sgl-deep-gemm"),
      "| sglang-kernel", md.version("sglang-kernel"),
      "| transformers", md.version("transformers"))
h = (pathlib.Path(deep_gemm.__file__).parent
     / "include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh")
print("  situ patched:", "kUseSitu" in h.read_text())
EOF

FAIL=0
"$PY" "$K3_ROOT/k3_verify.py" || FAIL=1
# ssh -n: without it ssh swallows the loop's stdin and only the first host runs.
while read -r host _; do
  [ -n "$host" ] || continue
  ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=8 "$host" \
    "$PY $K3_ROOT/k3_verify.py" || { echo "   FAILED on $host"; FAIL=1; }
done < <(tail -n +2 "$HOSTFILE")
[ "$FAIL" -eq 0 ] || die "verification failed on at least one node"

cat <<EOF

$(printf '\033[1;32m')Environment ready.$(printf '\033[0m')
  root      $K3_ROOT
  venv      $V          (python $("$PY" -V 2>&1 | awk '{print $2}'))
  sglang    $SG @ ${SGLANG_COMMIT:0:12}   (official upstream)
  serve     $V/bin/sglang serve ...
  cubins    $CUBIN_POOL

Next: bash $K3_ROOT/launch_k3.sh --all
EOF
