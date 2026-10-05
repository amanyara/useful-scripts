#!/usr/bin/env bash
set -Eeuo pipefail

# =========================
# 固定配置
# =========================

BASE_DIR="/root/paddlejob/workspace/env_run/afs_mount_xh_gcc82"
CLIENT_TAR="${BASE_DIR}/output.tar.gz"
CLIENT_DIR="${BASE_DIR}/output"
AFS_MOUNT_BIN="${CLIENT_DIR}/bin/afs_mount"

# irepo 下载地址和 token（token 通过环境变量传入，勿硬编码）
IREPO_URL="${IREPO_URL:-https://irepo.baidu-int.com/rest/prod/v3/baidu/inf/afs-api/releases/2.0.9.4397/files}"
IREPO_TOKEN="${IREPO_TOKEN:?请先 export IREPO_TOKEN=<irepo token>}"

# AFS 挂载配置（账号密码通过环境变量传入，勿硬编码）
MOUNT_POINT="${MOUNT_POINT:-/home/slurm/data/AMU_SAFE_DATA/}"
AFS_URI="${AFS_URI:-afs://xh-yq-fenhe.afs.baidu.com:8177/user/safe-data-2026-2}"
AFS_USERNAME="${AFS_USERNAME:?请先 export AFS_USERNAME=<afs 用户名>}"
AFS_PASSWORD="${AFS_PASSWORD:?请先 export AFS_PASSWORD=<afs 密码>}"

LOG_FILE="${BASE_DIR}/afs_mount.log"

# =========================
# 工具函数
# =========================

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "${LOG_FILE}"
}

die() {
    log "ERROR: $*"
    exit 1
}

is_mounted() {
    mountpoint -q "${MOUNT_POINT}" 2>/dev/null
}

prepare_dirs() {
    mkdir -p "${BASE_DIR}"
    mkdir -p "${MOUNT_POINT}"
    touch "${LOG_FILE}"
}

download_client_if_needed() {
    if [[ -x "${AFS_MOUNT_BIN}" ]]; then
        log "afs_mount client already exists: ${AFS_MOUNT_BIN}"
        return 0
    fi

    log "downloading afs_mount client..."

    cd "${BASE_DIR}"

    wget -O "${CLIENT_TAR}" \
        --no-check-certificate \
        --header "IREPO-TOKEN:${IREPO_TOKEN}" \
        "${IREPO_URL}"

    log "extracting afs_mount client..."

    tar -xf "${CLIENT_TAR}" -C "${BASE_DIR}"

    [[ -x "${AFS_MOUNT_BIN}" ]] || die "afs_mount binary not found: ${AFS_MOUNT_BIN}"

    log "afs_mount client ready: ${AFS_MOUNT_BIN}"
}

mount_afs() {
    if is_mounted; then
        log "mount point already mounted: ${MOUNT_POINT}"
        log "skip mounting"
        exit 0
    fi

    log "mounting AFS..."
    log "local mount point: ${MOUNT_POINT}"
    log "remote afs uri: ${AFS_URI}"
    log "username: ${AFS_USERNAME}"

    cd "${CLIENT_DIR}"

    nohup "${AFS_MOUNT_BIN}" \
        --username="${AFS_USERNAME}" \
        --password="${AFS_PASSWORD}" \
        "${MOUNT_POINT}" \
        "${AFS_URI}" >> "${LOG_FILE}" 2>&1 &

    local pid=$!
    log "afs_mount started, pid=${pid}"

    for i in {1..15}; do
        sleep 1

        if is_mounted; then
            log "AFS mounted successfully"
            log "checking mount point content..."
            ls -lh "${MOUNT_POINT}" | head -20 | tee -a "${LOG_FILE}" || true
            return 0
        fi
    done

    log "mount command started, but mountpoint check did not pass"
    log "please check log: ${LOG_FILE}"
    pgrep -af "afs_mount.*${MOUNT_POINT}" || true
}

# =========================
# 主流程
# =========================

prepare_dirs
download_client_if_needed
mount_afs
