# useful-scripts

日常研发中积累的实用脚本集合。

## 脚本索引

### GPU

| 脚本 | 说明 |
|------|------|
| [`gpu/gpu_stress_test.py`](gpu/gpu_stress_test.py) | torch + DDP 多卡压测，三种模式：`compute`（纯矩阵乘打满算力）、`mixed`（前向+反向+优化器，模拟训练）、`memory`（占满显存）。死循环运行，Ctrl+C 停止。 |
| [`gpu/gpu_burn.py`](gpu/gpu_burn.py) | GPU 打满工具。支持按卡号选择、控制目标利用率（占空比）、按比例占用显存、限时运行。 |

### Shell 环境

| 脚本 | 说明 |
|------|------|
| [`shell/setup_env_final.sh`](shell/setup_env_final.sh) | 一键配置 zsh + oh-my-zsh。核心是把通用配置抽到 `~/.shell_common`，实现 bashrc / zshrc 安全隔离：在 zsh 里误 `source ~/.bashrc` 不会污染 prompt，同时保留原 bashrc 里的 PATH（如 `ducc`）。附带 `pcp` 并行复制函数。 |

### 存储

| 脚本 | 说明 |
|------|------|
| [`storage/afs_mount.sh`](storage/afs_mount.sh) | 从 irepo 下载 afs_mount 客户端并挂载 AFS，带日志与幂等检查。 |

### 训练

| 脚本 | 说明 |
|------|------|
| [`training/lite_sft_pipeline.sh`](training/lite_sft_pipeline.sh) | Lite SFT 一条龙：拉模型/数据 → 生成 env（自动算步数）→ 分发起训 → 多节点自检 → 起训 → 看状态。权重/数据落本地 NVMe，内网 HTTP 并行拉取。 |

### SGLang 推理

| 脚本 | 说明 |
|------|------|
| [`sglang/setup_k3_env.sh`](sglang/setup_k3_env.sh) | Kimi-K3（1.42 TiB / mxfp4）SGLang 原生环境构建，4 节点 × 8 卡 SM100 + CUDA 12.9，不走 Docker。幂等，可重复执行。 |
| [`sglang/launch_k3.sh`](sglang/launch_k3.sh) | Kimi-K3 多节点启动/停止。`--all` 经 ssh 扇出到所有节点并等待就绪，`--stop` 按进程组彻底清理占卡进程。 |

## 使用说明

### 需要自行提供环境变量的脚本

为避免把凭据写进仓库，以下脚本的敏感信息改为从环境变量读取，**运行前必须先 export**：

```bash
# storage/afs_mount.sh
export IREPO_TOKEN=<irepo token>
export AFS_USERNAME=<afs 用户名>
export AFS_PASSWORD=<afs 密码>

# shell/setup_env_final.sh
export http_proxy=http://user:pass@host:port
export https_proxy=http://user:pass@host:port
```

### 环境依赖

- GPU 脚本：`torch`（含 CUDA）
- `shell/setup_env_final.sh`：`zsh`、`git`、`curl`、`rsync`、`coreutils`（脚本会自动安装）
- `training/lite_sft_pipeline.sh`：`orterun`(openmpi)、`curl`、`rsync`
- `sglang/*`：`uv`、`nvcc`、`cuobjdump`，需配置免密 ssh 到 hostfile 中所有节点

## 说明

- 脚本中的内网地址、路径、主机名等均为特定环境配置，按需修改。
- 所有脚本均已在对应场景实际使用过，改动前建议先读懂注释里的「踩坑记录」。
