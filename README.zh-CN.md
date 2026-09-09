# DAG：从零环境构建、模型下载与训练

[English](./README.md) | [中文](./README.zh-CN.md)

本文档适用于从空 Conda 环境开始运行 `diffusion_affordance` 中的 DAG 模型。

推荐系统：Linux + NVIDIA GPU。当前已验证的运行组合为 Python 3.9、PyTorch 1.13.1、torchvision 0.14.1 和 CUDA 11.x。

本方案只使用 Conda 创建 Python 环境，不使用 Conda 安装 Python 包。后续依赖统一使用 pip 安装，避免 Conda 与 pip 混装 CUDA/PyTorch 二进制包。

> 说明：本文所有相对路径均相对于项目根目录。模型权重统一放在项目内的 `ckpt/` 目录下。Detectron2 等外部依赖假设位于项目根目录的父目录（即 `../detectron2-main`），如实际目录结构不同请自行调整。

## 1. 创建 Conda 环境

```bash
conda create -n DAG python=3.9 -y
conda activate DAG
```

确认系统编译工具和 CUDA 工具链已经存在：

```bash
command -v gcc
command -v g++
command -v nvcc
```

如果其中某个命令不存在，需要由系统管理员安装系统级编译工具和 CUDA Toolkit；不建议为了补系统工具再向 Conda 环境安装另一套编译器。

确认 Python：

```bash
python --version
```

应显示 `Python 3.9.x`。

## 2. 使用 pip 安装 PyTorch 和 CUDA wheel

不要先安装其他会依赖 PyTorch 的 CUDA 扩展，先固定 PyTorch 版本：

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==1.13.1+cu117 \
  torchvision==0.14.1+cu117 \
  torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu117
```

检查 CUDA：

```bash
python - <<'PY'
import torch
import torchvision

print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('cuda_available:', torch.cuda.is_available())
print('gpu_count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('gpu_name:', torch.cuda.get_device_name(0))
PY
```

## 3. 安装基础 Python 包

```bash
python -m pip install \
  numpy==1.24.1 \
  scipy==1.10.0 \
  pandas==2.0.3 \
  scikit-learn==1.2.0 \
  pyyaml==6.0.1 \
  pillow==10.2.0 \
  opencv-python \
  matplotlib==3.6.3 \
  tqdm==4.66.2 \
  einops==0.7.0 \
  timm==0.9.16 \
  tensorboard==2.14.0 \
  tensorboardX==2.6.2.2 \
  plyfile \
  addict \
  ftfy \
  regex \
  safetensors
```

## 4. 安装 CLIP 和 Transformers

```bash
python -m pip install \
  transformers==4.26.1 \
  huggingface-hub==0.21.4 \
  sentencepiece==0.2.0 \
  open-clip-torch==2.24.0
```

安装项目内的 CLIP：

```bash
cd CLIP1
python -m pip install -e .
cd ..
```

## 5. 安装 ODISE、LDM 和 Detectron2 依赖

DAG 的图像分支使用 ODISE/LDM，实际需要 `ldm`、`omegaconf`、`fvcore`、`iopath` 和 Detectron2：

```bash
python -m pip install \
  stable-diffusion-sdkit==2.1.3 \
  omegaconf==2.1.1 \
  fvcore==0.1.5.post20221221 \
  iopath==0.1.9 \
  hydra-core==1.3.2
```

当前项目使用本地 Detectron2 源码。安装前确保目录存在：

```bash
cd ../detectron2-main
pip install -e .
cd ../diffusion_affordance
```

如果 Detectron2 编译失败，先确认：

```bash
which gcc
which g++
which nvcc
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

## 6. 安装点云 CUDA 扩展

```bash
python -m pip install KNN-CUDA==0.2
python -m pip install pointnet2-ops==3.0.0
```

检查：

```bash
python - <<'PY'
import knn_cuda
import pointnet2_ops
print('knn_cuda: ok')
print('pointnet2_ops: ok')
PY
```

首次运行可能出现：

```text
Unable to load pointnet2_ops cpp extension. JIT Compiling.
```

这是首次 JIT 编译提示，通常不是错误。

## 7. 安装其他运行依赖

```bash
python -m pip install \
  open3d==0.18.0 \
  openexr==1.3.9 \
  mitsuba==3.0.1 \
  modelscope
```

如果 `openexr` 安装失败，先由系统管理员安装 OpenEXR 开发库，再重新执行 pip：

```bash
sudo apt-get install libopenexr-dev
python -m pip install openexr
```

## 8. 核心模块检查

在项目根目录运行：

```bash
python - <<'PY'
import torch
import torchvision
import transformers
import ldm
import detectron2
import fvcore
import iopath
import xformers
import knn_cuda
import pointnet2_ops
import open3d
import timm
import tensorboardX

print('all core imports: ok')
print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('cuda:', torch.cuda.is_available())
PY
```

## 9. 下载 Stable Diffusion v1.5

DAG 的 ODISE/LDM 会读取以下文件：

```text
ckpt/sd/v1-5-pruned-emaonly.ckpt
```

使用 ModelScope 下载：

```bash
mkdir -p ckpt/sd

modelscope download \
  --model AI-ModelScope/stable-diffusion-v1-5 \
  --include v1-5-pruned-emaonly.ckpt \
  --local_dir ckpt/sd
```

## 10. 下载 CLIP 权重

DAG 使用 OpenCLIP 的 `ViT-L-14-336` 模型（OpenAI 权重）。代码从以下路径加载：

```text
ckpt/clip/ViT-L-14-336px.pt
```

将权重下载到 `ckpt/clip/`：

```bash
mkdir -p ckpt/clip

wget -O ckpt/clip/ViT-L-14-336px.pt \
  https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt
```

这是 OpenCLIP 使用的 OpenAI CLIP `ViT-L/14@336px` checkpoint。

## 11. 放置 Uni3D 权重
下载uni3d base模型：https://huggingface.co/BAAI/Uni3D/blob/main/modelzoo/uni3d-b/model.pt
DAG 当前代码读取：

```text
ckpt/uni3d.pt
```

如果文件在其他位置：

```bash
cp /path/to/uni3d.pt ckpt/uni3d.pt
```

检查 checkpoint：

```bash
python - <<'PY'
import torch

path = 'ckpt/uni3d.pt'
checkpoint = torch.load(path, map_location='cpu')
print('checkpoint_keys:', list(checkpoint.keys())[:10])
print('module_parameters:', len(checkpoint['module']))
PY
```

Uni3D 参数会在 DAG 训练中被冻结，不参与更新。

## 12. ODISE 权重

该文件应已包含在项目内：

```text
model/odise/feature_extractor.pth
```


## 13. 检查所有模型文件

```bash
test -f ckpt/sd/v1-5-pruned-emaonly.ckpt && echo sd_ok

test -f model/odise/feature_extractor.pth && echo odise_ok

test -f ckpt/uni3d.pt && echo uni3d_ok

test -f ckpt/clip/ViT-L-14-336px.pt && echo clip_ok
```

预期输出：

```text
sd_ok
odise_ok
uni3d_ok
clip_ok
```

## 14. 检查数据集

Dataset：https://drive.google.com/drive/folders/1F242TsdXjRZkKQotiBsiN2u6rJAGRZ2W (PIAD), https://drive.google.com/drive/folders/1n_L_mSmVpAM-1ASoW2T2MltYkaiA_X9X (PIAD2)

默认配置文件为：

```text
config/config_seen.yaml
```

Seen 数据集需要以下文件：

```text
Data/Seen/Img_Train.txt
Data/Seen/Img_Test.txt
Data/Seen/Point_Train.txt
Data/Seen/Point_Test.txt
Data/Seen/Box_Train.txt
Data/Seen/Box_Test.txt
Data/Seen/affordance_json_paths.txt
Data/Seen/affordance_json_test_paths.txt
```

检查：

```bash
find Data/Seen -maxdepth 1 -type f -print
```

## 15. 训练 Smoke Test

24 GB 显存下建议使用：

```yaml
batch_size: 1
Epoch: 1
```

可以直接临时复制配置，不修改原配置：

```bash
cp config/config_seen.yaml config/config_dag_smoke.yaml
sed -i 's/^batch_size:.*/batch_size: 1/' config/config_dag_smoke.yaml
sed -i 's/^Epoch:.*/Epoch: 1/' config/config_dag_smoke.yaml
```

运行：

```bash
conda activate DAG

python train_for_DAG.py \
  --name DAG_smoke_test \
  --use_gpu True \
  --yaml config/config_dag_smoke.yaml
```

看到以下输出，说明已经完成模型初始化并进入第一个训练 batch：

```text
Start loading train data---
train data loading finish, loading data files:4150
Start loading val data---
val data loading finish, loading data files:1012
Epoch:0 strat-------
Epoch:0 | iteration:0 | loss:...
```

完整 epoch 结束后，应看到 `EVALUATION`。输出目录为：

```text
runs/train/DAG_smoke_test/
```

结束后可删除临时 smoke-test 配置（`config/config_dag_smoke.yaml`）。

## 16. 正式训练

确认 Smoke Test 正常后，将配置设置为需要的训练轮数，例如：

```yaml
batch_size: 16
Epoch: 100
```

运行：

```bash
conda activate DAG

python train_for_DAG.py \
  --name DAG \
  --use_gpu True \
  --yaml config/config_seen.yaml
```

训练结果写入：

```text
runs/train/DAG/
```

## 17. 常见问题

### CUDA out of memory

将 `batch_size` 降为 `1`，并关闭其他 GPU 进程：

```bash
nvidia-smi
```

### `pointnet2_ops` JIT 编译提示

首次启动时正常，等待编译完成即可。

### `No module named triton`

这是 xformers 性能警告，通常不影响正确性。

### `Checkpoint not found`

检查是否从项目根目录运行，并确认权重路径与本文档完全一致。

### Detectron2 编译失败

优先使用 `../detectron2-main` 的本地源码安装，不要混用其他版本：

```bash
cd ../detectron2-main
python -m pip install -e .
cd ../diffusion_affordance
```

## 18. 版本基线

当前已验证的核心版本：

```text
Python 3.9.19
PyTorch 1.13.1+cu116
torchvision 0.14.1+cu116
CUDA available: True
```

不要在安装点云扩展和 xformers 后再升级 PyTorch，否则可能造成 CUDA 二进制不兼容。
