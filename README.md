# ECCV2026 DAG: Environment Setup, Model Download and Training

[English](./README.md) | [中文](./README.zh-CN.md)

This document covers running the DAG model in `diffusion_affordance` from an empty Conda environment.

Recommended system: Linux + NVIDIA GPU. The verified combination is Python 3.9, PyTorch 1.13.1, torchvision 0.14.1 and CUDA 11.x.

This guide uses Conda only to create the Python environment, not to install Python packages. All dependencies are installed with pip to avoid mixing Conda and pip CUDA/PyTorch binaries.

> Note: All relative paths in this document are relative to the project root directory. External dependencies such as Detectron2 and cached model weights are assumed to live under the parent directory of the project (i.e., `../detectron2-main`, `../autodl-tmp`). Adjust them if your directory layout differs.

## 1. Create the Conda Environment

```bash
conda create -n DAG python=3.9 -y
conda activate DAG
```

Make sure the system build tools and CUDA toolchain exist:

```bash
command -v gcc
command -v g++
command -v nvcc
```

If any command is missing, have the system administrator install the system-level build tools and CUDA Toolkit. Do not install a second toolchain into the Conda environment just to fill in system tools.

Verify Python:

```bash
python --version
```

It should print `Python 3.9.x`.

## 2. Install PyTorch and CUDA Wheels with pip

Do not install other CUDA extensions that depend on PyTorch first. Pin the PyTorch version first:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==1.13.1+cu117 \
  torchvision==0.14.1+cu117 \
  torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu117
```

Check CUDA:

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

## 3. Install Basic Python Packages

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

## 4. Install CLIP and Transformers

```bash
python -m pip install \
  transformers==4.26.1 \
  huggingface-hub==0.21.4 \
  sentencepiece==0.2.0 \
  open-clip-torch==2.24.0
```

Install the in-project CLIP:

```bash
cd CLIP1
python -m pip install -e .
cd ..
```

## 5. Install ODISE, LDM and Detectron2 Dependencies

The image branch of DAG uses ODISE/LDM and needs `ldm`, `omegaconf`, `fvcore`, `iopath` and Detectron2:

```bash
python -m pip install \
  stable-diffusion-sdkit==2.1.3 \
  omegaconf==2.1.1 \
  fvcore==0.1.5.post20221221 \
  iopath==0.1.9 \
  hydra-core==1.3.2
```

This project uses a local Detectron2 source tree. Make sure the directory exists before installing:

```bash
cd ../detectron2-main
pip install -e .
cd ../diffusion_affordance
```

If Detectron2 fails to build, first check:

```bash
which gcc
which g++
which nvcc
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

## 6. Install Point Cloud CUDA Extensions

```bash
python -m pip install KNN-CUDA==0.2
python -m pip install pointnet2-ops==3.0.0
```

Check:

```bash
python - <<'PY'
import knn_cuda
import pointnet2_ops
print('knn_cuda: ok')
print('pointnet2_ops: ok')
PY
```

On the first run you may see:

```text
Unable to load pointnet2_ops cpp extension. JIT Compiling.
```

This is the first-time JIT compilation message and is usually not an error.

## 7. Install Other Runtime Dependencies

```bash
python -m pip install \
  open3d==0.18.0 \
  openexr==1.3.9 \
  mitsuba==3.0.1 \
  modelscope
```

If `openexr` fails to install, have the system administrator install the OpenEXR development library first, then run pip again:

```bash
sudo apt-get install libopenexr-dev
python -m pip install openexr
```

## 8. Core Module Check

Run from the project root directory:

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

## 9. Download Stable Diffusion v1.5

The ODISE/LDM branch of DAG reads the following file:

```text
../autodl-tmp/sd/v1-5-pruned-emaonly.ckpt
```

Download it with ModelScope:

```bash
mkdir -p ../autodl-tmp/sd

modelscope download \
  --model AI-ModelScope/stable-diffusion-v1-5 \
  --include v1-5-pruned-emaonly.ckpt \
  --local_dir ../autodl-tmp/sd
```

## 10. Download CLIP ViT-L/14

DAG currently uses a fixed local directory:

```text
../autodl-tmp/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41
```

Download:

```bash
mkdir -p ../autodl-tmp/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41

modelscope download \
  --model AI-ModelScope/clip-vit-large-patch14 \
  --local_dir ../autodl-tmp/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41
```

The directory should contain at least:

```text
config.json
tokenizer_config.json
tokenizer.json
vocab.json
merges.txt
pytorch_model.bin or model.safetensors
```

## 11. Place the Uni3D Weights

Download the Uni3D base model: https://huggingface.co/BAAI/Uni3D/blob/main/modelzoo/uni3d-b/model.pt
The current DAG code reads:

```text
ckpt/uni3d.pt
```

If the file is somewhere else:

```bash
cp /path/to/uni3d.pt ckpt/uni3d.pt
```

Check the checkpoint:

```bash
python - <<'PY'
import torch

path = 'ckpt/uni3d.pt'
checkpoint = torch.load(path, map_location='cpu')
print('checkpoint_keys:', list(checkpoint.keys())[:10])
print('module_parameters:', len(checkpoint['module']))
PY
```

Uni3D parameters are frozen during DAG training and do not receive updates.

## 12. ODISE Weights

This file should already be included in the project:

```text
model/odise/feature_extractor.pth
```

LongCLIP has been removed from the DAG main path, so `longclip-L.pt` is not required.

## 13. Check All Model Files

```bash
test -f ckpt/sd/v1-5-pruned-emaonly.ckpt && echo sd_ok

test -f model/odise/feature_extractor.pth && echo odise_ok

test -f ckpt/uni3d.pt && echo uni3d_ok

test -f ckpt/clip/ViT-L-14-336px.pt && echo clip_ok
```

Expected output:

```text
sd_ok
odise_ok
uni3d_ok
clip_ok
```

## 14. Check the Dataset
Dataset：https://drive.google.com/drive/folders/1F242TsdXjRZkKQotiBsiN2u6rJAGRZ2W (PIAD), https://drive.google.com/drive/folders/1n_L_mSmVpAM-1ASoW2T2MltYkaiA_X9X (PIAD2)

The default config file is:

```text
config/config_seen.yaml
```

The Seen dataset requires the following files:

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

Check:

```bash
find Data/Seen -maxdepth 1 -type f -print
```

## 15. Training Smoke Test

For a 24 GB GPU, it is recommended to use:

```yaml
batch_size: 1
Epoch: 1
```

You can copy the config to a temporary file without modifying the original config:

```bash
cp config/config_seen.yaml config/config_dag_smoke.yaml
sed -i 's/^batch_size:.*/batch_size: 1/' config/config_dag_smoke.yaml
sed -i 's/^Epoch:.*/Epoch: 1/' config/config_dag_smoke.yaml
```

Run:

```bash
conda activate DAG

python train_for_DAG.py \
  --name DAG_smoke_test \
  --use_gpu True \
  --yaml config/config_dag_smoke.yaml
```

Seeing the output below means the model has finished initialization and entered the first training batch:

```text
Start loading train data---
train data loading finish, loading data files:4150
Start loading val data---
val data loading finish, loading data files:1012
Epoch:0 strat-------
Epoch:0 | iteration:0 | loss:...
```

After a full epoch you should see `EVALUATION`. The output directory is:

```text
runs/train/DAG_smoke_test/
```

You can delete the temporary smoke-test config (`config/config_dag_smoke.yaml`) afterwards.

## 16. Full Training

After the smoke test passes, set the config to the desired number of epochs, for example:

```yaml
batch_size: 16
Epoch: 100
```

Run:

```bash
conda activate DAG

python train_for_DAG.py \
  --name DAG \
  --use_gpu True \
  --yaml config/config_seen.yaml
```

Training results are written to:

```text
runs/train/DAG/
```

## 17. FAQ

### CUDA out of memory

Lower `batch_size` to `1` and stop other GPU processes:

```bash
nvidia-smi
```

### `pointnet2_ops` JIT compilation message

This is normal on first startup; wait for the compilation to finish.

### `No module named triton`

This is an xformers performance warning and usually does not affect correctness.

### `Checkpoint not found`

Make sure you run from the project root directory and that the weight paths match this document exactly.

### Detectron2 build failure

Prefer installing from the local source at `../detectron2-main` and do not mix other versions:

```bash
cd ../detectron2-main
python -m pip install -e .
cd ../diffusion_affordance
```

## 18. Version Baseline

Verified core versions:

```text
Python 3.9.19
PyTorch 1.13.1+cu116
torchvision 0.14.1+cu116
CUDA available: True
```

Do not upgrade PyTorch after installing the point cloud extensions and xformers, otherwise CUDA binaries may become incompatible.
