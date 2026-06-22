# Environment and Dependencies

This project is centered on `maintrain_method2_relative_lr.py`.

## Runtime

- Python 3.8+
- NVIDIA GPU recommended
- CUDA-compatible PyTorch build
- Windows or Linux

The script defaults to:

```text
CUDA_VISIBLE_DEVICES=0,1
```

Adjust this value in the script or in your runtime environment if your GPU ids differ.

## Python Packages

Install dependencies with:

```bash
pip install -r requirements.txt
```

Main packages:

```text
torch
torchvision
numpy
Pillow
tqdm
pyiqa
ftfy
regex
```

## CLIP Assets

The default CLIP model is:

```text
clip_model/ViT-B-32.pt
```

The project also expects the local `CLIP/` package to be available because `clip_score.py` imports it.

## Data Layout

Default training layout:

```text
data/
+-- raw/
+-- reference/
+-- splits/
    +-- green.txt
    +-- blue.txt
    +-- green-blue.txt
    +-- Dark.txt
    +-- haze.txt
```

`raw/` contains underwater inputs.
`reference/` contains paired target images.
`splits/` contains degradation-type file lists.

## Checkpoints

Default checkpoint layout:

```text
checkpoints/
+-- prompt_after_stage1.pth
+-- pretrained_network.pth
```

For a fresh training run without prompt pretraining, use:

```bash
python maintrain_method2_relative_lr.py --load_pretrain_prompt false
```
