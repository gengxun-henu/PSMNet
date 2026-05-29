# Pyramid Stereo Matching Network

This repository contains the code (in PyTorch) for "[Pyramid Stereo Matching Network](https://arxiv.org/abs/1803.08669)" paper (CVPR 2018) by [Jia-Ren Chang](https://jiarenchang.github.io/) and [Yong-Sheng Chen](https://people.cs.nctu.edu.tw/~yschen/).

#### changelog
2020/12/20: Update PSMNet: now support torch 1.6.0 / torchvision 0.5.0 and python 3.7, Removed inconsistent indentation.

2020/12/20: Our proposed Real-Time Stereo can be found here [Real-time Stereo](https://github.com/JiaRenChang/RealtimeStereo).
### Citation
```
@inproceedings{chang2018pyramid,
  title={Pyramid Stereo Matching Network},
  author={Chang, Jia-Ren and Chen, Yong-Sheng},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5410--5418},
  year={2018}
}
```

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)
4. [DOM-Anchored Dual-Reference Mode](#dom-anchored-dual-reference-mode)
5. [Contacts](#contacts)

## Introduction

Recent work has shown that depth estimation from a stereo pair of images can be formulated as a supervised learning task to be resolved with convolutional neural networks (CNNs). However, current architectures rely on patch-based Siamese networks, lacking the means to exploit context information for finding correspondence in illposed regions. To tackle this problem, we propose PSMNet, a pyramid stereo matching network consisting of two main modules: spatial pyramid pooling and 3D CNN. The spatial pyramid pooling module takes advantage of the capacity of global context information by aggregating context in different scales and locations to form a cost volume. The 3D CNN learns to regularize cost volume using stacked multiple hourglass networks in conjunction with intermediate supervision.

<img align="center" src="https://user-images.githubusercontent.com/11732099/43501836-1d32897c-958a-11e8-8083-ad41ec26be17.jpg">

## Usage

### Dependencies

- [Python 3.7](https://www.python.org/downloads/)
- [PyTorch(1.6.0+)](http://pytorch.org)
- torchvision 0.5.0
- [KITTI Stereo](http://www.cvlibs.net/datasets/kitti/eval_stereo.php)
- [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

```
Usage of Scene Flow dataset
Download RGB cleanpass images and its disparity for three subset: FlyingThings3D, Driving, and Monkaa.
Put them in the same folder.
And rename the folder as: "driving_frames_cleanpass", "driving_disparity", "monkaa_frames_cleanpass", "monkaa_disparity", "frames_cleanpass", "frames_disparity".
```
### Notice
1. Warning of upsample function in PyTorch 0.4.1+: add "align_corners=True" to upsample functions.
2. Output disparity may be better with multipling by 1.17. Reported from issues [#135](https://github.com/JiaRenChang/PSMNet/issues/135) and [#113](https://github.com/JiaRenChang/PSMNet/issues/113).

### Train
As an example, use the following command to train a PSMNet on Scene Flow

```
python main.py --maxdisp 192 \
               --model stackhourglass \
               --datapath (your scene flow data folder)\
               --epochs 10 \
               --loadmodel (optional)\
               --savemodel (path for saving model)
```

As another example, use the following command to finetune a PSMNet on KITTI 2015

```
python finetune.py --maxdisp 192 \
                   --model stackhourglass \
                   --datatype 2015 \
                   --datapath (KITTI 2015 training data folder) \
                   --epochs 300 \
                   --loadmodel (pretrained PSMNet) \
                   --savemodel (path for saving model)
```
You can also see those examples in run.sh.

### Evaluation
Use the following command to evaluate the trained PSMNet on KITTI 2015 test data

```
python submission.py --maxdisp 192 \
                     --model stackhourglass \
                     --KITTI 2015 \
                     --datapath (KITTI 2015 test data folder) \
                     --loadmodel (finetuned PSMNet) \
```

### Pretrained Model
※NOTE: The pretrained model were saved in .tar; however, you don't need to untar it. Use torch.load() to load it.

Update: 2018/9/6 We released the pre-trained KITTI 2012 model.

Update: 2021/9/22 a pretrained model using torch 1.8.1 (the previous model weight are trained torch 0.4.1)

| KITTI 2015 |  Scene Flow | KITTI 2012 | Scene Flow (torch 1.8.1)
|---|---|---|---|
|[Google Drive](https://drive.google.com/file/d/1pHWjmhKMG4ffCrpcsp_MTXMJXhgl3kF9/view?usp=sharing)|[Google Drive](https://drive.google.com/file/d/1xoqkQ2NXik1TML_FMUTNZJFAHrhLdKZG/view?usp=sharing)|[Google Drive](https://drive.google.com/file/d/1p4eJ2xDzvQxaqB20A_MmSP9-KORBX1pZ/view?usp=sharing)| [Google Drive](https://drive.google.com/file/d/1NDKrWHkwgMKtDwynXVU12emK3G5d5kkp/view?usp=sharing)

### Test on your own stereo pair
```
python Test_img.py --loadmodel (finetuned PSMNet) --leftimg ./left.png --rightimg ./right.png
```

## Results

### Evaluation of PSMNet with different settings
<img align="center" src="https://user-images.githubusercontent.com/11732099/37817886-45a12ece-2eb3-11e8-8254-ae92c723b2f6.png">

※Note that the reported 3-px validation errors were calculated using KITTI's official matlab code, not our code.

### Results on KITTI 2015 leaderboard
[Leaderboard Link](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

| Method | D1-all (All) | D1-all (Noc)| Runtime (s) |
|---|---|---|---|
| PSMNet | 2.32 % | 2.14 % | 0.41 |
| [iResNet-i2](https://arxiv.org/abs/1712.01039) | 2.44 % | 2.19 % | 0.12 |
| [GC-Net](https://arxiv.org/abs/1703.04309) | 2.87 % | 2.61 % | 0.90 |
| [MC-CNN](https://github.com/jzbontar/mc-cnn) | 3.89 % | 3.33 % | 67 |

### Qualitative results
#### Left image
<img align="center" src="http://www.cvlibs.net/datasets/kitti/results/efb9db97938e12a20b9c95ce593f633dd63a2744/image_0/000004_10.png">

#### Predicted disparity
<img align="center" src="http://www.cvlibs.net/datasets/kitti/results/efb9db97938e12a20b9c95ce593f633dd63a2744/result_disp_img_0/000004_10.png">

#### Error
<img align="center" src="http://www.cvlibs.net/datasets/kitti/results/efb9db97938e12a20b9c95ce593f633dd63a2744/errors_disp_img_0/000004_10.png">

### Visualization of Receptive Field
We visualize the receptive fields of different settings of PSMNet, full setting and baseline.

Full setting: dilated conv, SPP, stacked hourglass

Baseline: no dilated conv, no SPP, no stacked hourglass

The receptive fields were calculated for the pixel at image center, indicated by the red cross.

<img align="center" src="https://user-images.githubusercontent.com/11732099/37876179-6d6dd97e-307b-11e8-803e-bcdbec29fb94.png">

---

## DOM-Anchored Dual-Reference Mode

This extension repurposes PSMNet for **LRO NAC (lunar/Mars) DEM height-correction inversion** in ortho-rectified (DOM) image space.

### Background

Given an initial DEM and camera position/attitude, both a real NAC image and a DEM-rendered simulation (via ISIS shadow module at the same epoch) can be projected onto the **same DOM grid** at the same resolution and coordinate system.

- **Left NAC** is matched with its **DEM render** (left geometry).
- **Right NAC** is matched with its **DEM render** (right geometry).
- Both renders use the **same initial DEM**, so the prior disparity between NAC and render is **zero** — residuals arise solely from DEM elevation error ΔZ.

In DOM space the residual displacement is:

```
δ = ΔZ · tan(t) · â
```

where `t` is the viewing zenith angle and `â` is the 2-D unit direction of the view azimuth projected onto the DOM plane (independent of solar illumination, determined only by viewing geometry).

Inverting:
```
ΔZ = |δ| / tan(t)
```

Because the left and right views have different azimuth directions (`â_L ≠ â_R`), both must yield the **same scalar ΔZ** — this left-right consistency constraint robustly determines ΔZ even in low-texture regions.

### Input Channel Layout

Each image tile has **1 + N** channels (default N = 5):

| Channel index | Content |
|---|---|
| 0 | Grayscale NAC (or DEM-rendered) intensity (single band) |
| 1 | Slope |
| 2 | cos(incidence angle) θ |
| 3 | Surface normal nx |
| 4 | Surface normal ny |
| 5 | Surface normal nz |

Auxiliary per-pixel geometric fields (provided separately at image resolution):

| Field | Shape | Content |
|---|---|---|
| `view_dir_L/R` | `[B, 2, H, W]` | (ax, ay) unit view-azimuth direction |
| `tan_t_L/R` | `[B, 1, H, W]` | tan(viewing zenith angle) |
| `mask_L/R` | `[B, 1, H, W]` | Valid-pixel mask (0 = shadowed/occluded) |

### Data Organisation

Each sample lives in a sub-directory containing the following GeoTIFF files:

```
dataset_root/
    scene_001/
        nac_l.tif           # left  NAC grayscale (1-band float32)
        nac_r.tif           # right NAC grayscale (1-band float32)
        render_l.tif        # DEM render for left  view (1-band float32)
        render_r.tif        # DEM render for right view (1-band float32)
        geom.tif            # 5-band geometry: [slope, cosθ, nx, ny, nz]
        view_dir_l.tif      # 2-band (ax, ay) left  view azimuth
        view_dir_r.tif      # 2-band (ax, ay) right view azimuth
        tan_t_l.tif         # 1-band tan(zenith) left
        tan_t_r.tif         # 1-band tan(zenith) right
        mask_l.tif          # 1-band uint8 valid mask left
        mask_r.tif          # 1-band uint8 valid mask right
        dz_gt.tif           # (optional) reference ΔZ for supervised training
        dem_init.tif        # (optional) initial DEM heights [float32]
        sun_dir_l.tif       # (optional) 3-band solar direction [sx, sy, sz]
        normal_base_l.tif   # (optional) 3-band surface normals from init DEM
    scene_002/
        ...
```

Generate CSV file lists with:

```bash
python dataloader/dom_listfile.py \
    --dataroot /path/to/dataset_root \
    --out_train train_dom.csv \
    --out_test  test_dom.csv \
    --train_ratio 0.9
```

### New argparse Parameters

| Parameter | Default | Description |
|---|---|---|
| `--dem_mode` | False | Enable DOM Dual-Reference mode |
| `--residual_window K` | 8 | Half-window for residual cost volume (2K+1 steps) |
| `--geom_channels N` | 5 | Number of geometry input channels (slope, cosθ, nx, ny, nz) |
| `--use_view_dir` | True | Use per-pixel view azimuth direction |
| `--dom_train_list` | — | Path to training CSV (required in `--dem_mode`) |
| `--dom_test_list` | — | Path to test/validation CSV |
| `--lambda_lr` | 1.0 | Weight for left-right ΔZ consistency loss |
| `--lambda_smooth` | 0.1 | Weight for edge-aware slope smoothness |
| `--lambda_slope` | 0.1 | Weight for slope-prior consistency |
| `--lambda_photo` | 1.0 | Weight for photometric self-supervision |
| `--lambda_data` | 1.0 | Weight for supervised ΔZ data term (needs `dz_gt`) |

### Training – DEM Mode

```bash
python main.py \
    --dem_mode \
    --model stackhourglass \
    --residual_window 8 \
    --geom_channels 5 \
    --dom_train_list /path/to/train_dom.csv \
    --dom_test_list  /path/to/test_dom.csv \
    --epochs 50 \
    --savemodel /path/to/checkpoints \
    --lambda_lr 1.0 \
    --lambda_smooth 0.1 \
    --lambda_photo 1.0
```

### Coarse-to-Fine Iterative Refinement

For best results, run multiple rounds of prediction → DEM update → re-render:

1. **Predict ΔZ** with the network on `(NAC, render_init)`.
2. **Update DEM**: `Z_curr = Z_init + ΔZ.detach()`.
3. **Re-render** `render_new = ISIS_shadow(Z_curr)` (external, non-differentiable).
4. **Re-run network** on `(NAC, render_new)` to predict residual ΔZ₂.
5. Repeat until convergence.

During each round, the network can be fine-tuned on the pseudo-labels from step 3 via the `isis_em_hook` interface in `losses.py`.

### Loss Functions (losses.py)

| Function | Symbol | Description |
|---|---|---|
| `lr_dem_consistency` | L_lr | Left-right ΔZ agreement (core constraint) |
| `edge_aware_slope_smooth` | L_smooth | Edge-aware smoothness of predicted ΔZ |
| `slope_consistency` | L_slope | Predicted gradient vs. prior DEM slope |
| `photometric_loss` | L_photo | Differentiable Lambertian rendering vs. NAC |
| `multi_scale_dem_loss` | L_data | Multi-scale smooth-L1 (requires dz_gt) |
| `total_dem_loss` | — | Weighted sum of all active terms |

Total loss:
```
L = λ_data · L_data  +  λ_lr · L_lr  +  λ_smooth · L_smooth
  + λ_slope · L_slope  +  λ_photo · L_photo
```

### Shadow / Occlusion Masks

Pass `mask_l` and `mask_r` ([B, 1, H, W], 0 = invalid) to the model forward call and to `total_dem_loss`. Loss terms automatically exclude invalid pixels, preventing shadow and occlusion regions from polluting the gradient.

### ISIS Rendering Note

ISIS shadow rendering is **not differentiable**. The `photometric_loss` function uses a differentiable Lambertian approximation for end-to-end training. For high-fidelity training with real ISIS renders, use the iterative EM loop via `isis_em_hook` in `losses.py` (see the docstring there for the full workflow).

---

## Contacts
followwar@gmail.com

Any discussions or concerns are welcomed!

## Introduction

Recent work has shown that depth estimation from a stereo pair of images can be formulated as a supervised learning task to be resolved with convolutional neural networks (CNNs). However, current architectures rely on patch-based Siamese networks, lacking the means to exploit context information for finding correspondence in illposed regions. To tackle this problem, we propose PSMNet, a pyramid stereo matching network consisting of two main modules: spatial pyramid pooling and 3D CNN. The spatial pyramid pooling module takes advantage of the capacity of global context information by aggregating context in different scales and locations to form a cost volume. The 3D CNN learns to regularize cost volume using stacked multiple hourglass networks in conjunction with intermediate supervision.

<img align="center" src="https://user-images.githubusercontent.com/11732099/43501836-1d32897c-958a-11e8-8083-ad41ec26be17.jpg">

## Usage

### Dependencies

- [Python 3.7](https://www.python.org/downloads/)
- [PyTorch(1.6.0+)](http://pytorch.org)
- torchvision 0.5.0
- [KITTI Stereo](http://www.cvlibs.net/datasets/kitti/eval_stereo.php)
- [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

```
Usage of Scene Flow dataset
Download RGB cleanpass images and its disparity for three subset: FlyingThings3D, Driving, and Monkaa.
Put them in the same folder.
And rename the folder as: "driving_frames_cleanpass", "driving_disparity", "monkaa_frames_cleanpass", "monkaa_disparity", "frames_cleanpass", "frames_disparity".
```
### Notice
1. Warning of upsample function in PyTorch 0.4.1+: add "align_corners=True" to upsample functions.
2. Output disparity may be better with multipling by 1.17. Reported from issues [#135](https://github.com/JiaRenChang/PSMNet/issues/135) and [#113](https://github.com/JiaRenChang/PSMNet/issues/113).

### Train
As an example, use the following command to train a PSMNet on Scene Flow

```
python main.py --maxdisp 192 \
               --model stackhourglass \
               --datapath (your scene flow data folder)\
               --epochs 10 \
               --loadmodel (optional)\
               --savemodel (path for saving model)
```

As another example, use the following command to finetune a PSMNet on KITTI 2015

```
python finetune.py --maxdisp 192 \
                   --model stackhourglass \
                   --datatype 2015 \
                   --datapath (KITTI 2015 training data folder) \
                   --epochs 300 \
                   --loadmodel (pretrained PSMNet) \
                   --savemodel (path for saving model)
```
You can also see those examples in run.sh.

### Evaluation
Use the following command to evaluate the trained PSMNet on KITTI 2015 test data

```
python submission.py --maxdisp 192 \
                     --model stackhourglass \
                     --KITTI 2015 \
                     --datapath (KITTI 2015 test data folder) \
                     --loadmodel (finetuned PSMNet) \
```

### Pretrained Model
※NOTE: The pretrained model were saved in .tar; however, you don't need to untar it. Use torch.load() to load it.

Update: 2018/9/6 We released the pre-trained KITTI 2012 model.

Update: 2021/9/22 a pretrained model using torch 1.8.1 (the previous model weight are trained torch 0.4.1)

| KITTI 2015 |  Scene Flow | KITTI 2012 | Scene Flow (torch 1.8.1)
|---|---|---|---|
|[Google Drive](https://drive.google.com/file/d/1pHWjmhKMG4ffCrpcsp_MTXMJXhgl3kF9/view?usp=sharing)|[Google Drive](https://drive.google.com/file/d/1xoqkQ2NXik1TML_FMUTNZJFAHrhLdKZG/view?usp=sharing)|[Google Drive](https://drive.google.com/file/d/1p4eJ2xDzvQxaqB20A_MmSP9-KORBX1pZ/view?usp=sharing)| [Google Drive](https://drive.google.com/file/d/1NDKrWHkwgMKtDwynXVU12emK3G5d5kkp/view?usp=sharing)

### Test on your own stereo pair
```
python Test_img.py --loadmodel (finetuned PSMNet) --leftimg ./left.png --rightimg ./right.png
```

## Results

### Evaluation of PSMNet with different settings
<img align="center" src="https://user-images.githubusercontent.com/11732099/37817886-45a12ece-2eb3-11e8-8254-ae92c723b2f6.png">

※Note that the reported 3-px validation errors were calculated using KITTI's official matlab code, not our code.

### Results on KITTI 2015 leaderboard
[Leaderboard Link](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

| Method | D1-all (All) | D1-all (Noc)| Runtime (s) |
|---|---|---|---|
| PSMNet | 2.32 % | 2.14 % | 0.41 |
| [iResNet-i2](https://arxiv.org/abs/1712.01039) | 2.44 % | 2.19 % | 0.12 |
| [GC-Net](https://arxiv.org/abs/1703.04309) | 2.87 % | 2.61 % | 0.90 |
| [MC-CNN](https://github.com/jzbontar/mc-cnn) | 3.89 % | 3.33 % | 67 |

### Qualitative results
#### Left image
<img align="center" src="http://www.cvlibs.net/datasets/kitti/results/efb9db97938e12a20b9c95ce593f633dd63a2744/image_0/000004_10.png">

#### Predicted disparity
<img align="center" src="http://www.cvlibs.net/datasets/kitti/results/efb9db97938e12a20b9c95ce593f633dd63a2744/result_disp_img_0/000004_10.png">

#### Error
<img align="center" src="http://www.cvlibs.net/datasets/kitti/results/efb9db97938e12a20b9c95ce593f633dd63a2744/errors_disp_img_0/000004_10.png">

### Visualization of Receptive Field
We visualize the receptive fields of different settings of PSMNet, full setting and baseline.

Full setting: dilated conv, SPP, stacked hourglass

Baseline: no dilated conv, no SPP, no stacked hourglass

The receptive fields were calculated for the pixel at image center, indicated by the red cross.

<img align="center" src="https://user-images.githubusercontent.com/11732099/37876179-6d6dd97e-307b-11e8-803e-bcdbec29fb94.png">



## Contacts
followwar@gmail.com

Any discussions or concerns are welcomed!
