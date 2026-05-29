# SPEC — DOM-Anchored Dual-Reference DEM Correction Mode

> Status: reflects the implementation at commit `c7d75384` on branch `master`.
> Scope: the `--dem_mode` extension of PSMNet for LRO NAC / planetary DEM
> height-error inversion. The original stereo path (`dem_mode=False`) is
> unchanged and out of scope here.

---

## 1. Purpose & Physical Model

### 1.1 Problem
Given an **initial DEM** and two LRO NAC images (left/right views) of the same
terrain, estimate the per-pixel DEM elevation error **ΔZ** so the DEM can be
corrected.

### 1.2 Key physical relation
After ortho-projecting all data onto a common **DOM grid**, a DEM elevation
error `ΔZ` produces a 2-D pixel displacement `δ` in the DOM plane:

```
δ = ΔZ · tan(t) · â
```

| Symbol | Meaning |
|--------|---------|
| `t`  | per-pixel viewing **zenith** angle |
| `â`  | per-pixel unit **azimuth** direction of the view, projected onto the DOM plane (2-D vector `(ax, ay)`) |

Inverting:

```
ΔZ = |δ| / tan(t)
```

### 1.3 Dual-reference anchoring
For each view, a DEM-**rendered** image is produced from the *same* initial DEM,
matching the NAC geometry/lighting. Because NAC and render share the same DEM,
the **prior disparity between them is zero**. Any residual displacement `δ` is
caused only by `ΔZ`. The cost volume is therefore built **symmetrically around
zero offset**.

### 1.4 Left-right constraint
Left and right views have different azimuth directions (`â_L ≠ â_R`) but observe
the **same** terrain → they must yield the **same** `ΔZ`. This is the core
geometric constraint enforced by `L_lr` (left-right ΔZ consistency).

---

## 2. Tensor / Data Contract

All maps are at **image resolution** unless noted. `B`=batch, `N`=`geom_channels`
(default 5), `H/W`=tile size.

### 2.1 Model inputs (DEM mode)

| Name | Shape | Description |
|------|-------|-------------|
| `nac_l`, `nac_r` | `[B, 1+N, H, W]` | NAC grayscale (ch 0) + N geometry channels |
| `render_l`, `render_r` | `[B, 1+N, H, W]` | DEM-rendered image (ch 0) + N geometry channels |
| `view_dir_l`, `view_dir_r` | `[B, 2, H, W]` | per-pixel `(ax, ay)` azimuth unit direction |
| `tan_t_l`, `tan_t_r` | `[B, 1, H, W]` | per-pixel `tan(zenith)` |
| `mask_l`, `mask_r` | `[B, 1, H, W]` | valid-pixel mask (0 = invalid/shadow/occluded) |

### 2.2 Optional inputs (enable extra losses)

| Name | Shape | Enables |
|------|-------|---------|
| `dz_gt` | `[B, 1, H, W]` | `L_data` (supervised) |
| `dem_init` | `[B, 1, H, W]` | `L_photo` (photometric) |
| `sun_dir_l` | `[B, 3, H, W]` | `L_photo` |
| `normal_base_l` | `[B, 3, H, W]` | `L_photo` |
| `slope_prior` | `[B, 1, H, W]` | `L_slope` (not wired from `main.py` by default — passed as `None`) |

### 2.3 Geometry channels (default N=5)
`geom.tif` band order: `[slope, cosθ(incidence), nx, ny, nz]`.

### 2.4 Model outputs

| Mode | Output |
|------|--------|
| **train** | `(dz_l_all, dz_r_all, dz_fused)` where `dz_l_all`/`dz_r_all` are tuples `(dz1, dz2, dz3)` of `[B,1,H,W]`, and `dz_fused` is `[B,1,H,W]` |
| **eval** | `dz_fused` `[B,1,H,W]` |

---

## 3. Architecture (`models/stackhourglass.py`)

### 3.1 Construction
```python
PSMNet(maxdisp, dem_mode=True, residual_window=K, geom_channels=N)
```
- `in_channels` auto-set to `1 + geom_channels` when `dem_mode=True`.
- Feature extractor: `feature_extraction(in_channels=1+N)` (SPP backbone from
  `submodule.py`), shared (same weights) across all four images.
- Adds a learnable `dz_fusion` head (only in DEM mode):
  `Conv2d(2→16,3) → BN → ReLU → Conv2d(16→1,1)`.

### 3.2 Forward pipeline (per view branch `_dem_branch`)
1. Extract features for `nac` and `render` → `[B, C, Hf, Wf]` at **1/4** resolution.
2. Down-sample `view_dir` to feature resolution (bilinear).
3. Build **2-D residual cost volume** centred at zero offset (see §4):
   `[B, 2C, 2K+1, Hf, Wf]`.
4. 3-D stacked-hourglass aggregation `_aggregate_cost` → soft-argmin index in
   `[0, 2K]`. (Disparity axis is zero-padded to a multiple of 4 then cropped.)
5. Centre the index: `delta = pred - K` (signed pixel residual).
6. Invert: `dz = |delta| / clamp(tan_t, min=1e-4)`.

Produces `(dz1, dz2, dz3)` (train) at scales weighted `0.5 / 0.7 / 1.0`.

### 3.3 Fusion
```
dz_fused = dz_fusion( concat([dz_l3, dz_r3], dim=1) )   # [B,1,H,W]
```

### 3.4 Key parameters
| Param | Default | Meaning |
|-------|---------|---------|
| `residual_window` (K) | 8 | half-window → `2K+1` displacement steps |
| `geom_channels` (N) | 5 | geometry channels appended to each image |
| `maxdisp` | 192 | used only by the standard stereo path |

---

## 4. 2-D Residual Cost Volume (`models/cost_volume.py`)

`residual_cost_volume_2d(fea_nac, fea_render, az_dir, K) → [B, 2C, 2K+1, H, W]`

For each step `k ∈ [-K, K]`, warp the rendered features by `k` pixels **along the
per-pixel azimuth direction** and concatenate with NAC features:

```
x_sample = x + k · ax(x,y)
y_sample = y + k · ay(x,y)
```

- Uses normalized grid `[-1,1]` + `F.grid_sample(mode='bilinear',
  padding_mode='border', align_corners=True)`.
- `az_dir` must already be at feature resolution.
- Channel layout per step: `[:C]=fea_nac`, `[C:]=warped_render`.

---

## 5. Loss Functions (`losses.py`)

`total_dem_loss(...)` returns `(scalar_total, loss_dict)`.

| Term | Weight (CLI) | Requires | Definition |
|------|--------------|----------|------------|
| `L_data` | `--lambda_data` (1.0) | `dz_gt` | multi-scale smooth-L1 on `(dz1,dz2,dz3)` (w=0.5/0.7/1.0) for L & R + smooth-L1 on `dz_fused`, averaged /3 |
| `L_lr` | `--lambda_lr` (1.0) | always | `smooth_l1(dz_l3, dz_r3)` over valid mask — **core constraint** |
| `L_smooth` | `--lambda_smooth` (0.1) | `guide_img` | edge-aware slope smoothness: `Σ|∂dz|·exp(-|∂I|)` |
| `L_slope` | `--lambda_slope` (0.1) | `slope_prior` | `smooth_l1(|∇dz|, slope_prior)` |
| `L_photo` | `--lambda_photo` (1.0) | `dem_init, sun_dir_l, normal_base_l` | `smooth_l1(I_render(Z_init+dz), I_NAC)` via differentiable Lambertian shading |

Notes:
- `main.py` passes `guide_img = nac_l[:, :1]` and `slope_prior=None`.
- `lambda_photo` is forced to 0 if `dem_init is None`; `lambda_data` forced to 0
  if `dz_gt is None`.
- `isis_em_hook()` is provided as an interface for non-differentiable ISIS
  re-rendering (coarse-to-fine EM), separate from the differentiable
  `photometric_loss`.

---

## 6. Data Pipeline

### 6.1 Dataset layout (`dom_listfile.py`)
```
dataset_root/
  scene_001/
    nac_l.tif nac_r.tif render_l.tif render_r.tif
    geom.tif
    view_dir_l.tif view_dir_r.tif
    tan_t_l.tif tan_t_r.tif
    mask_l.tif mask_r.tif
    dz_gt.tif          # optional
    dem_init.tif       # optional
    sun_dir_l.tif      # optional
    normal_base_l.tif  # optional
  scene_002/ ...
```

### 6.2 CSV file-list format
Columns (comma-separated, `#` = comment), **min 11 required**:
```
nac_l, nac_r, render_l, render_r, geom,
view_dir_l, view_dir_r, tan_t_l, tan_t_r,
mask_l, mask_r [, dz_gt [, dem_init [, sun_dir_l [, normal_base_l]]]]
```

Generate lists:
```bash
python dataloader/dom_listfile.py \
  --dataroot /path/to/dataset_root \
  --out_train train_dom.csv \
  --out_test  test_dom.csv \
  --train_ratio 0.9
```

### 6.3 Loader (`dom_loader.DOMDataset`)
- Reads multi-band GeoTIFFs via **rasterio** (falls back to PIL, single-band only).
- Normalizes intensity: `(x - 0.445) / 0.269`.
- Concatenates `[intensity(1), geom(N)]` → `[1+N, H, W]` per image.
- Random crop to `patch_size=(256,256)` when `training=True`.
- Returns a **dict** (keys match §2.1/§2.2); optional fields are `None` if absent.

---

## 7. CLI / Training (`main.py`)

### 7.1 Relevant flags
| Flag | Default | Notes |
|------|---------|-------|
| `--dem_mode` | off | enables this whole mode |
| `--datapath` | — | root prepended to relative CSV paths |
| `--dom_train_list` | **required** in dem_mode | training CSV |
| `--dom_test_list` | **required** in dem_mode | test CSV |
| `--residual_window` | 8 | K |
| `--geom_channels` | 5 | N |
| `--use_view_dir` | True | (declared; azimuth always used in `_dem_branch`) |
| `--lambda_lr/smooth/slope/photo/data` | 1/0.1/0.1/1/1 | loss weights |
| `--epochs` | 10 | |
| `--loadmodel` / `--savemodel` | — | checkpoint I/O |

### 7.2 Fixed runtime behaviour (current code)
- DEM-mode batch sizes are hardcoded: **train=4, test=2** (workers 4/2).
- Optimizer: `Adam(lr=0.001, betas=(0.9,0.999))`.
- `adjust_learning_rate()` is a **no-op** (always 0.001).
- Checkpoints saved per epoch as `checkpoint_<epoch>.tar` (`{epoch, state_dict, train_loss}`).
- Test metric: `L1(dz_fused, dz_gt)` over `(mask_l>0)&(mask_r>0)`; returns 0 if no `dz_gt`.

### 7.3 Example invocation
```bash
python main.py \
  --dem_mode \
  --datapath /data/lro \
  --dom_train_list train_dom.csv \
  --dom_test_list  test_dom.csv \
  --residual_window 8 \
  --geom_channels 5 \
  --lambda_lr 1.0 --lambda_smooth 0.1 --lambda_slope 0.1 \
  --lambda_photo 1.0 --lambda_data 1.0 \
  --epochs 50 --savemodel ./checkpoints/
```

---

## 8. Dependencies
`torch>=1.10` (uses `torch.meshgrid(indexing='ij')`), `torchvision`, `numpy`,
`rasterio` (recommended; required for multi-band GeoTIFF), `Pillow` (fallback).

---

## 9. Known Gaps / TODO (from current code)
1. `slope_prior` is never wired in `main.py` (passed as `None`) → `L_slope` is
   effectively inactive unless code is modified.
2. `mask_l/mask_r` are accepted by `forward()` but **not used internally** (only
   in losses) — documented as "future loss weighting."
3. DEM-mode batch sizes & LR are hardcoded (no CLI control).
4. No dedicated DEM-mode inference/export script (only train+test in `main.py`);
   `submission.py`/`Test_img.py` are stereo-only.
5. `dz_fusion` fuses only the **finest** scale (`dz_l3`,`dz_r3`); multi-scale or
   geometry-aware fusion is not implemented.
6. PIL fallback in `dom_loader` only supports single-band images.
