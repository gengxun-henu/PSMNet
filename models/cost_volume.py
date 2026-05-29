"""2-D Residual Cost Volume for DOM-Anchored DEM Correction.

Physical background
-------------------
After ortho-rectification to a common DOM grid, residual pixel shifts between
a real NAC image and a DEM-rendered image arise solely from DEM elevation error
ΔZ.  The 2-D displacement is:

    δ = ΔZ · tan(t) · â

where
    t  : viewing zenith angle (per-pixel scalar)
    â  : unit direction of the view azimuth projected onto the DOM plane
           (2-D vector field, per pixel)

Because the prior disparity is **zero** (both sides are anchored to the same
DEM), the cost volume is built symmetrically around zero offset.

Usage
-----
    cost = residual_cost_volume_2d(fea_nac, fea_render, az_dir, K)

where az_dir = (ax, ay) is the per-pixel 2-D unit direction already
down-sampled to feature map resolution (1/4 of the image).
"""

from __future__ import print_function
import torch
import torch.nn.functional as F


def residual_cost_volume_2d(fea_nac, fea_render, az_dir, K):
    """Build a 2-D residual cost volume centred at zero offset.

    For each candidate residual step k in [-K, …, K] (total 2K+1 steps), the
    rendered feature map is warped by k pixels along the per-pixel direction
    az_dir and then concatenated with the NAC feature map.

    The warp formula (per pixel (x, y)):
        x_sample = x + k · ax(x,y)
        y_sample = y + k · ay(x,y)

    Note that az_dir already encodes direction; the magnitude k scales it.

    Args:
        fea_nac    (Tensor): [B, C, H, W] – NAC image features.
        fea_render (Tensor): [B, C, H, W] – DEM-rendered image features.
        az_dir     (Tensor): [B, 2, H, W] – per-pixel (ax, ay) unit direction
                             at feature map resolution (1/4 of image size).
        K          (int)   : half-window size; creates 2K+1 displacement steps.

    Returns:
        cost (Tensor): [B, 2C, 2K+1, H, W]
    """
    B, C, H, W = fea_nac.shape
    device = fea_nac.device
    dtype = fea_nac.dtype

    cost = fea_nac.new_zeros(B, 2 * C, 2 * K + 1, H, W)

    # Normalised base coordinate grids in [-1, 1]
    # xs: [B, H, W],  ys: [B, H, W]
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    ys_grid, xs_grid = torch.meshgrid(ys, xs, indexing='ij')  # [H, W]
    xs_grid = xs_grid.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    ys_grid = ys_grid.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]

    # az_dir[:, 0]: ax component; az_dir[:, 1]: ay component  [B, H, W]
    ax = az_dir[:, 0]  # [B, H, W]
    ay = az_dir[:, 1]  # [B, H, W]

    # Convert pixel-unit step to normalised-grid units
    # In normalised coords a shift of 1 pixel = 2/(W-1) horizontally
    dx_unit = 2.0 / (W - 1)
    dy_unit = 2.0 / (H - 1)

    for idx, k in enumerate(range(-K, K + 1)):
        # Warp the rendered features: sample at position shifted by k along â
        x_s = xs_grid + k * ax * dx_unit  # [B, H, W]
        y_s = ys_grid + k * ay * dy_unit  # [B, H, W]

        grid = torch.stack([x_s, y_s], dim=-1)  # [B, H, W, 2]
        warped_render = F.grid_sample(
            fea_render, grid,
            mode='bilinear', padding_mode='border', align_corners=True
        )  # [B, C, H, W]

        cost[:, :C, idx]  = fea_nac
        cost[:, C:, idx]  = warped_render

    return cost.contiguous()
