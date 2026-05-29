"""Loss functions for DOM-Anchored DEM height-correction inversion.

Physical relationship (key formula)
-------------------------------------
After ortho-projection to the DOM grid, a DEM elevation error ΔZ causes a
2-D pixel displacement in the DOM plane:

    δ = ΔZ · tan(t) · â

where
    t  : per-pixel viewing zenith angle
    â  : per-pixel unit direction of the view azimuth projected onto the DOM
         plane (2-D vector; independent of solar illumination)

Inverting the above:
    ΔZ = |δ| / tan(t)

Left and right NAC views have different azimuth directions (â_L ≠ â_R), but
both observe the *same* terrain, so they must yield the **same** ΔZ.  The
left-right consistency loss (``lr_dem_consistency``) enforces this constraint.

Module contents
---------------
lr_dem_consistency     – left/right ΔZ agreement (core geometric constraint)
edge_aware_slope_smooth – edge-aware smoothness on predicted ΔZ
slope_consistency       – predicted slope vs. prior DEM slope
photometric_loss        – differentiable Lambertian rendering loss (L_photo)
multi_scale_dem_loss    – multi-scale smooth-L1 on ΔZ (when labels available)
total_dem_loss          – weighted sum of all active terms
"""

from __future__ import print_function
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Left-Right ΔZ Consistency (core constraint of the dual-reference setup)
# ---------------------------------------------------------------------------

def lr_dem_consistency(dz_l, dz_r, mask_l=None, mask_r=None):
    """Left-right ΔZ consistency loss.

    Both the left and right DEM-residual branches observe the same ground
    elevation error ΔZ.  Their predictions must agree:
        L_lr = smooth_l1(ΔZ_L, ΔZ_R)

    Physically: ΔZ_L = |δ_L| / tan(t_L),  ΔZ_R = |δ_R| / tan(t_R).

    Args:
        dz_l  (Tensor): [B, 1, H, W] ΔZ predicted from the left branch.
        dz_r  (Tensor): [B, 1, H, W] ΔZ predicted from the right branch.
        mask_l (Tensor | None): [B, 1, H, W] valid-pixel mask for left view.
        mask_r (Tensor | None): [B, 1, H, W] valid-pixel mask for right view.

    Returns:
        Scalar loss tensor.
    """
    if mask_l is not None and mask_r is not None:
        valid = (mask_l > 0) & (mask_r > 0)
        if valid.sum() == 0:
            return dz_l.new_tensor(0.0)
        return F.smooth_l1_loss(dz_l[valid], dz_r[valid], reduction='mean')
    return F.smooth_l1_loss(dz_l, dz_r, reduction='mean')


# ---------------------------------------------------------------------------
# 2. Edge-Aware Slope Smoothness
# ---------------------------------------------------------------------------

def edge_aware_slope_smooth(dz, guide_img):
    """Edge-aware slope smoothness regulariser.

    Penalises spatial gradients of the predicted ΔZ field while respecting
    image/DEM edges (weighted by exp(-|∇guide|)).  Suppresses noise in
    shadow/low-texture regions.

        L_smooth = Σ |∂x ΔZ| · exp(-|∂x I|)
                 + Σ |∂y ΔZ| · exp(-|∂y I|)

    Args:
        dz        (Tensor): [B, 1, H, W] predicted elevation correction.
        guide_img (Tensor): [B, C, H, W] guide image (NAC or rendered image).

    Returns:
        Scalar loss tensor.
    """
    # Spatial gradients of ΔZ
    dz_dx = torch.abs(dz[:, :, :, :-1] - dz[:, :, :, 1:])   # [B,1,H,W-1]
    dz_dy = torch.abs(dz[:, :, :-1, :] - dz[:, :, 1:, :])   # [B,1,H-1,W]

    # Guide gradients (mean over channels)
    g_dx = torch.mean(torch.abs(guide_img[:, :, :, :-1] - guide_img[:, :, :, 1:]),
                      dim=1, keepdim=True)                    # [B,1,H,W-1]
    g_dy = torch.mean(torch.abs(guide_img[:, :, :-1, :] - guide_img[:, :, 1:, :]),
                      dim=1, keepdim=True)                    # [B,1,H-1,W]

    smooth_x = (dz_dx * torch.exp(-g_dx)).mean()
    smooth_y = (dz_dy * torch.exp(-g_dy)).mean()
    return smooth_x + smooth_y


# ---------------------------------------------------------------------------
# 3. Slope Consistency (predicted ΔZ gradient vs. prior DEM slope)
# ---------------------------------------------------------------------------

def _image_gradient(tensor):
    """Return (dx, dy) gradient pair of a [B,1,H,W] tensor (via finite diff)."""
    dx = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]   # [B,1,H,W-1]
    dy = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]   # [B,1,H-1,W]
    return dx, dy


def slope_consistency(dz, slope_prior, mask=None):
    """Penalise deviation of predicted ΔZ gradient from prior DEM slope.

    Encourages the large-scale structure of the predicted correction to align
    with the trusted (coarse) initial DEM:
        L_slope = smooth_l1(|∇(Z_init + ΔZ)|, slope_prior)

    Here we approximate ∇(Z_init + ΔZ) with ∇ΔZ alone (∇Z_init is fixed and
    cancels in the loss derivative).

    Args:
        dz          (Tensor): [B, 1, H, W] predicted ΔZ.
        slope_prior (Tensor): [B, 1, H, W] prior DEM slope magnitude (pixels).
        mask        (Tensor | None): [B, 1, H, W] valid mask.

    Returns:
        Scalar loss tensor.
    """
    dx, dy = _image_gradient(dz)
    slope_pred_x = torch.abs(dx)
    slope_pred_y = torch.abs(dy)

    # Resize slope_prior to match gradient dimensions and average both axes
    sp_x = slope_prior[:, :, :, :slope_pred_x.shape[3]]
    sp_y = slope_prior[:, :, :slope_pred_y.shape[2], :]

    if mask is not None:
        mx = mask[:, :, :, :slope_pred_x.shape[3]]
        my = mask[:, :, :slope_pred_y.shape[2], :]
        loss_x = F.smooth_l1_loss(slope_pred_x[mx > 0], sp_x[mx > 0], reduction='mean') if (mx > 0).any() else dz.new_tensor(0.0)
        loss_y = F.smooth_l1_loss(slope_pred_y[my > 0], sp_y[my > 0], reduction='mean') if (my > 0).any() else dz.new_tensor(0.0)
    else:
        loss_x = F.smooth_l1_loss(slope_pred_x, sp_x, reduction='mean')
        loss_y = F.smooth_l1_loss(slope_pred_y, sp_y, reduction='mean')

    return 0.5 * (loss_x + loss_y)


# ---------------------------------------------------------------------------
# 4. Photometric (self-supervised) Loss  L_photo
# ---------------------------------------------------------------------------

def differentiable_lambertian_render(dem_update, sun_dir, normal_base,
                                     albedo=None):
    """Differentiable Lambertian rendering approximation.

    Computes a per-pixel intensity proportional to:
        I_render ≈ albedo · max(0, n · sun_dir)

    where n is the surface normal updated with predicted ΔZ.

    NOTE: ISIS shadow rendering is **not differentiable**.  This function
    provides a *differentiable approximation* for gradient-based training.
    For high-fidelity training with real ISIS renders, use the iterative EM
    loop described below (``isis_em_hook``).

    Args:
        dem_update  (Tensor): [B, 1, H, W] Z_init + ΔZ (updated DEM height).
        sun_dir     (Tensor): [B, 3, H, W] per-pixel unit solar direction
                              (sx, sy, sz) in the DOM coordinate frame.
        normal_base (Tensor): [B, 3, H, W] surface normals from the initial
                              DEM; used as a warm-start – we add a small normal
                              perturbation from ΔZ gradients.
        albedo      (Tensor | None): [B, 1, H, W] surface albedo.  If None,
                                    uniform albedo = 1 is assumed.

    Returns:
        I_render (Tensor): [B, 1, H, W] approximate rendered radiance.
    """
    # Estimate updated normals by perturbing normal_base with ΔZ gradients
    # ∂ΔZ/∂x, ∂ΔZ/∂y → small perturbation to (nx, ny, nz)
    # Use zero-padding so output shape matches input
    dz_dx = F.pad(dem_update[:, :, :, 1:] - dem_update[:, :, :, :-1],
                  (0, 1, 0, 0))
    dz_dy = F.pad(dem_update[:, :, 1:, :] - dem_update[:, :, :-1, :],
                  (0, 0, 0, 1))

    # Updated normal (un-normalised): n_upd ≈ n_base + (-∂ΔZ/∂x, -∂ΔZ/∂y, 0)
    n_upd = normal_base.clone()
    n_upd[:, 0:1] = n_upd[:, 0:1] - dz_dx
    n_upd[:, 1:2] = n_upd[:, 1:2] - dz_dy
    # Normalise
    n_norm = n_upd / (n_upd.norm(dim=1, keepdim=True).clamp(min=1e-6))

    # Lambertian shading: cosθ = n · sun_dir
    cos_theta = (n_norm * sun_dir).sum(dim=1, keepdim=True).clamp(min=0.0)

    if albedo is not None:
        I_render = albedo * cos_theta
    else:
        I_render = cos_theta

    return I_render


def photometric_loss(dz, nac_img, dem_init, sun_dir, normal_base,
                     albedo=None, mask=None):
    """Self-supervised photometric loss (differentiable Lambertian).

    Predicts what the scene should look like under the updated DEM and
    compares it to the true NAC image:
        L_photo = smooth_l1(I_render(Z_init + ΔZ), I_NAC)

    This implements the differentiable approximation.

    For a full ISIS-quality render, use the ``isis_em_hook`` interface.

    Args:
        dz         (Tensor): [B, 1, H, W] predicted elevation correction.
        nac_img    (Tensor): [B, 1, H, W] true NAC grayscale image.
        dem_init   (Tensor): [B, 1, H, W] initial DEM height field.
        sun_dir    (Tensor): [B, 3, H, W] unit solar direction.
        normal_base(Tensor): [B, 3, H, W] surface normals from initial DEM.
        albedo     (Tensor | None): [B, 1, H, W] surface albedo.
        mask       (Tensor | None): [B, 1, H, W] valid-pixel mask.

    Returns:
        Scalar loss tensor.
    """
    dem_update = dem_init + dz
    I_render = differentiable_lambertian_render(
        dem_update, sun_dir, normal_base, albedo=albedo)

    if mask is not None:
        valid = mask > 0
        if not valid.any():
            return dz.new_tensor(0.0)
        return F.smooth_l1_loss(I_render[valid], nac_img[valid], reduction='mean')
    return F.smooth_l1_loss(I_render, nac_img, reduction='mean')


# ---------------------------------------------------------------------------
# ISIS EM hook (non-differentiable external renderer placeholder)
# ---------------------------------------------------------------------------

def isis_em_hook(dz, dem_init, render_fn, nac_img, mask=None):
    """Placeholder / interface for iterative EM with external ISIS rendering.

    ISIS shadow rendering is NOT differentiable.  This function provides the
    interface for a coarse-to-fine iterative EM loop:

        1. Network predicts ΔZ.
        2. Call ``render_fn(dem_init + ΔZ)`` to get a new ISIS render
           (external process, no gradient).
        3. Use the new render as a pseudo-label and compute a supervised loss.
        4. Back-propagate through step 1 only.

    Coarse-to-fine workflow:
        Repeat for each refinement iteration:
            a. nac_render = render_fn(Z_curr)     # ISIS, no grad
            b. dz_pred = model(nac, nac_render)   # network forward
            c. Z_curr = Z_curr + dz_pred.detach() # update DEM
            d. loss = photometric_loss(dz_pred, nac, Z_curr, ...)
            e. loss.backward(); optimizer.step()

    Args:
        dz        (Tensor): [B, 1, H, W] current ΔZ prediction (with grad).
        dem_init  (Tensor): [B, 1, H, W] current DEM (no grad, updated in loop).
        render_fn (callable): function(dem_tensor) → rendered image tensor.
                              Must handle detached numpy/torch input.
                              Returns [B, 1, H, W] torch Tensor (no grad).
        nac_img   (Tensor): [B, 1, H, W] true NAC image.
        mask      (Tensor | None): valid-pixel mask.

    Returns:
        Scalar smooth-L1 loss between ISIS re-render and real NAC.

    Note:
        Because render_fn is non-differentiable, gradient flows only through
        dz via the pseudo-label mechanism.  For true differentiable rendering
        use ``photometric_loss`` with the Lambertian approximation.
    """
    with torch.no_grad():
        dem_updated = dem_init + dz.detach()
        # render_fn should return a tensor on the same device
        pseudo_render = render_fn(dem_updated)  # [B, 1, H, W], no grad

    # Pseudo-supervised loss: make network render match pseudo-render
    # (gradient flows through dz implicitly via the differentiable approximation,
    #  or directly if the loss is on a differentiable path)
    if mask is not None:
        valid = mask > 0
        if not valid.any():
            return dz.new_tensor(0.0)
        return F.smooth_l1_loss(nac_img[valid], pseudo_render[valid].detach(),
                                reduction='mean')
    return F.smooth_l1_loss(nac_img, pseudo_render.detach(), reduction='mean')


# ---------------------------------------------------------------------------
# 5. Multi-scale data term (when reference ΔZ labels are available)
# ---------------------------------------------------------------------------

def multi_scale_dem_loss(dz_preds, dz_gt, mask=None):
    """Multi-scale supervised loss on ΔZ (PSMNet 0.5/0.7/1.0 weighting).

    Args:
        dz_preds (tuple): (dz1, dz2, dz3) from the DEM branch, [B,1,H,W] each.
        dz_gt    (Tensor): [B, 1, H, W] ground-truth ΔZ (reference DEM error).
        mask     (Tensor | None): [B, 1, H, W] valid mask.

    Returns:
        Scalar loss.
    """
    weights = [0.5, 0.7, 1.0]
    loss = dz_preds[0].new_tensor(0.0)
    for w, dz_pred in zip(weights, dz_preds):
        if mask is not None:
            valid = mask.squeeze(1) > 0
            if not valid.any():
                continue
            loss = loss + w * F.smooth_l1_loss(
                dz_pred.squeeze(1)[valid],
                dz_gt.squeeze(1)[valid],
                reduction='mean')
        else:
            loss = loss + w * F.smooth_l1_loss(
                dz_pred, dz_gt, reduction='mean')
    return loss


# ---------------------------------------------------------------------------
# 6. Combined total loss
# ---------------------------------------------------------------------------

def total_dem_loss(
        dz_l_all, dz_r_all, dz_fused,
        # optional supervision
        dz_gt=None,
        # photometric self-supervision
        nac_l=None, dem_init=None, sun_dir_l=None, normal_base_l=None,
        albedo_l=None,
        # geometry auxiliary
        slope_prior=None,
        guide_img=None,
        # masks
        mask_l=None, mask_r=None,
        # loss weights (all positive scalars; set 0 to disable a term)
        lambda_lr=1.0,
        lambda_smooth=0.1,
        lambda_slope=0.1,
        lambda_photo=1.0,
        lambda_data=1.0,
):
    """Compute the weighted combination of all DEM-mode loss terms.

    Active terms:
        L_data   (λ_data)   – multi-scale smooth-L1 on ΔZ (needs dz_gt)
        L_lr     (λ_lr)     – left-right ΔZ consistency (always active)
        L_smooth (λ_smooth) – edge-aware slope smoothness (needs guide_img)
        L_slope  (λ_slope)  – slope vs. prior DEM (needs slope_prior)
        L_photo  (λ_photo)  – differentiable photometric (needs nac_l, dem_init,
                              sun_dir_l, normal_base_l)

    Args: see individual loss functions for tensor shapes.

    Returns:
        total_loss (Tensor): scalar.
        loss_dict  (dict):   per-term scalar values for logging.
    """
    loss_dict = {}
    total = dz_fused.new_tensor(0.0)

    # L_data: supervised by reference ΔZ when available
    if dz_gt is not None and lambda_data > 0:
        l_data = multi_scale_dem_loss(dz_l_all, dz_gt, mask=mask_l)
        l_data = l_data + multi_scale_dem_loss(dz_r_all, dz_gt, mask=mask_r)
        l_data = l_data + F.smooth_l1_loss(dz_fused, dz_gt, reduction='mean')
        l_data = l_data / 3.0
        loss_dict['L_data'] = l_data.item()
        total = total + lambda_data * l_data

    # L_lr: left-right ΔZ consistency (core geometric constraint)
    if lambda_lr > 0:
        l_lr = lr_dem_consistency(dz_l_all[2], dz_r_all[2],
                                  mask_l=mask_l, mask_r=mask_r)
        loss_dict['L_lr'] = l_lr.item()
        total = total + lambda_lr * l_lr

    # L_smooth: edge-aware smoothness
    if guide_img is not None and lambda_smooth > 0:
        l_smooth = edge_aware_slope_smooth(dz_fused, guide_img)
        loss_dict['L_smooth'] = l_smooth.item()
        total = total + lambda_smooth * l_smooth

    # L_slope: slope consistency with prior DEM
    if slope_prior is not None and lambda_slope > 0:
        l_slope = slope_consistency(dz_fused, slope_prior, mask=mask_l)
        loss_dict['L_slope'] = l_slope.item()
        total = total + lambda_slope * l_slope

    # L_photo: differentiable photometric self-supervision
    if (nac_l is not None and dem_init is not None
            and sun_dir_l is not None and normal_base_l is not None
            and lambda_photo > 0):
        l_photo = photometric_loss(
            dz_fused, nac_l[:, :1],  # use only the intensity channel
            dem_init, sun_dir_l, normal_base_l,
            albedo=albedo_l, mask=mask_l)
        loss_dict['L_photo'] = l_photo.item()
        total = total + lambda_photo * l_photo

    loss_dict['total'] = total.item()
    return total, loss_dict
