"""PSMNet – stacked hourglass variant with optional DOM-DEM mode.

Original stereo path (``dem_mode=False``, default)
----------------------------------------------------
Accepts (left, right) RGB images, builds the standard horizontal-shift
cost volume, and returns disparity predictions.  Behaviour is identical to
the original PSMNet paper.

DOM-Anchored Dual-Reference mode (``dem_mode=True``)
-----------------------------------------------------
Designed for LRO NAC height-correction inversion in ortho (DOM) space.

Physical setup
~~~~~~~~~~~~~~
Four images are provided, all projected onto the **same DOM grid**:

    NAC_L    – left  LRO NAC (true image, 1 channel)
    render_L – DEM-rendered image matching NAC_L geometry/lighting
    NAC_R    – right LRO NAC (true image, 1 channel)
    render_R – DEM-rendered image matching NAC_R geometry/lighting

Because *both* rendered images use the **same initial DEM**, the prior
disparity between NAC_L↔render_L (and NAC_R↔render_R) is **zero**.
Any residual pixel displacement δ comes only from the DEM elevation error ΔZ:

    δ = ΔZ · tan(t) · â          (per-pixel, 2-D vector in DOM plane)

where t is the viewing zenith angle and â is the 2-D azimuth unit direction.
Inverting:
    ΔZ = |δ| / tan(t)

Two independent observations of ΔZ (from left and right views) provide a
strong left-right consistency constraint for robust inversion.

Input channels
~~~~~~~~~~~~~~
Each image tile has shape [B, 1+N, H, W]:
    channel 0        : grayscale NAC (or rendered) intensity
    channels 1..1+N  : geometry (e.g. N=5: slope, cosθ, nx, ny, nz)

Geometric auxiliary inputs (at image resolution, down-sampled internally):
    view_dir_L/R : [B, 2, H, W]  – (ax, ay) per-pixel azimuth unit direction
    tan_t_L/R    : [B, 1, H, W]  – per-pixel tan(zenith angle)
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodule import *
from .cost_volume import residual_cost_volume_2d


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                                   nn.BatchNorm3d(inplanes*2))  # +conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                                   nn.BatchNorm3d(inplanes))  # +x

    def forward(self, x, presqu, postsqu):

        out  = self.conv1(x)       # in:1/4 out:1/8
        pre  = self.conv2(out)     # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out  = self.conv3(pre)     # in:1/8 out:1/16
        out  = self.conv4(out)     # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out  = self.conv6(post)    # in:1/8 out:1/4

        return out, pre, post


class PSMNet(nn.Module):
    def __init__(self, maxdisp, dem_mode=False, in_channels=3,
                 residual_window=8, geom_channels=5):
        """Initialise PSMNet (stacked hourglass).

        Args:
            maxdisp (int): Maximum disparity (original stereo mode).
            dem_mode (bool): If True, activate DOM-Anchored Dual-Reference
                mode.  Default False preserves original behaviour.
            in_channels (int): Input channels for feature extraction.
                Ignored when ``dem_mode=True`` (auto-set to 1+geom_channels).
            residual_window (int): Half-window K for the 2-D residual cost
                volume in DEM mode (total 2K+1 steps).  Default 8.
            geom_channels (int): Number of geometry input channels N
                (slope, cosθ, nx, ny, nz → default 5).  Used only when
                ``dem_mode=True``.
        """
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.dem_mode = dem_mode
        self.residual_window = residual_window

        if dem_mode:
            # 1 grayscale channel + N geometry channels
            in_channels = 1 + geom_channels

        self.feature_extraction = feature_extraction(in_channels=in_channels)

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        if dem_mode:
            # Learnable fusion of left & right ΔZ into a single output
            self.dz_fusion = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1, bias=True),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_standard_cost(self, refimg_fea, targetimg_fea):
        """Original horizontal-shift integer cost volume (stereo mode)."""
        cost = refimg_fea.new_zeros(
            refimg_fea.size()[0],
            refimg_fea.size()[1] * 2,
            self.maxdisp // 4,
            refimg_fea.size()[2],
            refimg_fea.size()[3],
        )
        for i in range(self.maxdisp // 4):
            if i > 0:
                cost[:, :refimg_fea.size()[1], i, :, i:]  = refimg_fea[:, :, :, i:]
                cost[:, refimg_fea.size()[1]:, i, :, i:]  = targetimg_fea[:, :, :, :-i]
            else:
                cost[:, :refimg_fea.size()[1], i, :, :]   = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :, :]   = targetimg_fea
        return cost.contiguous()

    def _downsample_aux(self, tensor, target_h, target_w):
        """Bilinear down-sample an auxiliary map to feature resolution."""
        return F.interpolate(tensor, size=(target_h, target_w),
                             mode='bilinear', align_corners=True)

    def _aggregate_cost(self, cost, img_h, img_w, num_disp):
        """Run stacked hourglass 3-D aggregation.

        Returns (pred1, pred2, pred3) in training mode, or pred3 in eval mode.
        ``num_disp`` is the disparity dimension size.

        The hourglass applies two rounds of stride-2 3-D convolutions on the
        disparity axis, so it requires that dimension to be divisible by 4.
        If ``num_disp`` is not divisible by 4 (as is common when using small
        residual windows), the cost volume is zero-padded along the disparity
        axis before aggregation and cropped back afterwards.
        """
        # Pad disparity dim to nearest multiple of 4
        pad_d = (4 - num_disp % 4) % 4
        if pad_d > 0:
            cost = F.pad(cost, (0, 0, 0, 0, 0, pad_d))  # pad last disparity dim
        num_disp_pad = num_disp + pad_d

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        if self.training:
            cost1 = F.interpolate(cost1, [num_disp_pad, img_h, img_w],
                                  mode='trilinear', align_corners=False)
            cost2 = F.interpolate(cost2, [num_disp_pad, img_h, img_w],
                                  mode='trilinear', align_corners=False)
            # Crop back to original num_disp if we padded
            if pad_d > 0:
                cost1 = cost1[:, :, :num_disp]
                cost2 = cost2[:, :, :num_disp]

            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparityregression(num_disp)(pred1)

            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparityregression(num_disp)(pred2)

        cost3 = F.interpolate(cost3, [num_disp_pad, img_h, img_w],
                              mode='trilinear', align_corners=False)
        if pad_d > 0:
            cost3 = cost3[:, :, :num_disp]
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)
        # 'softmax(c)' learns "similarity"; 'softmax(-c)' learns 'matching cost'.
        # Either works due to feature-based cost volume flexibility.
        pred3 = disparityregression(num_disp)(pred3)

        if self.training:
            return pred1, pred2, pred3
        else:
            return pred3

    # ------------------------------------------------------------------
    # DEM branch: regress residual displacement, invert ΔZ
    # ------------------------------------------------------------------

    def _dem_branch(self, fea_nac, fea_render, view_dir, tan_t, img_h, img_w):
        """Single DEM-residual branch.

        Residual displacement along the view azimuth direction â:
            δ = ΔZ · tan(t) · â
        Inverse:
            ΔZ = |δ| / tan(t)

        The cost volume is centred at zero offset (prior disparity = 0)
        because both NAC and render are anchored to the same DEM.

        Args:
            fea_nac    : [B, C, Hf, Wf] – NAC features at feature resolution
            fea_render : [B, C, Hf, Wf] – rendered image features
            view_dir   : [B, 2, H, W]   – (ax,ay) azimuth direction, image res
            tan_t      : [B, 1, H, W]   – tan(zenith angle), image res
            img_h, img_w: original image height/width

        Returns (training):
            (dz1, dz2, dz3)  – multi-scale ΔZ [B,1,H,W] each
            (pred1,pred2,pred3) – raw hourglass outputs (scalar offset index)
        Returns (eval):
            dz3  – [B,1,H,W]
            pred3
        """
        K = self.residual_window
        num_disp = 2 * K + 1
        Hf, Wf = fea_nac.shape[2], fea_nac.shape[3]

        # Down-sample geometry maps to feature resolution (1/4 of image)
        vd_fea  = self._downsample_aux(view_dir, Hf, Wf)   # [B,2,Hf,Wf]
        tnt_fea = self._downsample_aux(tan_t, Hf, Wf)       # [B,1,Hf,Wf] (unused in cost; used after)

        # Build 2-D residual cost volume centred at zero displacement
        cost = residual_cost_volume_2d(fea_nac, fea_render, vd_fea, K)

        # 3-D hourglass aggregation → scalar index in [0, 2K]
        agg_result = self._aggregate_cost(cost, img_h, img_w, num_disp)

        tan_t_safe = tan_t.clamp(min=1e-4)

        if self.training:
            pred1, pred2, pred3 = agg_result
            # Centre: subtract K so that 0 offset → residual displacement = 0
            delta1 = pred1 - K   # [B,1,H,W] signed pixel residual
            delta2 = pred2 - K
            delta3 = pred3 - K
            # Invert: ΔZ = |δ| / tan(t)
            dz1 = torch.abs(delta1) / tan_t_safe
            dz2 = torch.abs(delta2) / tan_t_safe
            dz3 = torch.abs(delta3) / tan_t_safe
            return (dz1, dz2, dz3), (pred1, pred2, pred3)
        else:
            pred3 = agg_result
            delta3 = pred3 - K
            dz3 = torch.abs(delta3) / tan_t_safe
            return dz3, pred3

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------

    def forward(self, left=None, right=None,
                nac_l=None, render_l=None, nac_r=None, render_r=None,
                view_dir_l=None, view_dir_r=None,
                tan_t_l=None, tan_t_r=None,
                mask_l=None, mask_r=None):
        """Forward pass – two operating modes.

        **Stereo mode** (``dem_mode=False``, default):
            model(left, right)
            → pred3  [eval]  or  (pred1, pred2, pred3)  [train]

        **DEM Dual-Reference mode** (``dem_mode=True``):
            model(nac_l=..., render_l=..., nac_r=..., render_r=...,
                  view_dir_l=..., view_dir_r=...,
                  tan_t_l=..., tan_t_r=...)
            → dz_fused [eval]  or  (dz_l_all, dz_r_all, dz_fused) [train]

        DEM-mode returns (training):
            dz_l_all : tuple (dz1_L, dz2_L, dz3_L)  [B,1,H,W] each
            dz_r_all : tuple (dz1_R, dz2_R, dz3_R)
            dz_fused : [B, 1, H, W]  fused ΔZ from both views

        Args (DEM mode):
            nac_l, render_l : [B, 1+N, H, W]  left NAC + DEM render
            nac_r, render_r : [B, 1+N, H, W]  right NAC + DEM render
            view_dir_l/r    : [B, 2, H, W]    per-pixel (ax,ay) view azimuth
            tan_t_l/r       : [B, 1, H, W]    per-pixel tan(viewing zenith)
            mask_l/r        : [B, 1, H, W]    optional valid-pixel masks
                              (0 = occluded/shadowed; not yet used internally
                               but passed in for future loss weighting)
        """
        if not self.dem_mode:
            # ---- Original stereo path ----
            refimg_fea    = self.feature_extraction(left)
            targetimg_fea = self.feature_extraction(right)
            cost = self._build_standard_cost(refimg_fea, targetimg_fea)
            return self._aggregate_cost(cost, left.size(2), left.size(3),
                                        self.maxdisp)

        # ---- DOM Dual-Reference DEM forward ----
        img_h, img_w = nac_l.shape[2], nac_l.shape[3]

        # Shared feature extraction for all four images
        fea_nac_l    = self.feature_extraction(nac_l)
        fea_render_l = self.feature_extraction(render_l)
        fea_nac_r    = self.feature_extraction(nac_r)
        fea_render_r = self.feature_extraction(render_r)

        # Branch A: left NAC ↔ left rendered image → ΔZ_L
        dz_l_result, raw_l = self._dem_branch(
            fea_nac_l, fea_render_l, view_dir_l, tan_t_l, img_h, img_w)

        # Branch B: right NAC ↔ right rendered image → ΔZ_R
        dz_r_result, raw_r = self._dem_branch(
            fea_nac_r, fea_render_r, view_dir_r, tan_t_r, img_h, img_w)

        if self.training:
            dz_l3 = dz_l_result[2]   # finest-scale ΔZ from left branch
            dz_r3 = dz_r_result[2]   # finest-scale ΔZ from right branch
        else:
            dz_l3 = dz_l_result      # [B,1,H,W]
            dz_r3 = dz_r_result

        # Learnable fusion of both ΔZ estimates
        dz_fused = self.dz_fusion(torch.cat([dz_l3, dz_r3], dim=1))  # [B,1,H,W]

        if self.training:
            return dz_l_result, dz_r_result, dz_fused
        else:
            return dz_fused
