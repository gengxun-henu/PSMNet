from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
from models import *
from losses import total_dem_loss

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/SceneFlowData/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default=None,
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

# ---- DOM Dual-Reference DEM mode ----
parser.add_argument('--dem_mode', action='store_true', default=False,
                    help='Enable DOM-Anchored Dual-Reference DEM correction mode. '
                         'When active, the model expects four images '
                         '(NAC_L, render_L, NAC_R, render_R) plus geometry '
                         'fields instead of the standard stereo pair.')
parser.add_argument('--residual_window', type=int, default=8,
                    help='Half-window K for the 2-D residual cost volume in '
                         'DEM mode (total 2K+1 displacement steps, default 8).')
parser.add_argument('--geom_channels', type=int, default=5,
                    help='Number of geometry input channels N appended to '
                         'each image in DEM mode '
                         '(default 5: slope, cosθ, nx, ny, nz).')
parser.add_argument('--use_view_dir', action='store_true', default=True,
                    help='Use per-pixel view azimuth direction in DEM mode '
                         '(default True).')
parser.add_argument('--dom_train_list', default=None,
                    help='Path to CSV file listing DEM-mode training samples '
                         '(see dataloader/dom_listfile.py).')
parser.add_argument('--dom_test_list', default=None,
                    help='Path to CSV file listing DEM-mode test samples.')

# ---- DEM loss weights ----
parser.add_argument('--lambda_lr', type=float, default=1.0,
                    help='Weight for left-right ΔZ consistency loss '
                         '(L_lr, core geometric constraint).')
parser.add_argument('--lambda_smooth', type=float, default=0.1,
                    help='Weight for edge-aware slope smoothness loss (L_smooth).')
parser.add_argument('--lambda_slope', type=float, default=0.1,
                    help='Weight for slope-prior consistency loss (L_slope).')
parser.add_argument('--lambda_photo', type=float, default=1.0,
                    help='Weight for photometric self-supervision loss (L_photo). '
                         'Set to 0 to disable.')
parser.add_argument('--lambda_data', type=float, default=1.0,
                    help='Weight for supervised ΔZ data term (L_data). '
                         'Active only when reference dz_gt is provided.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

if args.dem_mode:
    # DOM Dual-Reference mode
    from dataloader.dom_loader import DOMDataset

    if args.dom_train_list is None or args.dom_test_list is None:
        raise ValueError(
            '--dom_train_list and --dom_test_list are required in --dem_mode')

    TrainImgLoader = torch.utils.data.DataLoader(
        DOMDataset(args.dom_train_list, datapath=args.datapath,
                   training=True, geom_channels=args.geom_channels),
        batch_size=4, shuffle=True, num_workers=4, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DOMDataset(args.dom_test_list, datapath=args.datapath,
                   training=False, geom_channels=args.geom_channels),
        batch_size=2, shuffle=False, num_workers=2, drop_last=False)
else:
    # Standard SceneFlow stereo mode
    all_left_img, all_right_img, all_left_disp, \
        test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath)

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True),
        batch_size=12, shuffle=True, num_workers=8, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=8, shuffle=False, num_workers=4, drop_last=False)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

if args.model == 'stackhourglass':
    model = stackhourglass(
        args.maxdisp,
        dem_mode=args.dem_mode,
        residual_window=args.residual_window,
        geom_channels=args.geom_channels,
    )
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'])

print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_stereo(imgL, imgR, disp_L):
    """Original stereo training step."""
    model.train()
    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    mask = disp_true < args.maxdisp
    mask.detach_()
    optimizer.zero_grad()

    if args.model == 'stackhourglass':
        output1, output2, output3 = model(imgL, imgR)
        output1 = torch.squeeze(output1, 1)
        output2 = torch.squeeze(output2, 1)
        output3 = torch.squeeze(output3, 1)
        loss = (0.5 * F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True)
                + 0.7 * F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True)
                + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True))
    elif args.model == 'basic':
        output = model(imgL, imgR)
        output = torch.squeeze(output, 1)
        loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

    loss.backward()
    optimizer.step()
    return loss.data


def train_dem(batch):
    """DEM Dual-Reference training step.

    Expects a batch dict from DOMDataset.
    Computes total_dem_loss and back-propagates.
    """
    model.train()

    def to_cuda(t):
        return t.cuda() if (args.cuda and t is not None) else t

    nac_l    = to_cuda(batch['nac_l'])
    render_l = to_cuda(batch['render_l'])
    nac_r    = to_cuda(batch['nac_r'])
    render_r = to_cuda(batch['render_r'])
    vd_l     = to_cuda(batch['view_dir_l'])
    vd_r     = to_cuda(batch['view_dir_r'])
    tnt_l    = to_cuda(batch['tan_t_l'])
    tnt_r    = to_cuda(batch['tan_t_r'])
    mask_l   = to_cuda(batch['mask_l'])
    mask_r   = to_cuda(batch['mask_r'])

    # Optional fields
    dz_gt         = to_cuda(batch.get('dz_gt'))
    dem_init      = to_cuda(batch.get('dem_init'))
    sun_dir_l     = to_cuda(batch.get('sun_dir_l'))
    normal_base_l = to_cuda(batch.get('normal_base_l'))

    optimizer.zero_grad()

    dz_l_all, dz_r_all, dz_fused = model(
        nac_l=nac_l, render_l=render_l,
        nac_r=nac_r, render_r=render_r,
        view_dir_l=vd_l, view_dir_r=vd_r,
        tan_t_l=tnt_l, tan_t_r=tnt_r,
        mask_l=mask_l, mask_r=mask_r,
    )

    # Guide image for edge-aware smoothness (use left NAC intensity channel)
    guide = nac_l[:, :1]  # [B, 1, H, W]

    loss, loss_dict = total_dem_loss(
        dz_l_all, dz_r_all, dz_fused,
        dz_gt=dz_gt,
        nac_l=nac_l,
        dem_init=dem_init,
        sun_dir_l=sun_dir_l,
        normal_base_l=normal_base_l,
        slope_prior=None,  # can pass a slope tensor if available
        guide_img=guide,
        mask_l=mask_l,
        mask_r=mask_r,
        lambda_lr=args.lambda_lr,
        lambda_smooth=args.lambda_smooth,
        lambda_slope=args.lambda_slope,
        lambda_photo=args.lambda_photo if dem_init is not None else 0.0,
        lambda_data=args.lambda_data if dz_gt is not None else 0.0,
    )

    loss.backward()
    optimizer.step()
    return loss.data, loss_dict


def train(batch_or_imgs, *extra):
    if args.dem_mode:
        return train_dem(batch_or_imgs)
    else:
        return train_stereo(batch_or_imgs, *extra), {}


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

def test_stereo(imgL, imgR, disp_true):
    model.eval()
    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()

    mask = disp_true < 192

    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2] // 16
        top_pad = (times + 1) * 16 - imgL.shape[2]
    else:
        top_pad = 0

    if imgL.shape[3] % 16 != 0:
        times = imgL.shape[3] // 16
        right_pad = (times + 1) * 16 - imgL.shape[3]
    else:
        right_pad = 0

    imgL = F.pad(imgL, (0, right_pad, top_pad, 0))
    imgR = F.pad(imgR, (0, right_pad, top_pad, 0))

    with torch.no_grad():
        output3 = model(imgL, imgR)
        output3 = torch.squeeze(output3)

    if top_pad != 0:
        img = output3[:, top_pad:, :]
    else:
        img = output3

    if len(disp_true[mask]) == 0:
        loss = 0
    else:
        loss = F.l1_loss(img[mask], disp_true[mask])

    return loss.data.cpu()


def test_dem(batch):
    model.eval()

    def to_cuda(t):
        return t.cuda() if (args.cuda and t is not None) else t

    nac_l    = to_cuda(batch['nac_l'])
    render_l = to_cuda(batch['render_l'])
    nac_r    = to_cuda(batch['nac_r'])
    render_r = to_cuda(batch['render_r'])
    vd_l     = to_cuda(batch['view_dir_l'])
    vd_r     = to_cuda(batch['view_dir_r'])
    tnt_l    = to_cuda(batch['tan_t_l'])
    tnt_r    = to_cuda(batch['tan_t_r'])
    mask_l   = to_cuda(batch['mask_l'])
    mask_r   = to_cuda(batch['mask_r'])
    dz_gt    = to_cuda(batch.get('dz_gt'))

    with torch.no_grad():
        dz_fused = model(
            nac_l=nac_l, render_l=render_l,
            nac_r=nac_r, render_r=render_r,
            view_dir_l=vd_l, view_dir_r=vd_r,
            tan_t_l=tnt_l, tan_t_r=tnt_r,
            mask_l=mask_l, mask_r=mask_r,
        )

    if dz_gt is not None:
        valid = (mask_l > 0) & (mask_r > 0)
        if valid.any():
            loss = F.l1_loss(dz_fused[valid], dz_gt[valid])
        else:
            loss = dz_fused.new_tensor(0.0)
        return loss.data.cpu()
    return 0.0


def test(batch_or_imgs, *extra):
    if args.dem_mode:
        return test_dem(batch_or_imgs)
    else:
        return test_stereo(batch_or_imgs, *extra)


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    start_full_time = time.time()
    for epoch in range(0, args.epochs):
        print('This is %d-th epoch' % epoch)
        total_train_loss = 0
        adjust_learning_rate(optimizer, epoch)

        ## training ##
        if args.dem_mode:
            for batch_idx, batch in enumerate(TrainImgLoader):
                start_time = time.time()
                loss, loss_dict = train(batch)
                print('Iter %d training loss = %.3f , time = %.2f  %s'
                      % (batch_idx, loss,
                         time.time() - start_time,
                         '  '.join(f'{k}={v:.4f}' for k, v in loss_dict.items())))
                total_train_loss += loss
        else:
            for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
                start_time = time.time()
                loss, _ = train(imgL_crop, imgR_crop, disp_crop_L)
                print('Iter %d training loss = %.3f , time = %.2f'
                      % (batch_idx, loss, time.time() - start_time))
                total_train_loss += loss

        print('epoch %d total training loss = %.3f'
              % (epoch, total_train_loss / len(TrainImgLoader)))

        # SAVE
        savefilename = args.savemodel + '/checkpoint_' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss / len(TrainImgLoader),
        }, savefilename)

    print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))

    # ------------- TEST ----------------------------------------------------
    total_test_loss = 0
    if args.dem_mode:
        for batch_idx, batch in enumerate(TestImgLoader):
            test_loss = test(batch)
            print('Iter %d test loss = %.3f' % (batch_idx, test_loss))
            total_test_loss += test_loss
    else:
        for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
            test_loss = test(imgL, imgR, disp_L)
            print('Iter %d test loss = %.3f' % (batch_idx, test_loss))
            total_test_loss += test_loss

    print('total test loss = %.3f' % (total_test_loss / len(TestImgLoader)))

    # SAVE test information
    savefilename = args.savemodel + 'testinformation.tar'
    torch.save({
        'test_loss': total_test_loss / len(TestImgLoader),
    }, savefilename)


if __name__ == '__main__':
    main()
