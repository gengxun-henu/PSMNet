"""DOM-Anchored Dual-Reference data loader for LRO NAC / planetary imagery.

Directory Convention
--------------------
Each sample lives in its own sub-directory (or uses flat file lists).
Expected files per sample:

    nac_l.tif         – left  LRO NAC grayscale image
    nac_r.tif         – right LRO NAC grayscale image
    render_l.tif      – DEM-rendered image (same geometry/lighting as nac_l)
    render_r.tif      – DEM-rendered image (same geometry/lighting as nac_r)
    geom.tif          – 5-band geometry raster [slope, cosθ, nx, ny, nz]
                        (bands: 0=slope, 1=cos(incidence), 2..4=surface normal)
    view_dir_l.tif    – 2-band view-azimuth direction for left  view [ax, ay]
    view_dir_r.tif    – 2-band view-azimuth direction for right view [ax, ay]
    tan_t_l.tif       – 1-band tan(viewing zenith) for left  view
    tan_t_r.tif       – 1-band tan(viewing zenith) for right view
    mask_l.tif        – 1-band uint8 valid mask for left  view (0=invalid)
    mask_r.tif        – 1-band uint8 valid mask for right view (0=invalid)
    dz_gt.tif         – (optional) 1-band float32 reference ΔZ for supervised
                        training (difference between refined and initial DEM)
    dem_init.tif      – (optional) initial DEM height field [float32]
    sun_dir_l.tif     – (optional) 3-band unit solar direction [sx, sy, sz]
    normal_base_l.tif – (optional) 3-band surface normal from initial DEM

File-list format (CSV, one sample per line)
--------------------------------------------
Modelled after ``listflowfile.py``.  Each line is a comma-separated list of
absolute (or relative-to-datapath) file paths:

    nac_l, nac_r, render_l, render_r, geom, view_dir_l, view_dir_r, tan_t_l, tan_t_r, mask_l, mask_r[, dz_gt[, dem_init[, sun_dir_l[, normal_base_l]]]]

Optional columns at the end can be omitted; the loader treats them as None.

Usage
-----
    from dataloader.dom_loader import DOMDataset
    dataset = DOMDataset(file_list_path, patch_size=(256, 256), training=True)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in loader:
        ...

Dependencies
------------
Requires ``rasterio`` for multi-band GeoTIFF reading (pip install rasterio).
Falls back to PIL for single-band images when rasterio is unavailable.
"""

import os
import csv
import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF

try:
    import rasterio
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False
    from PIL import Image


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_tif(path, expected_bands=None):
    """Read a GeoTIFF (or PNG/JPEG) file and return a float32 numpy array.

    Shape: (bands, H, W).
    If rasterio is unavailable, falls back to PIL (single-band only).
    """
    if _HAS_RASTERIO:
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)  # (bands, H, W)
    else:
        from PIL import Image as _Image
        img = _Image.open(path)
        arr = np.array(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis]          # (1, H, W)
        else:
            arr = arr.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)

    if expected_bands is not None and arr.shape[0] != expected_bands:
        raise ValueError(
            f"Expected {expected_bands} bands in {path}, got {arr.shape[0]}")
    return arr


def _to_tensor(arr):
    """Convert (C, H, W) float32 numpy array to [C, H, W] float32 tensor."""
    return torch.from_numpy(arr)


def _normalise_intensity(arr, mean=0.445, std=0.269):
    """Normalise a single-band image (mean/std similar to grayscale ImageNet)."""
    return (arr - mean) / std


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DOMDataset(data.Dataset):
    """Torch Dataset for DOM-Anchored Dual-Reference DEM correction.

    Each sample contains:
        nac_l       : [1+N, H, W]  left NAC grayscale + N geometry channels
        render_l    : [1+N, H, W]  left rendered image + N geometry channels
        nac_r       : [1+N, H, W]  right NAC + N geometry channels
        render_r    : [1+N, H, W]  right rendered image + N geometry channels
        view_dir_l  : [2, H, W]    (ax, ay) view azimuth direction – left
        view_dir_r  : [2, H, W]    (ax, ay) view azimuth direction – right
        tan_t_l     : [1, H, W]    tan(zenith) – left
        tan_t_r     : [1, H, W]    tan(zenith) – right
        mask_l      : [1, H, W]    valid mask  – left
        mask_r      : [1, H, W]    valid mask  – right

    Optional (set to zero tensor if not provided):
        dz_gt       : [1, H, W]    reference ΔZ
        dem_init    : [1, H, W]    initial DEM
        sun_dir_l   : [3, H, W]    solar direction for left view
        normal_base_l:[3, H, W]   surface normals from initial DEM

    Args:
        file_list   (str): Path to CSV file listing samples (see module doc).
        datapath    (str): Root directory prepended to relative file paths.
        patch_size  (tuple): (H_patch, W_patch) for random crop during training.
        training    (bool): If True, apply random crop augmentation.
        geom_channels (int): Number of geometry channels N (default 5).
        intensity_mean (float): Normalisation mean for grayscale intensity.
        intensity_std  (float): Normalisation std  for grayscale intensity.
    """

    CSV_COLS = [
        'nac_l', 'nac_r', 'render_l', 'render_r',
        'geom', 'view_dir_l', 'view_dir_r',
        'tan_t_l', 'tan_t_r',
        'mask_l', 'mask_r',
        'dz_gt',          # optional
        'dem_init',       # optional
        'sun_dir_l',      # optional
        'normal_base_l',  # optional
    ]

    def __init__(self, file_list, datapath='', patch_size=(256, 256),
                 training=True, geom_channels=5,
                 intensity_mean=0.445, intensity_std=0.269):
        self.datapath = datapath
        self.patch_size = patch_size
        self.training = training
        self.geom_channels = geom_channels
        self.intensity_mean = intensity_mean
        self.intensity_std = intensity_std

        self.samples = self._parse_file_list(file_list)

    def _parse_file_list(self, file_list):
        samples = []
        with open(file_list, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                row = [r.strip() for r in row]
                if not row or row[0].startswith('#'):
                    continue  # skip comments / blank lines
                if len(row) < 11:
                    raise ValueError(
                        f"Each CSV row must have at least 11 columns; "
                        f"got {len(row)}: {row}")
                entry = {}
                for i, col in enumerate(self.CSV_COLS):
                    if i < len(row) and row[i]:
                        path = row[i]
                        if self.datapath and not os.path.isabs(path):
                            path = os.path.join(self.datapath, path)
                        entry[col] = path
                    else:
                        entry[col] = None
                samples.append(entry)
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_and_normalise_nac(self, path):
        """Read grayscale NAC or render image; return (1, H, W) normalised."""
        arr = _read_tif(path, expected_bands=1)
        arr = _normalise_intensity(arr, self.intensity_mean, self.intensity_std)
        return arr  # (1, H, W)

    def _load_geom(self, path):
        """Read N-band geometry raster; return (N, H, W)."""
        return _read_tif(path, expected_bands=self.geom_channels)

    def _random_crop_indices(self, h, w):
        ph, pw = self.patch_size
        y0 = random.randint(0, h - ph)
        x0 = random.randint(0, w - pw)
        return y0, x0

    def _crop(self, arr, y0, x0):
        ph, pw = self.patch_size
        return arr[:, y0:y0 + ph, x0:x0 + pw]

    def __getitem__(self, index):
        s = self.samples[index]

        # --- Read all required bands ---
        nac_l_raw    = self._load_and_normalise_nac(s['nac_l'])    # (1,H,W)
        nac_r_raw    = self._load_and_normalise_nac(s['nac_r'])
        render_l_raw = self._load_and_normalise_nac(s['render_l'])
        render_r_raw = self._load_and_normalise_nac(s['render_r'])
        geom         = self._load_geom(s['geom'])                  # (N,H,W)

        view_dir_l = _read_tif(s['view_dir_l'], expected_bands=2)
        view_dir_r = _read_tif(s['view_dir_r'], expected_bands=2)
        tan_t_l    = _read_tif(s['tan_t_l'], expected_bands=1)
        tan_t_r    = _read_tif(s['tan_t_r'], expected_bands=1)
        mask_l     = _read_tif(s['mask_l'], expected_bands=1)
        mask_r     = _read_tif(s['mask_r'], expected_bands=1)

        # Optional fields
        dz_gt         = _read_tif(s['dz_gt'])          if s['dz_gt']         else None
        dem_init      = _read_tif(s['dem_init'])        if s['dem_init']      else None
        sun_dir_l     = _read_tif(s['sun_dir_l'])       if s['sun_dir_l']     else None
        normal_base_l = _read_tif(s['normal_base_l'])   if s['normal_base_l'] else None

        H, W = nac_l_raw.shape[1], nac_l_raw.shape[2]

        # --- Concatenate image + geometry channels ---
        # Shape: (1+N, H, W)
        nac_l_full    = np.concatenate([nac_l_raw,    geom], axis=0)
        render_l_full = np.concatenate([render_l_raw, geom], axis=0)
        nac_r_full    = np.concatenate([nac_r_raw,    geom], axis=0)
        render_r_full = np.concatenate([render_r_raw, geom], axis=0)

        # --- Optional random crop (training augmentation) ---
        if self.training:
            ph, pw = self.patch_size
            if H < ph or W < pw:
                raise ValueError(
                    f"Image ({H}x{W}) smaller than patch ({ph}x{pw})")
            y0, x0 = self._random_crop_indices(H, W)
            crop = lambda a: self._crop(a, y0, x0)

            nac_l_full    = crop(nac_l_full)
            render_l_full = crop(render_l_full)
            nac_r_full    = crop(nac_r_full)
            render_r_full = crop(render_r_full)
            view_dir_l    = crop(view_dir_l)
            view_dir_r    = crop(view_dir_r)
            tan_t_l       = crop(tan_t_l)
            tan_t_r       = crop(tan_t_r)
            mask_l        = crop(mask_l)
            mask_r        = crop(mask_r)
            if dz_gt         is not None: dz_gt         = crop(dz_gt)
            if dem_init      is not None: dem_init      = crop(dem_init)
            if sun_dir_l     is not None: sun_dir_l     = crop(sun_dir_l)
            if normal_base_l is not None: normal_base_l = crop(normal_base_l)

        # --- Convert to tensors ---
        def to_t(a):
            return _to_tensor(a) if a is not None else None

        sample = {
            'nac_l':        _to_tensor(nac_l_full.astype(np.float32)),
            'render_l':     _to_tensor(render_l_full.astype(np.float32)),
            'nac_r':        _to_tensor(nac_r_full.astype(np.float32)),
            'render_r':     _to_tensor(render_r_full.astype(np.float32)),
            'view_dir_l':   _to_tensor(view_dir_l.astype(np.float32)),
            'view_dir_r':   _to_tensor(view_dir_r.astype(np.float32)),
            'tan_t_l':      _to_tensor(tan_t_l.astype(np.float32)),
            'tan_t_r':      _to_tensor(tan_t_r.astype(np.float32)),
            'mask_l':       _to_tensor(mask_l.astype(np.float32)),
            'mask_r':       _to_tensor(mask_r.astype(np.float32)),
            'dz_gt':        to_t(dz_gt.astype(np.float32)) if dz_gt is not None else None,
            'dem_init':     to_t(dem_init.astype(np.float32)) if dem_init is not None else None,
            'sun_dir_l':    to_t(sun_dir_l.astype(np.float32)) if sun_dir_l is not None else None,
            'normal_base_l':to_t(normal_base_l.astype(np.float32)) if normal_base_l is not None else None,
        }
        return sample
