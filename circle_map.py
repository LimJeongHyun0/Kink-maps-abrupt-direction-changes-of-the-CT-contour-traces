"""Step 2: detect the reconstruction (FOV) circle in each slice image, convert the
red-trace segments into circle-normalized coordinates (X,Y in unit disk, Y downward),
and save overlay images for visual verification."""
import os, json
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
Image.MAX_IMAGE_PIXELS = None

tr = json.load(open('/home/claude/traces.json'))

def fit_circle(xs, ys):
    A = np.c_[xs, ys, np.ones(xs.size)]
    b = xs.astype(float)**2 + ys.astype(float)**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0]/2.0, sol[1]/2.0
    R = np.sqrt(max(sol[2] + cx*cx + cy*cy, 1e-9))
    return cx, cy, R

def _fit_from_mask(mask, scale, W, H):
    h, w = mask.shape
    mask = ndi.binary_closing(mask, iterations=3)
    mask = ndi.binary_fill_holes(mask)
    mask = ndi.binary_opening(mask, iterations=5)
    lab, n = ndi.label(mask)
    if n == 0:
        return None
    if n > 1:
        sizes = ndi.sum(mask, lab, range(1, n + 1))
        mask = lab == (1 + int(np.argmax(sizes)))
    frac = mask.mean()
    edge = mask & ~ndi.binary_erosion(mask)
    ys, xs = np.nonzero(edge)
    keepf = (xs > 4) & (xs < w - 5) & (ys > 4) & (ys < h - 5)
    xs, ys = xs[keepf].astype(float), ys[keepf].astype(float)
    if xs.size < 50:
        return None
    cx, cy, R = fit_circle(xs, ys)
    inl = 1.0
    for it in range(6):
        d = np.hypot(xs - cx, ys - cy)
        res = np.abs(d - R)
        tol = max(0.01 * R, 2.0) if it > 1 else max(0.03 * R, 4.0)
        keep = res < tol
        if keep.sum() < 30:
            break
        cx, cy, R = fit_circle(xs[keep], ys[keep])
        inl = float(keep.mean())
    area_R = np.sqrt(mask.sum() / np.pi)
    return dict(W=W, H=H, cx=(cx + 0.5) * scale, cy=(cy + 0.5) * scale, R=R * scale,
                areaR=area_R * scale, inlier=inl, frac=float(frac))

def detect_fov(path):
    im = Image.open(path).convert('L')
    W, H = im.size
    a = np.asarray(im).astype(np.float32)
    scale = 1
    if max(W, H) > 1200:
        scale = int(np.ceil(max(W, H) / 900))
        a = a[::scale, ::scale]
    h, w = a.shape
    c = 40
    # --- method A: texture (local std); background of screenshots is flat ---
    m = ndi.uniform_filter(a, 7); m2 = ndi.uniform_filter(a * a, 7)
    lstd = np.sqrt(np.maximum(m2 - m * m, 0))
    corners = [lstd[:c, :c], lstd[:c, -c:], lstd[-c:, :c], lstd[-c:, -c:]]
    bg = min(np.median(x) for x in corners)
    resA = None
    if bg < 1.0:
        resA = _fit_from_mask(lstd > max(3 * bg, 0.75), scale, W, H)
        if resA is not None and not (0.25 < resA['frac'] < 0.97 and resA['inlier'] > 0.5):
            resA = None
    if resA is not None:
        resA['method'] = 'texture'
        return resA
    # --- method B: intensity (dithered / quantized exports): smooth out the dither ---
    s = ndi.gaussian_filter(a, 5)
    thr = threshold_otsu(s)
    cornersI = [s[:c, :c], s[:c, -c:], s[-c:, :c], s[-c:, -c:]]
    bgI = np.median([np.median(x) for x in cornersI])
    mask = (s > thr) if bgI < thr else (s < thr)
    resB = _fit_from_mask(mask, scale, W, H)
    if resB is None:
        raise RuntimeError('FOV detection failed: ' + path)
    resB['method'] = 'intensity'
    return resB

os.makedirs('/home/claude/check', exist_ok=True)
for name, S in tr.items():
    for rec in S['slides']:
        f = detect_fov(rec['image'])
        rec['fov'] = f
    # ---- set-level outlier repair: a slice whose circle deviates > 3% of W from the
    #      set median is a detection failure -> replace with the median (in relative units)
    rel = np.array([[r['fov']['cx'] / r['fov']['W'], r['fov']['cy'] / r['fov']['H'],
                     r['fov']['R'] / r['fov']['W']] for r in S['slides']])
    med = np.median(rel, axis=0)
    for r, v in zip(S['slides'], rel):
        if np.any(np.abs(v - med) > 0.03):
            f = r['fov']
            print('   repaired %s slice %d: (%.3f,%.3f,%.3f) -> median (%.3f,%.3f,%.3f)' % (
                name, r['idx'], v[0], v[1], v[2], med[0], med[1], med[2]))
            f['cx'] = med[0] * f['W']; f['cy'] = med[1] * f['H']; f['R'] = med[2] * f['W']
            f['method'] += '+median'
    # ---- map traces into the unit-circle frame (after repair) ----
    for rec in S['slides']:
        f = rec['fov']
        W, H = f['W'], f['H']
        segs = []
        for u0, v0, u1, v1 in rec['lines']:
            X0 = (u0 * W - f['cx']) / f['R']; Y0 = (v0 * H - f['cy']) / f['R']
            X1 = (u1 * W - f['cx']) / f['R']; Y1 = (v1 * H - f['cy']) / f['R']
            segs.append([X0, Y0, X1, Y1])
        rec['segs_circ'] = segs
    Rs = [r['fov']['R'] / r['fov']['W'] for r in S['slides']]
    cxs = [r['fov']['cx'] / r['fov']['W'] for r in S['slides']]
    cys = [r['fov']['cy'] / r['fov']['H'] for r in S['slides']]
    print('%-14s R/W=%.3f..%.3f  cx/W=%.3f..%.3f  cy/H=%.3f..%.3f  areaR/R=%.3f..%.3f' % (
        name, min(Rs), max(Rs), min(cxs), max(cxs), min(cys), max(cys),
        min(r['fov']['areaR']/r['fov']['R'] for r in S['slides']),
        max(r['fov']['areaR']/r['fov']['R'] for r in S['slides'])))

json.dump(tr, open('/home/claude/traces_circ.json', 'w'), indent=1)

# ---- overlay check sheets: raw slice + extracted red lines + detected circle ----
def overlay_sheet(name, out):
    S = tr[name]
    tiles = []
    for rec in S['slides']:
        im = Image.open(rec['image']).convert('RGB')
        W, H = im.size
        f = rec['fov']
        dr = ImageDraw.Draw(im)
        lw = max(2, W // 300)
        for u0, v0, u1, v1 in rec['lines']:
            dr.line([(u0 * W, v0 * H), (u1 * W, v1 * H)], fill=(255, 0, 0), width=lw)
        dr.ellipse([f['cx'] - f['R'], f['cy'] - f['R'], f['cx'] + f['R'], f['cy'] + f['R']],
                   outline=(0, 255, 0), width=lw)
        tiles.append(im.resize((300, 300)))
    sheet = Image.new('RGB', (300 * 5, 300 * 2), (255, 255, 255))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 5) * 300, (i // 5) * 300))
    sheet.save(out)

for nm in tr:
    overlay_sheet(nm, '/home/claude/check/overlay_%s.png' % nm)
print('saved overlays')
