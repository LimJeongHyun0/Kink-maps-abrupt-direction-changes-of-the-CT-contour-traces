"""Kink maps: highlight where straight contour elements abruptly change direction.

Definition
  * trace segments are chained into polylines (shared end points, tol 0.02 R)
  * at every interior joint v the turning angle kappa_v (0 = straight continuation,
    90 = right-angle bend, 180 = reversal) between the incoming and outgoing element is measured
  * along the polyline, the local kink intensity is the Gaussian-weighted total turning within
    ~0.03 R:  I(s) = sum_v kappa_v * exp(-(s - s_v)^2 / (2 sigma^2))
        -> straight parts  = 0  (blue, hue 225)
        -> sharp cusp      = kappa (red at >= 90 deg, hue 0)
  * surface version: adaptive-kernel local mean of I sampled along the traces
"""
import os, json, csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
from matplotlib import font_manager
from scipy.ndimage import gaussian_filter, zoom

tr = json.load(open('/home/claude/traces_circ.json'))
OUT = '/home/claude/out_kink'
for sub in ('lines', 'filled', 'filled_faded', 'qa'):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

N, SS, GRID = 1000, 2, 400
CEN, RPX = N / 2.0, 0.45 * N
SAT, VAL, HUE_MAX = 0.80, 1.00, 225.0
KMAX = 60.0                 # turning angle mapped to red (>= 60 deg saturates)
CHAIN_TOL = 0.02            # end-point matching tolerance (unit-disk units)
MIN_SEG = 0.01              # joints closer than this are merged before measuring angles
SIGMA_S = 0.06              # along-line spread of a kink's influence
DS = 0.004                  # resampling step along polylines
SIG_MIN, K_NN, ALPHA = 0.035, 12, 0.9
FONT = font_manager.findfont('DejaVu Sans')

def val_to_rgb(v):
    v = np.asarray(v, dtype=float)
    hue = HUE_MAX * (1.0 - np.clip(v, 0, KMAX) / KMAX) / 360.0
    return mcolors.hsv_to_rgb(np.stack([hue, np.full_like(hue, SAT), np.full_like(hue, VAL)], -1))

# ------------------------------------------------------------------ polylines
def chain_polylines(segs, tol=CHAIN_TOL):
    segs = [np.array([[s[0], s[1]], [s[2], s[3]]]) for s in segs]
    used = [False] * len(segs)
    polys = []
    for i in range(len(segs)):
        if used[i]:
            continue
        used[i] = True
        pts = [segs[i][0], segs[i][1]]
        grown = True
        while grown:
            grown = False
            for j in range(len(segs)):
                if used[j]:
                    continue
                a, b = segs[j]
                if np.hypot(*(pts[-1] - a)) < tol:
                    pts.append(b); used[j] = True; grown = True
                elif np.hypot(*(pts[-1] - b)) < tol:
                    pts.append(a); used[j] = True; grown = True
                elif np.hypot(*(pts[0] - b)) < tol:
                    pts.insert(0, a); used[j] = True; grown = True
                elif np.hypot(*(pts[0] - a)) < tol:
                    pts.insert(0, b); used[j] = True; grown = True
        P = np.array(pts)
        # merge joints that are too close to give a meaningful direction
        keep = [0]
        for k in range(1, len(P)):
            if np.hypot(*(P[k] - P[keep[-1]])) >= MIN_SEG or k == len(P) - 1:
                keep.append(k)
        P = P[keep]
        if len(P) >= 2 and np.hypot(*(P[-1] - P[-2])) < 1e-9:
            P = P[:-1]
        polys.append(P)
    return polys

def polyline_kinks(P):
    """returns arc-length positions s_v and turning angles kappa_v (deg) of interior joints"""
    d = np.diff(P, axis=0)
    L = np.hypot(d[:, 0], d[:, 1])
    s = np.r_[0, np.cumsum(L)]
    kappa, sv = [], []
    for i in range(1, len(P) - 1):
        a, b = d[i - 1], d[i]
        c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        kappa.append(float(np.degrees(np.arccos(np.clip(c, -1, 1)))))
        sv.append(s[i])
    return s, np.array(sv), np.array(kappa)

def sample_polyline(P, s, sv, kappa):
    """dense samples along P with local kink intensity"""
    total = s[-1]
    if total < 1e-9:
        return np.zeros((0, 2)), np.zeros(0)
    ss = np.arange(0, total, DS)
    ss = np.r_[ss, total]
    X = np.interp(ss, s, P[:, 0]); Y = np.interp(ss, s, P[:, 1])
    I = np.zeros_like(ss)
    for sk, kk in zip(sv, kappa):
        I += kk * np.exp(-0.5 * ((ss - sk) / SIGMA_S) ** 2)
    return np.c_[X, Y], I

def analyse(S):
    segs = [s for rec in S['slides'] for s in rec['segs_circ']]
    polys = chain_polylines(segs)
    pts_all, I_all, kinks = [], [], []
    total_len, total_turn = 0.0, 0.0
    for P in polys:
        s, sv, kappa = polyline_kinks(P)
        pts, I = sample_polyline(P, s, sv, kappa)
        pts_all.append(pts); I_all.append(I)
        total_len += s[-1]; total_turn += kappa.sum() if kappa.size else 0.0
        for sk, kk in zip(sv, kappa):
            kinks.append((float(np.interp(sk, s, P[:, 0])), float(np.interp(sk, s, P[:, 1])), float(kk)))
    pts_all = np.vstack(pts_all); I_all = np.concatenate(I_all)
    kinks = np.array(kinks).reshape(-1, 3)
    stats = dict(n_polylines=len(polys), n_joints=int(len(kinks)),
                 n_kinks_ge45=int(np.sum(kinks[:, 2] >= 45)) if kinks.size else 0,
                 n_kinks_ge30=int(np.sum(kinks[:, 2] >= 30)) if kinks.size else 0,
                 total_length_R=float(total_len),
                 turning_per_length=float(total_turn / max(total_len, 1e-9)),     # deg per R of trace
                 kinks45_per_length=float((np.sum(kinks[:, 2] >= 45) if kinks.size else 0) / max(total_len, 1e-9)),
                 frac_len_I_gt30=float(np.mean(I_all > 30)) if I_all.size else 0.0,
                 mean_I=float(I_all.mean()) if I_all.size else 0.0)
    return polys, pts_all, I_all, kinks, stats

# ------------------------------------------------------------------ rendering
def to_px(X, Y, n):
    c, r = n / 2.0, 0.45 * n
    return (c + X * r, c + Y * r)

def render_lines(polys, kinks, stats, label, annotate=True, mark_kinks=False):
    big = N * SS
    im = Image.new('RGB', (big, big), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    c, r = big / 2.0, 0.45 * big
    dr.ellipse([c - r, c - r, c + r, c + r], outline=(0, 0, 0), width=3 * SS)
    # draw low-intensity pieces first, hot pieces on top
    pieces = []
    for P in polys:
        s, sv, kappa = polyline_kinks(P)
        pts, I = sample_polyline(P, s, sv, kappa)
        for k in range(len(pts) - 1):
            pieces.append((0.5 * (I[k] + I[k + 1]), pts[k], pts[k + 1]))
    pieces.sort(key=lambda t: t[0])
    for val, p0, p1 in pieces:
        col = tuple((val_to_rgb(val) * 255 + 0.5).astype(int))
        dr.line([to_px(*p0, big), to_px(*p1, big)], fill=col, width=7 * SS)
        dr.ellipse([to_px(*p1, big)[0] - 3.5 * SS, to_px(*p1, big)[1] - 3.5 * SS,
                    to_px(*p1, big)[0] + 3.5 * SS, to_px(*p1, big)[1] + 3.5 * SS], fill=col)  # round joints
    if mark_kinks:      # ring marker on sharp cusps (>= 60 deg)
        for x, y, kk in kinks:
            if kk >= 60:
                px, py = to_px(x, y, big)
                rr = 14 * SS
                dr.ellipse([px - rr, py - rr, px + rr, py + rr], outline=(200, 0, 0), width=3 * SS)
    if annotate:
        f = ImageFont.truetype(FONT, int(0.028 * big))
        dr.text((0.03 * big, 0.008 * big), 'colour = local turning angle (0° straight → ≥60° sharp cusp)',
                fill=(60, 60, 60), font=f)
        f2 = ImageFont.truetype(FONT, int(0.032 * big))
        dr.text((0.03 * big, 0.958 * big), '%s    cusps ≥45°: %d    turning %.0f°/R' % (
            label, stats['n_kinks_ge45'], stats['turning_per_length']), fill=(40, 40, 40), font=f2)
    return im.resize((N, N), Image.LANCZOS)

yy, xx = np.mgrid[0:N, 0:N]
RR = np.hypot(xx - CEN + 0.5, yy - CEN + 0.5)
INSIDE = RR <= RPX
RING = np.clip(2.0 - np.abs(RR - RPX), 0, 1)[..., None]

def kernel_field(P, V):
    P = P.astype(np.float32); V = V.astype(np.float32)
    g = np.linspace(-1.0, 1.0, GRID, dtype=np.float32)
    GX, GY = np.meshgrid(g, g)
    Q = np.c_[GX.ravel(), GY.ravel()]
    rq = np.hypot(Q[:, 0], Q[:, 1])
    F = np.full(Q.shape[0], np.nan, dtype=np.float32)
    Dmin = np.full(Q.shape[0], 2.0, dtype=np.float32)
    sel = np.nonzero(rq <= 1.03)[0]
    for i in range(0, sel.size, 2500):
        idx = sel[i:i + 2500]
        D = np.sqrt(((Q[idx, None, :] - P[None, :, :]) ** 2).sum(-1))
        dk = np.partition(D, K_NN, axis=1)[:, K_NN]
        Dmin[idx] = D.min(1)
        sig = np.maximum(SIG_MIN, ALPHA * dk)[:, None]
        W = np.exp(-0.5 * (D / sig) ** 2)
        F[idx] = (W @ V) / np.maximum(W.sum(1), 1e-12)
    F = F.reshape(GRID, GRID)
    F = np.where(np.isnan(F), np.nanmean(F), F)
    F = gaussian_filter(F, 1.0)
    F = zoom(F, N / GRID, order=3)
    Dmin = zoom(Dmin.reshape(GRID, GRID), N / GRID, order=1)
    return np.clip(F, 0.0, KMAX), Dmin

def render_filled(F, polys=None, Dmin=None):
    rgb = val_to_rgb(F)
    if Dmin is not None:
        f = np.clip((Dmin - 0.12) / 0.25, 0, 1)[..., None] * 0.7
        rgb = rgb * (1 - f) + f
    out = np.ones((N, N, 3)); out[INSIDE] = rgb[INSIDE]
    out = out * (1 - RING)
    im = Image.fromarray((out * 255 + 0.5).astype(np.uint8))
    if polys is not None:
        dr = ImageDraw.Draw(im)
        for P in polys:
            dr.line([to_px(x, y, N) for x, y in P], fill=(0, 0, 0), width=2)
    return im

# ------------------------------------------------------------------ run
import sys
targets = sys.argv[1:] if len(sys.argv) > 1 else list(tr.keys())
summary = {}
for name in targets:
    S = tr[name]
    polys, pts, I, kinks, st = analyse(S)
    summary[name] = st
    curv, spec = name.split('_')
    label = '%s  %s' % (spec, curv)
    render_lines(polys, kinks, st, label, annotate=True).save(os.path.join(OUT, 'lines', 'kink_lines_%s.png' % name))
    render_lines(polys, kinks, st, label, annotate=False).save(os.path.join(OUT, 'lines', 'kink_lines_plain_%s.png' % name))
    render_lines(polys, kinks, st, label, annotate=True, mark_kinks=True).save(os.path.join(OUT, 'lines', 'kink_lines_marked_%s.png' % name))
    F, Dmin = kernel_field(pts, I)
    render_filled(F).save(os.path.join(OUT, 'filled', 'kink_fill_%s.png' % name))
    render_filled(F, Dmin=Dmin).save(os.path.join(OUT, 'filled_faded', 'kink_fill_faded_%s.png' % name))
    render_filled(F, polys=polys).save(os.path.join(OUT, 'qa', 'qa_kink_fill_%s.png' % name))
    print('%-14s polylines=%2d joints=%3d cusps>=45:%3d (>=30:%3d)  turning=%5.0f°/R  len=%.2fR  frac(I>30)=%.2f' % (
        name, st['n_polylines'], st['n_joints'], st['n_kinks_ge45'], st['n_kinks_ge30'],
        st['turning_per_length'], st['total_length_R'], st['frac_len_I_gt30']))
    if len(targets) == 1:
        print('  joints (x, y, turning deg):')
        for x, y, kk in sorted(kinks.tolist(), key=lambda t: -t[2]):
            print('    (%+.2f, %+.2f)  %5.1f°' % (x, y, kk))

with open(os.path.join(OUT, 'kink_metrics.csv'), 'a', newline='') as f:
    w = csv.writer(f)
    if f.tell() == 0:
        w.writerow(['set', 'n_polylines', 'n_joints', 'n_cusps_ge45', 'n_cusps_ge30', 'total_length_R',
                    'turning_deg_per_R', 'cusps45_per_R', 'frac_len_I_gt30', 'mean_I'])
    for name, st in summary.items():
        w.writerow([name, st['n_polylines'], st['n_joints'], st['n_kinks_ge45'], st['n_kinks_ge30'],
                    '%.3f' % st['total_length_R'], '%.1f' % st['turning_per_length'],
                    '%.2f' % st['kinks45_per_length'], '%.3f' % st['frac_len_I_gt30'], '%.1f' % st['mean_I']])
print('done')
