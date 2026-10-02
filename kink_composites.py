import os, json, csv
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches

OUT = '/home/claude/out_kink'
N = 1000; CEN, RPX = N / 2.0, 0.45 * N
KMAX, SAT, VAL, HUE_MAX = 60.0, 0.80, 1.00, 225.0

def val_to_rgb(v):
    v = np.asarray(v, dtype=float)
    hue = HUE_MAX * (1.0 - np.clip(v, 0, KMAX) / KMAX) / 360.0
    return mcolors.hsv_to_rgb(np.stack([hue, np.full_like(hue, SAT), np.full_like(hue, VAL)], -1))
CMAP = mcolors.ListedColormap(val_to_rgb(np.linspace(0, KMAX, 256)))
NORM = mcolors.Normalize(vmin=0, vmax=KMAX)

metrics = {}
with open(os.path.join(OUT, 'kink_metrics.csv')) as f:
    for row in csv.DictReader(f):
        metrics[row['set']] = row

ROWS_L = [('T1D3', 'D3 (High)'), ('T1D2', 'D2 (Mid)'), ('T1D1', 'D1 (Low)')]
ROWS_R = [('T3D1', 'T3'), ('T2D1', 'T2'), ('T1D1', 'T1')]

def composite(subdir, prefix, fname, title_note):
    fig = plt.figure(figsize=(13.0, 9.8), dpi=150)
    gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 0.35, 1, 1], wspace=0.05, hspace=0.26,
                          left=0.09, right=0.89, top=0.90, bottom=0.03)
    for c0, rows, btitle in [(0, ROWS_L, 'Density series (T1 fixed)'), (3, ROWS_R, 'Diameter series (D1 fixed)')]:
        for ci, curv in enumerate(['Convex', 'Concave']):
            for ri, (spec, rlabel) in enumerate(rows):
                ax = fig.add_subplot(gs[ri, c0 + ci])
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                name = '%s_%s' % (curv, spec)
                p = os.path.join(OUT, subdir, '%s_%s.png' % (prefix, name))
                if os.path.exists(p):
                    ax.imshow(Image.open(p))
                    m = metrics[name]
                    ax.set_xlabel('cusps ≥45°: %s   ·   turning %s°/R\nlength with turning >30°: %d%%' % (
                        m['n_cusps_ge45'], m['turning_deg_per_R'].split('.')[0], round(100 * float(m['frac_len_I_gt30']))),
                                  fontsize=8, color='0.25', labelpad=2)
                else:
                    ax.imshow(np.ones((N, N, 3)))
                    ax.add_patch(matplotlib.patches.Circle((CEN, CEN), RPX, fill=False, ls='--', lw=1.2, ec='0.55'))
                ax.text(0.5, 1.01, spec, transform=ax.transAxes, ha='center', va='bottom', fontsize=9, color='0.35')
                if ri == 0:
                    ax.set_title(curv, fontsize=13, fontweight='bold', pad=16)
                if ci == 0:
                    ax.text(-0.10, 0.5, rlabel, transform=ax.transAxes, ha='right', va='center',
                            fontsize=12, fontweight='bold')
        fig.text(0.26 if c0 == 0 else 0.72, 0.965, btitle, ha='center', va='center', fontsize=14, fontweight='bold')
    cax = fig.add_axes([0.915, 0.20, 0.018, 0.55])
    cb = matplotlib.colorbar.ColorbarBase(cax, cmap=CMAP, norm=NORM, ticks=[0, 15, 30, 45, 60])
    cb.set_label('Local turning angle of contour trace (°)', fontsize=10)
    cb.ax.set_yticklabels(['0', '15', '30', '45', '≥60'])
    cb.ax.tick_params(labelsize=9)
    fig.text(0.924, 0.775, 'sharp cusp', ha='center', fontsize=8, color='0.4')
    fig.text(0.924, 0.175, 'straight', ha='center', fontsize=8, color='0.4')
    pass
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)

composite('lines', 'kink_lines_plain', 'composite_kink_lines.png',
          'colour = direction change between consecutive straight trace elements, spread ±0.1 R along the line')
composite('filled', 'kink_fill', 'composite_kink_filled.png',
          'surface = adaptive-kernel local mean of the line values (hot spots = cusps)')
composite('filled_faded', 'kink_fill_faded', 'composite_kink_filled_faded.png',
          'faded toward white where no trace lies within ~0.12–0.37 R')

# ---------------- summary chart ----------------
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=150, gridspec_kw=dict(wspace=0.35))
keys = [('cusps45_per_R', 'Sharp cusps (≥45°) per unit trace length (1/R)'),
        ('turning_deg_per_R', 'Total turning per unit trace length (°/R)'),
        ('frac_len_I_gt30', 'Length fraction with turning > 30°')]
for ax, (key, ylab) in zip(axes, keys):
    groups = [('density', ['T1D1', 'T1D2', 'T1D3']), ('diameter', ['T1D1', 'T2D1', 'T3D1'])]
    x = 0; ticks = []; labels = []
    for gname, specs in groups:
        for spec in specs:
            for j, (curv, col) in enumerate([('Convex', '#d95f02'), ('Concave', '#1b9e77')]):
                nm = '%s_%s' % (curv, spec)
                if nm in metrics:
                    ax.bar(x + j * 0.38, float(metrics[nm][key]), width=0.36, color=col)
            ticks.append(x + 0.19); labels.append(spec); x += 1
        x += 0.6
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylab, fontsize=9)
    ax.axvline(2.8 + 0.19, color='0.7', lw=0.8, ls=':')
    ymax = ax.get_ylim()[1]
    ax.text(1.19, ymax * 0.98, 'density series (T1)', ha='center', va='top', fontsize=8, color='0.4')
    ax.text(4.79, ymax * 0.98, 'diameter series (D1)', ha='center', va='top', fontsize=8, color='0.4')
    ax.spines[['top', 'right']].set_visible(False)
axes[0].legend(handles=[matplotlib.patches.Patch(color='#d95f02', label='Convex'),
                        matplotlib.patches.Patch(color='#1b9e77', label='Concave')],
               fontsize=8, frameon=False, loc='upper left', bbox_to_anchor=(0, 0.9))
fig.suptitle('Kink statistics of CT contour traces (abrupt direction changes)', fontsize=11)
fig.savefig(os.path.join(OUT, 'kink_metrics_chart.png'), bbox_inches='tight')
plt.close(fig)
print('composites done')
