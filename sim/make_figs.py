"""Figures for the paper (run in a kernel where figure-style helpers are loaded, after analyze.main()).
exec(open('sim/make_figs.py').read()) with variables: runs, seq, main_t, dl, pc (from analyze.main()).
Outputs: figs/fig2_frames.pdf/.png, figs/fig3_heatmaps.pdf/.png, figs/fig4_mechanism.pdf/.png
"""
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt, os
from matplotlib.colors import TwoSlopeNorm
os.makedirs('figs', exist_ok=True)
apply_figure_style(sizes=(8, 7, 6))
COL = {'flow': '#D55E00', 'geom': '#0072B2', 'none': '#7F7F7F'}
SEN_COL = {'rgbd': '#0072B2', 'stereo': '#E69F00'}
LBL = {'flow': 'FLOW', 'geom': 'GEOM', 'none': 'NONE', 'rgbd': 'RGB-D', 'stereo': 'Stereo'}
IEEE_COL = 3.5  # inches, single column

# ---------------------------------------------------------------- Fig. 2: frames at L0..L3 (scene 1, D2, same pose)
fi = 180
fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.55))
for l, ax in enumerate(axs):
    p = f'figs/frames/s1_L{l}_D2_left_{fi:06d}.png'
    im = plt.imread(p) if os.path.exists(p) else np.zeros((480, 640))
    ax.imshow(im, cmap='gray', vmin=0, vmax=1, interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    s1 = seq[(seq.tex == l)]
    ax.set_title(f'L{l} ($\\beta$={s1.beta.median():.1f}, $\\bar N_F$={s1.n_fast.median():.0f})', fontsize=7)
fig.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.01, wspace=0.03)
fig.savefig('figs/fig2_frames.pdf'); fig.savefig('figs/fig2_frames.png', dpi=300); plt.close(fig)

# ---------------------------------------------------------------- Fig. 3: Delta ATE heat maps
vals = dl.dATE.values * 100.0  # cm
vmax = np.nanpercentile(np.abs(vals), 95) if np.isfinite(vals).any() else 1.0
vmax = max(np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0, 2.0)
norm = mpl.colors.SymLogNorm(linthresh=1.0, linscale=0.6, vmin=-vmax, vmax=vmax, base=10)
fig, axs = plt.subplots(2, 2, figsize=(IEEE_COL, 3.3), sharex=True, sharey=True)
for i, sen in enumerate(['rgbd', 'stereo']):
    for j, f in enumerate(['flow', 'geom']):
        ax = axs[i, j]
        M = np.full((4, 3), np.nan); S = np.zeros((4, 3), bool); FL = np.zeros((4, 3), int)
        for _, r in dl[(dl.sensor == sen) & (dl['filter'] == f)].iterrows():
            M[int(r.tex), int(r.dyn)] = r.dATE * 100; S[int(r.tex), int(r.dyn)] = bool(r.sig) if r.sig == r.sig else False
            FL[int(r.tex), int(r.dyn)] = int(r.n_fail_f)
        im = ax.imshow(np.ma.masked_invalid(M), cmap='RdBu_r', norm=norm, aspect='auto')
        for (a, b), v in np.ndenumerate(M):
            if np.isnan(v):
                ax.add_patch(mpl.patches.Rectangle((b - .5, a - .5), 1, 1, fill=False, hatch='///', ec='0.6', lw=0))
                ax.text(b, a, 'n/a', ha='center', va='center', fontsize=6, color='0.3')
            else:
                txt = f'{v:+.1f}' if abs(v) < 10 else f'{v:+.0f}'
                if S[a, b]: txt += '*'
                c = 'white' if abs(v) > 0.25 * vmax else 'black'
                ax.text(b, a, txt, ha='center', va='center', fontsize=6, color=c)
            if FL[a, b] > 0:
                ax.text(b + 0.45, a - 0.45, f'{FL[a, b]}F', ha='right', va='top', fontsize=5, color='0.15')
        ax.set_xticks(range(3)); ax.set_xticklabels(['D0', 'D1', 'D2']); ax.set_yticks(range(4)); ax.set_yticklabels(['L0', 'L1', 'L2', 'L3'])
        ax.set_title(f'{LBL[f]}, {LBL[sen]}', fontsize=8, loc='left')
        ax.tick_params(length=0)
cax = fig.add_axes([0.87, 0.2, 0.025, 0.6])
cb = fig.colorbar(im, cax=cax, ticks=[-50, -10, -1, 0, 1, 10, 50]); cb.ax.set_yticklabels(['-50', '-10', '-1', '0', '1', '10', '50']); cb.set_label('$\\Delta$ATE [cm]  (<0: filter helps)', fontsize=7); cb.ax.tick_params(labelsize=6)
fig.subplots_adjust(left=0.1, right=0.84, top=0.93, bottom=0.08, wspace=0.08, hspace=0.25)
fig.savefig('figs/fig3_heatmaps.pdf'); fig.savefig('figs/fig3_heatmaps.png', dpi=300); plt.close(fig)

# ---------------------------------------------------------------- Fig. 4: (a) FRR vs texture, (b) dATE vs N_FAST by sensor
fig, axs = plt.subplots(1, 2, figsize=(IEEE_COL, 1.75))
ax = axs[0]
for f in ['flow', 'geom']:
    for d, ls, mk in [(0, '--', 'o'), (2, '-', 's')]:
        z = pc[(pc.sensor == 'rgbd') & (pc['filter'] == f) & (pc.dyn == d)]
        med = main_t[(main_t.sensor == 'rgbd') & (main_t['filter'] == f) & (main_t.dyn == d)].set_index('tex').FRR * 100
        ax.plot(med.index, med.values, ls=ls, marker=mk, ms=3, color=COL[f], lw=1.2, label=f'{LBL[f]} D{d}')
        ax.scatter(z.tex + (0.06 if f == 'geom' else -0.06), z.FRR * 100, s=5, color=COL[f], alpha=0.35, lw=0)
ax.set_xticks(range(4)); ax.set_xticklabels(['L0', 'L1', 'L2', 'L3']); ax.set_ylabel('FRR [%]'); ax.set_xlabel('texture level')
ax.margins(0.06); ax.set_ylim(-1, 30); ax.legend(frameon=False, fontsize=5.5, handlelength=2.0, loc='upper left', ncol=2, columnspacing=0.8)
ax.set_title('RGB-D', fontsize=7, loc='left')
panel_letter(ax, 'a', case='lower')
ax = axs[1]
for sen in ['rgbd', 'stereo']:
    for f, mk in [('flow', '^'), ('geom', 'o')]:
        z = pc[(pc.sensor == sen) & (pc['filter'] == f) & (pc.dyn > 0)]
        ax.scatter(z.n_fast, z.dATE * 100, s=10, marker=mk, facecolor=SEN_COL[sen] if f == 'geom' else 'none', edgecolor=SEN_COL[sen], lw=0.8,
                   label=f'{LBL[f]}, {LBL[sen]}')
ax.axhline(0, color='0.5', lw=0.6, zorder=0)
ax.set_xscale('log'); ax.set_xticks([20, 50, 100, 200, 400]); ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter()); ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter()); ax.set_xlabel('$\\bar N_F$ (FAST corners / frame)'); ax.set_ylabel('$\\Delta$ATE [cm]')
ax.set_yscale('symlog', linthresh=1.0); ax.set_yticks([-100, -10, -1, 0, 1, 10, 100]); ax.set_yticklabels(['-100', '-10', '-1', '0', '1', '10', '100'])
ax.legend(frameon=False, fontsize=5.5, handletextpad=0.2, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=0.6)
panel_letter(ax, 'b', case='lower')
axs[0].set_title('RGB-D', fontsize=7, loc='left')
fig.subplots_adjust(left=0.13, right=0.98, top=0.80, bottom=0.24, wspace=0.6)
fig.savefig('figs/fig4_mechanism.pdf'); fig.savefig('figs/fig4_mechanism.png', dpi=300); plt.close(fig)
