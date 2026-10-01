"""Aggregate grid results: tables (CSV + LaTeX rows), statistics (JSON). Figures are made in make_figs.py.
Run from the workspace root: python sim/analyze.py
"""
import os, json, itertools, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import stats

RES = 'results'; C_MIN = 0.8; MIN_OK = 3
FILTERS = ['none', 'flow', 'geom']; SENSORS = ['rgbd', 'stereo']; TEX = [0, 1, 2, 3]; DYN = [0, 1, 2]

def holm(p):
    p = np.asarray(p, float); m = np.sum(~np.isnan(p)); order = np.argsort(np.where(np.isnan(p), np.inf, p))
    adj = np.full_like(p, np.nan); run = 0.0
    for r, i in enumerate(order):
        if np.isnan(p[i]): continue
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj

def load():
    runs = pd.read_csv(os.path.join(RES, 'runs.csv')).drop_duplicates('run_id', keep='last')
    runs['failed'] = (runs['C'] < C_MIN) | runs['ate'].isna() | (runs['exit'] != 0)
    # filter decisions only exist where the filter actually ran (GEOM needs an initialised map)
    nr = (runs['filter'] != 'none') & (runs['ran_frac'] < 0.05)
    runs.loc[nr, ['P', 'R', 'FRR']] = np.nan
    seq = pd.read_csv(os.path.join(RES, 'seq_stats.csv')).drop_duplicates('seq', keep='last')
    return runs, seq

def med_ok(x):
    x = x.dropna()
    return (float(np.median(x)) if len(x) >= MIN_OK else np.nan)

def main():
    runs, seq = load()
    out = {}
    # ---------------- Table II: texture / dynamics statistics per level
    t2 = []
    for l in TEX:
        s = seq[seq.tex == l]
        r = runs[(runs.tex == l) & (runs['filter'] == 'none')]
        t2.append(dict(level=f'L{l}', beta=float(s.beta.median()), n_fast=float(s.n_fast.median()), h_grad=float(s.h_grad.median()),
                       kp_stereo=float(r[r.sensor == 'stereo'].N_mean.median()), kp_rgbd=float(r[r.sensor == 'rgbd'].N_mean.median()),
                       stereo_depth_frac=float(r[r.sensor == 'stereo'].depth_frac.median()),
                       n_fast_s1=float(s[s.scene == 1].n_fast.median()), n_fast_s2=float(s[s.scene == 2].n_fast.median())))
    t2 = pd.DataFrame(t2); t2.to_csv(os.path.join(RES, 'table_texture.csv'), index=False)
    rho = {f'D{d}': float(seq[seq.dyn == d].rho.median()) for d in DYN}
    out['rho'] = rho
    # ---------------- Table IV: main grid (pooled over scenes and runs)
    rows = []
    for sen, l, d, f in itertools.product(SENSORS, TEX, DYN, FILTERS):
        r = runs[(runs.sensor == sen) & (runs.tex == l) & (runs.dyn == d) & (runs['filter'] == f)]
        ok = r[~r.failed]
        rows.append(dict(sensor=sen, tex=l, dyn=d, filter=f, n=len(r), n_fail=int(r.failed.sum()),
                         ate=med_ok(ok.ate), ate_q1=float(ok.ate.quantile(.25)) if len(ok) else np.nan, ate_q3=float(ok.ate.quantile(.75)) if len(ok) else np.nan,
                         rpe=med_ok(ok.rpe), C=float(r.C.median()) if len(r) else np.nan, n_lost=float(r.n_lost.median()) if len(r) else np.nan,
                         ns=float(r.ns_mean.median()) if len(r) else np.nan,
                         P=float(r.P.median()) if len(r) else np.nan, R=float(r.R.median()) if len(r) else np.nan, FRR=float(r.FRR.median()) if len(r) else np.nan))
    main_t = pd.DataFrame(rows); main_t.to_csv(os.path.join(RES, 'table_main.csv'), index=False)
    # ---------------- Delta ATE per cell + within-cell rank-sum tests (Holm over all cells)
    dl = []
    for sen, l, d, f in itertools.product(SENSORS, TEX, DYN, ['flow', 'geom']):
        base = runs[(runs.sensor == sen) & (runs.tex == l) & (runs.dyn == d) & (runs['filter'] == 'none') & ~runs.failed].ate
        x = runs[(runs.sensor == sen) & (runs.tex == l) & (runs.dyn == d) & (runs['filter'] == f) & ~runs.failed].ate
        dA = (med_ok(x) - med_ok(base)) if (len(x) >= MIN_OK and len(base) >= MIN_OK) else np.nan
        p = stats.mannwhitneyu(x, base, alternative='two-sided').pvalue if (len(x) >= MIN_OK and len(base) >= MIN_OK) else np.nan
        nf_f = int(runs[(runs.sensor == sen) & (runs.tex == l) & (runs.dyn == d) & (runs['filter'] == f)].failed.sum())
        nf_b = int(runs[(runs.sensor == sen) & (runs.tex == l) & (runs.dyn == d) & (runs['filter'] == 'none')].failed.sum())
        dl.append(dict(sensor=sen, tex=l, dyn=d, filter=f, dATE=dA, p=p, n_fail_f=nf_f, n_fail_none=nf_b))
    dl = pd.DataFrame(dl); dl['p_holm'] = holm(dl.p.values); dl['sig'] = dl.p_holm < 0.05
    dl.to_csv(os.path.join(RES, 'delta_ate_cells.csv'), index=False)
    # ---------------- per-scene cells (for correlations)
    nf = seq.set_index(['scene', 'tex', 'dyn']).n_fast
    pc = []
    for sc, sen, l, d in itertools.product([1, 2], SENSORS, TEX, DYN):
        r = runs[(runs.scene == sc) & (runs.sensor == sen) & (runs.tex == l) & (runs.dyn == d)]
        if not len(r): continue
        base = r[(r['filter'] == 'none') & ~r.failed]
        for f in ['flow', 'geom']:
            x = r[(r['filter'] == f) & ~r.failed]; rf = r[r['filter'] == f]
            pc.append(dict(scene=sc, sensor=sen, tex=l, dyn=d, filter=f, n_fast=float(nf.get((sc, l, d), np.nan)),
                           dATE=(med_ok(x.ate) - med_ok(base.ate)) if len(x) >= MIN_OK and len(base) >= MIN_OK else np.nan,
                           dC=float(rf.C.median() - r[r['filter'] == 'none'].C.median()),
                           FRR=float(rf.FRR.median()), R=float(rf.R.median()), P=float(rf.P.median()),
                           dns=float(rf.ns_mean.median() - r[r['filter'] == 'none'].ns_mean.median()),
                           fail_f=int(rf.failed.sum()), fail_none=int(r[r['filter'] == 'none'].failed.sum())))
    pc = pd.DataFrame(pc); pc.to_csv(os.path.join(RES, 'per_scene_cells.csv'), index=False)
    def sp(a, b):
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() < 5: return dict(rho=np.nan, p=np.nan, n=int(m.sum()))
        r = stats.spearmanr(a[m], b[m]); return dict(rho=float(r.statistic), p=float(r.pvalue), n=int(m.sum()))
    # ---------------- H1: benefit (-dATE) vs texture
    h1 = {}
    for sen, f in itertools.product(SENSORS, ['flow', 'geom']):
        q = pc[(pc.sensor == sen) & (pc['filter'] == f) & (pc.dyn > 0)]
        h1[f'{f}_{sen}'] = dict(spearman_benefit_vs_nfast=sp(-q.dATE.values, q.n_fast.values),
                                dATE_L0=float(dl[(dl.sensor == sen) & (dl['filter'] == f) & (dl.tex == 0) & (dl.dyn > 0)].dATE.median()),
                                dATE_L3=float(dl[(dl.sensor == sen) & (dl['filter'] == f) & (dl.tex == 3) & (dl.dyn > 0)].dATE.median()),
                                dC_L0=float(q[q.tex == 0].dC.median()), dC_L3=float(q[q.tex == 3].dC.median()))
        # paired: benefit in rich (L0,L1) vs poor (L2,L3) texture, paired by scene x dyn x (L0<->L2, L1<->L3)
        rich = q[q.tex.isin([0, 1])].assign(k=lambda z: z.tex); poor = q[q.tex.isin([2, 3])].assign(k=lambda z: z.tex - 2)
        mg = rich.merge(poor, on=['scene', 'dyn', 'k'], suffixes=('_r', '_p'))
        a = mg.dATE_r.values; b = mg.dATE_p.values
        m = ~(np.isnan(a) | np.isnan(b))
        h1[f'{f}_{sen}']['signedrank_rich_vs_poor_p'] = float(stats.wilcoxon(a[m], b[m]).pvalue) if m.sum() >= 5 else np.nan
        h1[f'{f}_{sen}']['n_pairs'] = int(m.sum())
    out['H1'] = h1
    # ---------------- H2: FRR GEOM vs FLOW, gap vs texture, mechanism correlations
    h2 = {}
    for sen in SENSORS:
        mg = pc[(pc.sensor == sen) & (pc['filter'] == 'flow')].merge(pc[(pc.sensor == sen) & (pc['filter'] == 'geom')], on=['scene', 'tex', 'dyn'], suffixes=('_f', '_g'))
        mg = mg.dropna(subset=['FRR_f', 'FRR_g'])
        gap = mg.FRR_f.values - mg.FRR_g.values
        h2[sen] = dict(FRR_flow_med=float(np.nanmedian(mg.FRR_f)), FRR_geom_med=float(np.nanmedian(mg.FRR_g)),
                       signedrank_geom_lt_flow_p=float(stats.wilcoxon(mg.FRR_g.values, mg.FRR_f.values, alternative='less').pvalue) if len(mg) >= 5 else np.nan,
                       frac_cells_geom_lower=float(np.mean(mg.FRR_g.values < mg.FRR_f.values)),
                       spearman_gap_vs_nfast=sp(gap, mg.n_fast_f.values))
        q = pc[(pc.sensor == sen) & (pc.dyn > 0)]
        h2[sen]['spearman_dATE_vs_FRR'] = sp(q.dATE.values, q.FRR.values)
        h2[sen]['spearman_dATE_vs_R'] = sp(q.dATE.values, q.R.values)
        h2[sen]['spearman_dATE_vs_dns'] = sp(q.dATE.values, q.dns.values)
        for f in ['flow', 'geom']:
            for l in TEX:
                z = pc[(pc.sensor == sen) & (pc['filter'] == f) & (pc.tex == l)]
                h2[sen][f'{f}_L{l}'] = dict(FRR=float(z.FRR.median()), R=float(z[z.dyn > 0].R.median()), P=float(z[z.dyn > 0].P.median()))
    out['H2'] = h2
    # ---------------- H3: GEOM advantage over FLOW by sensor
    h3 = {}
    for sen in SENSORS:
        q = pc[(pc.sensor == sen) & (pc.dyn > 0)]
        mg = q[q['filter'] == 'flow'].merge(q[q['filter'] == 'geom'], on=['scene', 'tex', 'dyn'], suffixes=('_f', '_g'))
        adv = mg.dATE_f.values - mg.dATE_g.values   # > 0: GEOM better than FLOW
        h3[sen] = dict(adv_by_tex={f'L{l}': (float(np.nanmedian(adv[mg.tex.values == l])) if np.any((mg.tex.values == l) & ~np.isnan(adv)) else np.nan) for l in TEX},
                       spearman_adv_vs_nfast=sp(adv, mg.n_fast_f.values))
    st = runs[(runs.sensor == 'stereo') & (runs['filter'] == 'none')]
    h3['stereo_depth_frac_by_tex'] = {f'L{l}': float(st[st.tex == l].depth_frac.median()) for l in TEX}
    out['H3'] = h3
    out['n_runs'] = int(len(runs)); out['n_failed'] = int(runs.failed.sum()); out['C_min'] = C_MIN
    out['fail_by'] = {f'{k[0]}_L{k[1]}_{k[2]}': int(v) for k, v in runs.groupby(['sensor', 'tex', 'filter']).failed.sum().items()}
    json.dump(out, open(os.path.join(RES, 'stats.json'), 'w'), indent=1, default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else str(x))
    return runs, seq, main_t, dl, pc, out

if __name__ == '__main__':
    main()
