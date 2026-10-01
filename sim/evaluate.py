"""Evaluate one ORB-SLAM2 run against ground truth.

ATE: translational RMSE after SE(3) Umeyama alignment (no scale) with evo.
RPE: translational RMSE over a fixed offset of 1 m (evo, delta_unit = m).
Completeness C = tracked frames in the final map / total frames. Filter decisions P/R/FRR from frames.csv.
"""
import os, sys, json
import numpy as np
import pandas as pd
from evo.tools import file_interface
from evo.core import sync, metrics
from evo.core.units import Unit

RPE_DELTA_M = 1.0
MIN_POSES = 30

def evaluate_run(run_dir, seq_dir):
    out = {}
    fr = pd.read_csv(os.path.join(run_dir, 'frames.csv'))
    T = len(fr)
    tr_path = os.path.join(run_dir, 'traj.txt')
    n_est = 0
    if os.path.exists(tr_path) and os.path.getsize(tr_path) > 0:
        est = file_interface.read_tum_trajectory_file(tr_path); n_est = est.num_poses
    out['T'] = T; out['n_tracked'] = n_est; out['C'] = n_est / T if T else np.nan
    out['ate'] = np.nan; out['rpe'] = np.nan
    if n_est >= MIN_POSES:
        ref = file_interface.read_tum_trajectory_file(os.path.join(seq_dir, 'groundtruth.txt'))
        ref_s, est_s = sync.associate_trajectories(ref, est, max_diff=0.005)
        est_a = __import__('copy').deepcopy(est_s)
        est_a.align(ref_s, correct_scale=False)
        ape = metrics.APE(metrics.PoseRelation.translation_part); ape.process_data((ref_s, est_a))
        out['ate'] = float(ape.get_statistic(metrics.StatisticsType.rmse))
        try:
            rpe = metrics.RPE(metrics.PoseRelation.translation_part, delta=RPE_DELTA_M, delta_unit=Unit.meters,
                              rel_delta_tol=0.2, all_pairs=True)
            rpe.process_data((ref_s, est_s)); out['rpe'] = float(rpe.get_statistic(metrics.StatisticsType.rmse))
        except Exception:
            out['rpe'] = np.nan
    st = fr['state'].values
    ok = st == 2
    out['n_lost'] = int(((st[1:] == 3) & (st[:-1] == 2)).sum())
    info = {}
    ip = os.path.join(run_dir, 'run_info.txt')
    if os.path.exists(ip):
        for l in open(ip):
            k, _, v = l.strip().partition('='); info[k] = v
    out['resets'] = int(info.get('resets', 0)); out['track_ms'] = float(info.get('mean_track_ms', 'nan'))
    out['ns_mean'] = float(fr.loc[ok, 'nInlStatic'].mean()) if ok.any() else 0.0
    out['ninl_mean'] = float(fr.loc[ok, 'nInl'].mean()) if ok.any() else 0.0
    out['N_mean'] = float(fr['N'].mean()); out['depth_frac'] = float((fr['nDepth'] / fr['N'].clip(lower=1)).mean())
    TP, FP, FN, TN = (int(fr[c].sum()) for c in ('TP', 'FP', 'FN', 'TN'))
    out.update(TP=TP, FP=FP, FN=FN, TN=TN, ran_frac=float(fr['ran'].mean()))
    out['P'] = TP / (TP + FP) if TP + FP else np.nan
    out['R'] = TP / (TP + FN) if TP + FN else np.nan
    out['FRR'] = FP / (FP + TN) if FP + TN else np.nan
    return out

if __name__ == '__main__':
    print(json.dumps(evaluate_run(sys.argv[1], sys.argv[2]), indent=1))
