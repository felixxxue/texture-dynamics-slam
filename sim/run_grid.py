"""Run the full factorial grid with render -> run -> evaluate -> delete streaming (disk budget).

python run_grid.py [--scenes 1 2] [--runs 5] [--slam-jobs 3] [--render-jobs 2] [--keep-frames]
Results: results/runs.csv (one row per run), results/seq_stats.csv (texture metrics per sequence),
figs/frames/ (a few example frames). Resumable: completed run ids are skipped.
"""
import os, sys, time, json, shutil, random, argparse, subprocess
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import evaluate_run

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(WS, 'bin', 'run_seq.exe'); VOC = os.path.join(WS, 'vocab', 'ORBvoc.bin')
DATA = os.path.join(WS, 'data'); RUNS = os.path.join(WS, 'runs'); RES = os.path.join(WS, 'results'); FIG = os.path.join(WS, 'figs', 'frames')
PY = sys.executable
FILTERS = ['none', 'flow', 'geom']; SENSORS = ['rgbd', 'stereo']
EXAMPLE_FRAMES = (60, 180, 300)

def seq_name(s, l, d): return f's{s}_L{l}_D{d}'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', type=int, nargs='+', default=[1, 2]); ap.add_argument('--tex', type=int, nargs='+', default=[0, 1, 2, 3])
    ap.add_argument('--dyn', type=int, nargs='+', default=[0, 1, 2]); ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--slam-jobs', type=int, default=3); ap.add_argument('--render-jobs', type=int, default=2)
    ap.add_argument('--keep-frames', action='store_true'); ap.add_argument('--filters', nargs='+', default=FILTERS)
    ap.add_argument('--extra', nargs='*', default=[], help='key=value passed to run_seq'); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    for d in (DATA, RUNS, RES, FIG): os.makedirs(d, exist_ok=True)
    res_csv = os.path.join(RES, f'runs{a.tag}.csv')
    done = set(pd.read_csv(res_csv)['run_id']) if os.path.exists(res_csv) else set()
    seqs = [(s, l, d) for s in a.scenes for l in a.tex for d in a.dyn]
    todo_render = [q for q in seqs]
    rendering = {}; ready = []; running = {}; per_seq_left = {}
    rng = random.Random(12345)
    log = open(os.path.join(RES, f'grid_log{a.tag}.txt'), 'a')
    def say(m):
        line = time.strftime('%H:%M:%S ') + m; print(line, flush=True); log.write(line + '\n'); log.flush()
    def jobs_for(q):
        js = [(q, sen, f, k) for sen in SENSORS for f in a.filters for k in range(a.runs)]
        js = [j for j in js if run_id(*j) not in done]; rng.shuffle(js); return js
    def run_id(q, sen, f, k): return f'{seq_name(*q)}_{sen}_{f}_r{k}{a.tag}'
    job_queue = []
    t0 = time.time()
    while todo_render or rendering or ready or running or job_queue:
        # start renders
        while todo_render and len(rendering) < a.render_jobs:
            q = todo_render.pop(0); sd = os.path.join(DATA, seq_name(*q))
            js = jobs_for(q)
            if not js: say(f'skip {seq_name(*q)} (all runs done)'); continue
            if os.path.exists(os.path.join(sd, 'meta.json')):
                ready.append((q, js)); continue
            lf = open(sd + '_render.log', 'w')
            p = subprocess.Popen([PY, os.path.join(WS, 'sim', 'render.py'), '--scene', str(q[0]), '--tex', str(q[1]), '--dyn', str(q[2]), '--out', sd], stdout=lf, stderr=subprocess.STDOUT)
            rendering[q] = (p, js, lf); say(f'render start {seq_name(*q)}')
        for q in list(rendering):
            p, js, lf = rendering[q]
            if p.poll() is not None:
                lf.close(); del rendering[q]
                if p.returncode != 0: say(f'RENDER FAILED {seq_name(*q)}'); continue
                say(f'render done {seq_name(*q)}'); ready.append((q, js))
        while ready:
            q, js = ready.pop(0); per_seq_left[q] = len(js); job_queue += js
        # start SLAM runs
        while job_queue and len(running) < a.slam_jobs:
            j = job_queue.pop(0); q, sen, f, k = j; sd = os.path.join(DATA, seq_name(*q))
            rid = run_id(*j); rd = os.path.join(RUNS, rid); os.makedirs(rd, exist_ok=True)
            lf = open(os.path.join(rd, 'stdout.txt'), 'w')
            p = subprocess.Popen([EXE, VOC, os.path.join(sd, 'settings.yaml'), sd, sen, f, rd] + a.extra, stdout=lf, stderr=subprocess.STDOUT)
            running[rid] = (p, j, lf, time.time())
        for rid in list(running):
            p, j, lf, ts = running[rid]
            if p.poll() is None:
                if time.time() - ts > 600: p.kill()
                continue
            lf.close(); del running[rid]
            q, sen, f, k = j; sd = os.path.join(DATA, seq_name(*q)); rd = os.path.join(RUNS, rid)
            row = dict(run_id=rid, scene=q[0], tex=q[1], dyn=q[2], sensor=sen, filter=f, rep=k, exit=p.returncode, wall_s=round(time.time() - ts, 1))
            try:
                row.update(evaluate_run(rd, sd))
            except Exception as ex:
                row['eval_error'] = str(ex)[:200]
            pd.DataFrame([row]).to_csv(res_csv, mode='a', header=not os.path.exists(res_csv), index=False)
            done.add(rid)
            say(f"{rid} exit={p.returncode} C={row.get('C', float('nan')):.3f} ATE={row.get('ate', float('nan')):.4f} P={row.get('P', float('nan')):.2f} FRR={row.get('FRR', float('nan')):.3f}")
            per_seq_left[q] -= 1
            if per_seq_left[q] == 0:
                # sequence finished: keep statistics and example frames, delete images
                st = pd.read_csv(os.path.join(sd, 'frame_stats.csv')); meta = json.load(open(os.path.join(sd, 'meta.json')))
                srow = dict(seq=seq_name(*q), scene=q[0], tex=q[1], dyn=q[2], beta=meta['beta'], n_fast=st['n_fast'].mean(),
                            n_fast_med=st['n_fast'].median(), h_grad=st['h_grad'].mean(), rho=st['rho'].mean(), render_s=meta['render_s'])
                sp = os.path.join(RES, 'seq_stats.csv')
                pd.DataFrame([srow]).to_csv(sp, mode='a', header=not os.path.exists(sp), index=False)
                for fi in EXAMPLE_FRAMES:
                    for sub in ('left', 'mask', 'depth'):
                        src = os.path.join(sd, sub, f'{fi:06d}.png')
                        if os.path.exists(src): shutil.copy(src, os.path.join(FIG, f'{seq_name(*q)}_{sub}_{fi:06d}.png'))
                for keep in ('groundtruth.txt', 'frame_stats.csv', 'meta.json', 'times.txt', 'settings.yaml'):
                    kd = os.path.join(RES, 'seq_meta', seq_name(*q)); os.makedirs(kd, exist_ok=True)
                    shutil.copy(os.path.join(sd, keep), kd)
                if not a.keep_frames: shutil.rmtree(sd, ignore_errors=True)
                say(f'sequence {seq_name(*q)} complete ({(time.time() - t0) / 60:.1f} min elapsed)')
        time.sleep(1.0)
    say('grid finished')

if __name__ == '__main__':
    main()
