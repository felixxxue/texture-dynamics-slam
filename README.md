# Dynamic-point filtering under controlled texture (ORB-SLAM2)

Code, results and run logs for the paper ([PDF](paper/main.pdf)) *"Does Dynamic-Point Filtering Help When Texture Is Scarce? A Controlled Study of ORB-SLAM2 Front-Ends in Synthetic Indoor Scenes"* (Zekui Xue, University of Bath; extended from the author's 2022 MSc dissertation).

The study renders 24 synthetic indoor sequences in which surface texture (L0–L3) and scene dynamics (D0–D2, 0/1/3 moving agents) are varied factorially along identical camera trajectories. It then runs ORB-SLAM2 (stereo and RGB-D) without filtering (NONE), with an optical-flow and epipolar-residual filter (FLOW), and with a multi-view depth-consistency filter (GEOM): 2 scenes × 4 × 3 × 3 filters × 2 sensors × 5 runs = 720 runs.

## Contents

| Path | Description |
|---|---|
| `sim/render.py` | CPU ray-caster (Open3D `RaycastingScene`) that writes stereo, depth, dynamic masks, ground truth, texture metrics and ORB-SLAM2 settings |
| `sim/run_grid.py` | Runs the full grid (render → run → evaluate → delete images) |
| `sim/evaluate.py` | ATE (SE(3) alignment) and RPE (1 m) with evo, completeness, per-keypoint P/R/FRR |
| `sim/analyze.py`, `sim/make_figs.py` | Tables, statistics (`results/stats.json`) and figures |
| `orbslam2/orbslam2_study.patch` | Patch against ORB-SLAM2 commit `f2e6f51cdc8d067655d90a78c06261378e07e8f3`: Windows/OpenCV-4 portability, headless build, binary vocabulary, FLOW/GEOM filters and logging |
| `orbslam2/DynFilter.cc`, `orbslam2/tools/` | The filter source and the sequence driver (`run_seq`) / vocabulary converter |
| `orbslam2/build_orb.py` | MSVC build script used for the paper (adapt the paths) |
| `results/` | One row per run (`runs.csv`), aggregated tables, per-sequence metadata (ground truth, texture statistics) |
| `data/run_logs.zip` | Per-run frame logs (`frames.csv`), estimated trajectories (`traj.txt`) and example frames |
| `figures/` | Figures 2–4 of the paper |

Rendered images are not stored. `sim/render.py` regenerates every sequence deterministically.

## Paper ↔ files

Every table, figure and number in the paper is either stored in a committed file under `results/` or recomputed from those files by `sim/analyze.py` / `sim/make_figs.py`. From the repository root, `python sim/make_figs.py` (which calls `sim/analyze.py`) regenerates all tables, `results/stats.json` and Figures 2–4 from the committed `results/runs.csv`, `results/seq_stats.csv` and `data/run_logs.zip` in about 10 s; no SLAM re-run is needed. Re-running it reproduces `stats.json`, `table_main.csv`, `table_texture.csv`, `delta_ate_cells.csv` and `latex_tables.tex` byte for byte. Figure styling may differ slightly from the committed PDFs.

| Paper | Produced by | Data |
|---|---|---|
| Table I (positioning) | literature, no data | — |
| Table II (texture and dynamics per level: β, N̄_F, H_grad, keypoints, dynamic-pixel ratio) | `sim/analyze.py` | `results/table_texture.csv`, `results/seq_stats.csv`, `results/beta_calibration.csv` |
| Table III (main grid: ATE / completeness C / failures) | `sim/analyze.py` | `results/table_main.csv` (LaTeX: `results/latex_tables.tex`) |
| Table IV (P, R, FRR, static inliers at D2) | `sim/analyze.py` | `results/latex_tables.tex`, `results/stats.json` |
| Fig. 1 (pipeline schematic) | drawn in LaTeX | — |
| Fig. 2 (frames L0–L3) | `sim/make_figs.py` (re-renders frame 180 with `sim/render.py`) | `figures/fig2_frames.pdf` |
| Fig. 3 (ΔATE heat maps, Holm-corrected tests) | `sim/make_figs.py` | `results/delta_ate_cells.csv`, `results/per_scene_cells.csv`, `results/stats.json` |
| Fig. 4 (FRR vs texture, mechanism) | `sim/make_figs.py` | `results/runs.csv`, `results/stats.json` |
| Sec. IV numbers (Spearman ρ, Holm p-values, median FRR, completeness changes, stereo-depth fraction, run times) | `sim/analyze.py` | `results/stats.json`, `results/runs.csv` |
| Per-run logs, trajectories | `sim/run_grid.py` | `data/run_logs.zip`, `results/grid_log.txt` |
| Paper PDF and LaTeX source | — | `paper/main.pdf`, `paper/main.tex`, `paper/main.bbl`, `paper/refs.bib` |

Pinned Python dependencies: `requirements.txt` (`pip install -r requirements.txt`).

## Reproducing

1. Get ORB-SLAM2 at commit `f2e6f51`, apply `orbslam2/orbslam2_study.patch`, copy `DynFilter.cc` to `src/` and the `tools/` sources, and build. The paper used MSVC 19.44 + OpenCV 4.10 on Windows; on Linux the original CMake build plus the new source files should also work, but this has not been tested.
2. Convert the vocabulary: `voc_convert ORBvoc.txt ORBvoc.bin`.
3. Python 3.12 and `pip install -r requirements.txt`.
4. `python sim/run_grid.py --slam-jobs 3 --render-jobs 2`, then `python sim/analyze.py`. A quick check: `python sim/run_grid.py --scenes 1 --tex 0 2 --runs 2 --tag _check`.

Filter parameters, the renderer settings and the protocol are described in Sec. III of the paper and fixed in the code.

## Licence

- Code: **GPL-3.0**, see `LICENSE` (the ORB-SLAM2 patch is a derivative of GPLv3 ORB-SLAM2).
- Data and results (`results/`, `data/`, `figures/`): **CC BY 4.0**, see `DATA_LICENSE.md`. All textures and scenes are procedural; no third-party assets are used.

## Citation

See `CITATION.cff`. The arXiv identifier will be added after publication.
