# Dynamic-point filtering under controlled texture (ORB-SLAM2)

Code, results and run logs for the paper *"Does Dynamic-Point Filtering Help When Texture Is Scarce? A Controlled Study of ORB-SLAM2 Front-Ends in Synthetic Indoor Scenes"* (Zekui Xue, University of Bath; extended from the author's 2022 MSc dissertation).

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

## Reproducing

1. Get ORB-SLAM2 at commit `f2e6f51`, apply `orbslam2/orbslam2_study.patch`, copy `DynFilter.cc` to `src/` and the `tools/` sources, and build. The paper used MSVC 19.44 + OpenCV 4.10 on Windows; on Linux the original CMake build plus the new source files should also work, but this has not been tested.
2. Convert the vocabulary: `voc_convert ORBvoc.txt ORBvoc.bin`.
3. Python 3.12 with `numpy scipy pandas matplotlib opencv-python-headless open3d evo`.
4. `python sim/run_grid.py --slam-jobs 3 --render-jobs 2`, then `python sim/analyze.py`. A quick check: `python sim/run_grid.py --scenes 1 --tex 0 2 --runs 2 --tag _check`.

Filter parameters, the renderer settings and the protocol are described in Sec. III of the paper and fixed in the code.

## Licence

- Code: **GPL-3.0**, see `LICENSE` (the ORB-SLAM2 patch is a derivative of GPLv3 ORB-SLAM2).
- Data and results (`results/`, `data/`, `figures/`): **CC BY 4.0**, see `DATA_LICENSE.md`. All textures and scenes are procedural; no third-party assets are used.

## Citation

See `CITATION.cff`. The arXiv identifier will be added after publication.
