# Dynamic-Point Filtering under Controlled Texture — ORB-SLAM2

Code, experimental results, and run logs for the research study:

**Does Dynamic-Point Filtering Help When Texture Is Scarce? A Controlled Study of ORB-SLAM2 Front-Ends in Synthetic Indoor Scenes**

**Author:** Zekui Xue
**Institution:** University of Bath
**Research origin:** MSc dissertation (2022)

[![Paper](https://img.shields.io/badge/arXiv-2610.10564-b31b1b.svg)](https://arxiv.org/abs/2610.10564)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-yellow)](https://huggingface.co/datasets/felixhsueh/texture-dynamics-slam-dataset)
[![Code License](https://img.shields.io/badge/Code-GPL--3.0-blue.svg)](LICENSE)
[![Data License](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](DATA_LICENSE.md)

## Research Resources

* **Paper:** [Read on arXiv](https://arxiv.org/abs/2610.10564)
* **Paper PDF:** [View the PDF in this repository](paper/paper.pdf)
* **Dataset:** [Texture-Dynamics SLAM Dataset on Hugging Face](https://huggingface.co/datasets/felixhsueh/texture-dynamics-slam-dataset)
* **Source code and results:** This repository
* **Code license:** GNU General Public License v3.0
* **Data and results license:** Creative Commons Attribution 4.0 International

## Overview

This study investigates how scene texture and dynamic objects affect ORB-SLAM2 performance, and whether dynamic-point filtering improves visual SLAM under controlled conditions.

The experiment uses **24 synthetic indoor sequences** in which surface texture and scene dynamics are systematically varied along identical camera trajectories. This controlled design helps isolate the effects of texture scarcity and moving objects on tracking and mapping performance.

The study evaluates four texture levels (L0–L3), three dynamic-scene conditions (D0–D2), three front-end configurations, two sensor configurations, and five runs per experimental condition.

The three filtering configurations are:

* **NONE:** No dynamic-point filtering.
* **FLOW:** Optical-flow and epipolar-residual filtering.
* **GEOM:** Multi-view depth-consistency filtering.

Both stereo and RGB-D configurations are evaluated, yielding **720 experimental runs**:

$$
2\ \text{scenes} \times 4\ \text{texture levels} \times 3\ \text{dynamic conditions} \times 3\ \text{filters} \times 2\ \text{sensors} \times 5\ \text{runs} = 720
$$

The repository provides the experiment scripts, ORB-SLAM2 modifications, evaluation tools, aggregate results, per-run logs, estimated trajectories, and figures associated with the paper.

Rendered images are not stored as a complete pre-rendered dataset. The rendering pipeline can regenerate the synthetic sequences deterministically.

## Research Questions

The study focuses on three questions:

1. Does dynamic-point filtering improve trajectory accuracy when scene texture is scarce?
2. How do texture level and scene dynamics interact with filtering performance?
3. What trade-offs emerge between trajectory accuracy, tracking completeness, and dynamic-point rejection?

The quantitative analysis includes absolute trajectory error (ATE), relative pose error (RPE), tracking completeness, keypoint-level precision and recall, false rejection rate (FRR), and statistical comparisons between experimental conditions.

See the paper for the full experimental protocol, quantitative findings, statistical tests, and limitations.

## Experimental Design

### Texture Conditions

Four texture levels are used to control the amount of visual texture available to the SLAM front-end:

* **L0–L3:** Increasing texture levels as defined by the renderer and experimental protocol.

The exact rendering parameters and texture metrics are implemented in `sim/render.py` and documented in the paper.

### Dynamic Conditions

Three dynamic-scene conditions are used:

* **D0:** No moving agents.
* **D1:** One moving agent.
* **D2:** Three moving agents.

The same camera trajectories are used across controlled conditions to support comparisons.

### Front-End Configurations

| Configuration | Description                                            |
| ------------- | ------------------------------------------------------ |
| `NONE`        | Baseline ORB-SLAM2 without dynamic-point filtering     |
| `FLOW`        | Filtering based on optical flow and epipolar residuals |
| `GEOM`        | Filtering based on multi-view depth consistency        |

### Sensor Configurations

* Stereo
* RGB-D

### Evaluation Metrics

| Metric         | Purpose                                                                                        |
| -------------- | ---------------------------------------------------------------------------------------------- |
| ATE            | Global trajectory accuracy after SE(3) alignment                                               |
| RPE            | Relative pose accuracy over a 1 m interval                                                     |
| Completeness   | Measures successful tracking or valid trajectory coverage as defined by the evaluation scripts |
| Precision (P)  | Precision of dynamic-point classification                                                      |
| Recall (R)     | Recall of dynamic-point classification                                                         |
| FRR            | False rejection rate                                                                           |
| Static inliers | Tracks or keypoints retained as static under the study's evaluation protocol                   |

The precise metric definitions and aggregation rules are implemented in `sim/evaluate.py` and `sim/analyze.py`.

## Repository Contents

| Path                            | Description                                                                                                                                                                                           |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sim/render.py`                 | CPU ray caster using Open3D `RaycastingScene`; generates stereo images, depth maps, dynamic masks, ground truth, texture metrics, and ORB-SLAM2 settings                                              |
| `sim/run_grid.py`               | Executes the experimental grid: render, run SLAM, evaluate, and remove generated images                                                                                                               |
| `sim/evaluate.py`               | Computes ATE using SE(3) alignment, RPE at 1 m, completeness, and per-keypoint precision, recall, and FRR                                                                                             |
| `sim/analyze.py`                | Aggregates experimental results, statistical analyses, tables, and summary statistics                                                                                                                 |
| `sim/make_figs.py`              | Generates the paper figures and associated analysis outputs                                                                                                                                           |
| `orbslam2/orbslam2_study.patch` | Patch against ORB-SLAM2 commit `f2e6f51cdc8d067655d90a78c06261378e07e8f3`, including Windows/OpenCV 4 portability changes, headless build support, binary vocabulary handling, filtering, and logging |
| `orbslam2/DynFilter.cc`         | Dynamic-point filtering implementation                                                                                                                                                                |
| `orbslam2/tools/`               | Sequence driver (`run_seq`) and vocabulary conversion utilities                                                                                                                                       |
| `orbslam2/build_orb.py`         | MSVC build script used for the study; local paths may need adjustment                                                                                                                                 |
| `results/runs.csv`              | Per-run experimental results                                                                                                                                                                          |
| `results/`                      | Aggregate tables, sequence metadata, statistical outputs, and grid logs                                                                                                                               |
| `data/run_logs.zip`             | Per-run frame logs, estimated trajectories, and example frames                                                                                                                                        |
| `figures/`                      | Figures associated with the paper                                                                                                                                                                     |
| `paper/`                        | Paper PDF and, if included, LaTeX source files                                                                                                                                                        |
| `requirements.txt`              | Pinned Python dependencies                                                                                                                                                                            |
| `LICENSE`                       | Code license                                                                                                                                                                                          |
| `DATA_LICENSE.md`               | Data and results license                                                                                                                                                                              |

## Paper-to-Files Mapping

The tables, figures, and reported numerical results are intended to be traceable to committed result files or reproducible analysis scripts.

| Paper item                                                   | Produced by                                                        | Source files                                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| Table I — Positioning                                        | Literature review                                                  | No experimental data                                                                 |
| Table II — Texture and dynamics by level                     | `sim/analyze.py`                                                   | `results/table_texture.csv`, `results/seq_stats.csv`, `results/beta_calibration.csv` |
| Table III — Main experimental grid                           | `sim/analyze.py`                                                   | `results/table_main.csv`, `results/latex_tables.tex`                                 |
| Table IV — Precision, recall, FRR, and static inliers at D2  | `sim/analyze.py`                                                   | `results/latex_tables.tex`, `results/stats.json`                                     |
| Figure 1 — Pipeline schematic                                | LaTeX                                                              | Paper source                                                                         |
| Figure 2 — Frames L0–L3                                      | `sim/make_figs.py`; frame 180 is re-rendered using `sim/render.py` | `figures/fig2_frames.pdf`                                                            |
| Figure 3 — ATE difference heat maps and Holm-corrected tests | `sim/make_figs.py`                                                 | `results/delta_ate_cells.csv`, `results/per_scene_cells.csv`, `results/stats.json`   |
| Figure 4 — FRR versus texture                                | `sim/make_figs.py`                                                 | `results/runs.csv`, `results/stats.json`                                             |
| Section IV numerical results                                 | `sim/analyze.py`                                                   | `results/stats.json`, `results/runs.csv`                                             |
| Per-run logs and trajectories                                | `sim/run_grid.py`                                                  | `data/run_logs.zip`, `results/grid_log.txt`                                          |
| Paper PDF and LaTeX source                                   | LaTeX                                                              | Files under `paper/`, if included                                                    |

**Important:** The table above documents the intended relationships. Actual file availability depends on the files committed to the repository.

## Reproducing the Analysis

If the committed result files are available, the aggregate tables, summary statistics, and figures can be regenerated without rerunning the SLAM experiments.

### 1. Set Up Python

Use a compatible Python environment. The study's dependency file is `requirements.txt`.

```bash
python -m venv .venv
```

Activate the environment.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Regenerate Tables and Figures

From the repository root:

```bash
python sim/analyze.py
python sim/make_figs.py
```

According to the project workflow, `sim/make_figs.py` invokes the analysis pipeline and regenerates the tables, `results/stats.json`, and Figures 2–4 from the committed result files.

The expected inputs include:

* `results/runs.csv`
* `results/seq_stats.csv`
* `data/run_logs.zip`

The original workflow reports an execution time of approximately 10 seconds for this analysis-only step. Actual runtime depends on the environment and the current implementation.

The project reports byte-for-byte reproducibility for the following outputs:

* `results/stats.json`
* `results/table_main.csv`
* `results/table_texture.csv`
* `results/delta_ate_cells.csv`
* `results/latex_tables.tex`

Figure styling may differ slightly from the committed PDFs.

**This analysis-only workflow does not rerun ORB-SLAM2.** It regenerates analysis outputs from previously committed experimental results.

## Reproducing the Full Experimental Grid

Full reproduction requires building the modified ORB-SLAM2 implementation and running the rendering and evaluation pipeline.

### 1. Obtain ORB-SLAM2

Obtain the ORB-SLAM2 source code at commit:

```text
f2e6f51cdc8d067655d90a78c06261378e07e8f3
```

Apply the study patch and copy the required filtering and tool sources according to the repository layout:

1. Apply `orbslam2/orbslam2_study.patch`.
2. Copy `DynFilter.cc` into the appropriate ORB-SLAM2 source directory.
3. Add the sources under `orbslam2/tools/` as required by the build instructions.
4. Build ORB-SLAM2 and configure the vocabulary and dataset paths.

The original study used **MSVC 19.44 and OpenCV 4.10 on Windows**. The original CMake build with the additional sources may work on Linux, but Linux reproduction has not been verified.

### 2. Convert the Vocabulary

Use the provided vocabulary conversion utility:

```bash
voc_convert ORBvoc.txt ORBvoc.bin
```

Make sure the executable is built and available on your `PATH`, or use its actual path.

### 3. Install Python Dependencies

Use a compatible Python environment and install:

```bash
pip install -r requirements.txt
```

Python 3.12 was used in the documented workflow. Other Python versions may require dependency adjustments.

### 4. Run the Full Grid

From the repository root:

```bash
python sim/run_grid.py --slam-jobs 3 --render-jobs 2
python sim/analyze.py
```

The grid runner performs rendering, SLAM execution, evaluation, and cleanup according to the implementation.

### 5. Run a Quick Check

A reduced run can be used to check the pipeline before launching the full experiment:

```bash
python sim/run_grid.py --scenes 1 --tex 0 2 --runs 2 --tag _check
```

This command is intended as a smoke test, not as a replacement for the full 720-run experiment.

### Reproduction Notes

* Update local paths in `orbslam2/build_orb.py` and any relevant configuration files.
* Ensure the ORB-SLAM2 vocabulary, executable, sensor configuration, and generated sequences match the study setup.
* Filtering parameters, renderer settings, and the experimental protocol are defined in the code and in Section III of the paper.
* Full reproduction time depends on the machine, build configuration, and rendering and SLAM settings.
* Do not interpret a successful analysis-only run as proof that the entire SLAM experiment has been independently reproduced.

## Dataset

The associated dataset is hosted on Hugging Face:

**[felixhsueh/texture-dynamics-slam-dataset](https://huggingface.co/datasets/felixhsueh/texture-dynamics-slam-dataset)**

The dataset and this repository serve related but distinct purposes:

* **GitHub:** Experimental code, result tables, analysis scripts, run logs, trajectories, and figures.
* **Hugging Face:** The published dataset artifact and its dataset card.

Refer to the dataset card for the exact files available for download, dataset organization, usage instructions, and licensing information.

The synthetic scenes and textures are procedurally generated according to the project description. The rendering pipeline is provided to support regeneration of experimental inputs.

## Paper and Dataset Relationship

This repository, the associated paper, and the Hugging Face dataset are intended to be used together:

* The paper documents the research question, methodology, results, and limitations.
* This repository provides the implementation and experimental analysis.
* The dataset repository provides a dedicated entry point for accessing the released dataset.

Paper: https://arxiv.org/abs/2610.10564

Dataset: https://huggingface.co/datasets/felixhsueh/texture-dynamics-slam-dataset

## Citation

If you use this work in academic research, please cite the associated paper.

```bibtex
@misc{xue2026dynamicpointfiltering,
  author        = {Xue, Zekui},
  title         = {Does Dynamic-Point Filtering Help When Texture Is Scarce? A Controlled Study of ORB-SLAM2 Front-Ends in Synthetic Indoor Scenes},
  year          = {2026},
  eprint        = {2610.10564},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}
```

The BibTeX entry above is a suggested format. Check the arXiv record for the official category and citation metadata before using it in a publication.

* **Paper:** https://arxiv.org/abs/2610.10564
* **Code:** https://github.com/felixxxue/texture-dynamics-slam
* **Dataset:** https://huggingface.co/datasets/felixhsueh/texture-dynamics-slam-dataset

## Limitations

This repository documents a controlled synthetic study. Its results should be interpreted within the limits of the scene-generation process, selected texture and dynamics conditions, ORB-SLAM2 implementation, and experimental protocol.

Synthetic indoor sequences enable controlled comparisons but do not, by themselves, establish that the same performance differences will occur in real-world environments.

The paper discusses the experimental findings and their limitations in detail. Platform-specific reproduction should be considered unverified unless it has been tested successfully.

## License

### Code

Code is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. See [`LICENSE`](LICENSE).

The ORB-SLAM2 study patch modifies GPLv3-licensed ORB-SLAM2 code. Review the relevant upstream license and the repository's license files when redistributing derivative works.

### Data and Results

The project declares the data and results under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. See [`DATA_LICENSE.md`](DATA_LICENSE.md).

The stated scope includes files under `results/`, `data/`, and `figures/`. Verify the exact scope and contents of those directories against the license notice before redistribution.

If you reuse the released data or results, retain appropriate attribution and refer to the associated paper.

---

For questions about the experimental protocol, code, or results, please open an issue in this repository.
