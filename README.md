# Egg segmentation and tracking (CV intern task)

**Report:** [report/CV_Intern_Task_Report.pdf](report/CV_Intern_Task_Report.pdf)
**Task 4 tracked video:** [Google Drive](https://drive.google.com/file/d/1hbDwUgPEwV0BYkP3mvo0WxX8rCifEjkQ/view?usp=sharing)
**AI-tool disclosure:** [docs/ai_usage.md](docs/ai_usage.md)

| Task | What | Code |
|---|---|---|
| 1 | Pretrained segmentation (SAM 3) → ground-truth masks | `scripts/run_task1.py`, `modal_app.py`, `scripts/make_gt.py` |
| 2 | Leakage-aware 4:1 dev/eval split, classical segmentation, experiment log | `scripts/make_split.py`, `scripts/task21_eda.py`, `scripts/task21_experiments.py`, `scripts/run_task2.py`, `src/eggseg/simple.py` |
| 3 | Evaluation against the Task 1 masks | `scripts/task3_eval.py`, `src/eggseg/metrics.py` |
| 4 | Egg tracking in the video (classical segmenter + Kalman filter tracker) | `scripts/run_task4.py`, `src/eggseg/track.py` |

## Data

- Dataset: [Roboflow](https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1) or the [zip file](https://drive.google.com/file/d/1a58rACcCFnaj0ZiDoNyGbDWXunRyWziv/view?usp=sharing). Copy the 110 JPGs into `data/images/`.
- Video: [YouTube](https://www.youtube.com/watch?v=Y4JyV94y2_A). Download it as mp4 and save it as `data/video/eggs.mp4`.

Both are not committed. Included in the repo:

```
data/gt_masks/           Task 1 ground truth: one union mask per image (PNG) + instances/ (per-egg masks, NPZ)
data/gt_masks/sources.csv  which Task 1 method each mask came from
data/sessions.csv        photo-session group per image (the leakage unit for the split)
data/scene_tags.csv      scene type per image (stratification, per-scene results)
splits/dev.txt, eval.txt the 87 / 23 image split
```

## Setup

Python 3.11+ (developed on 3.13, Windows).

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements-local.txt
```

Tasks 2–4 run on CPU. Task 1 needs a GPU (see below). The ground-truth masks it produces are already in `data/gt_masks/`, so Tasks 2–4 can be run without it.

## Task 1: pretrained segmentation

Run on a Modal cloud GPU (H100), which is what was used:

1. `python -m modal setup`
2. `facebook/sam3` is gated: request access on Hugging Face, create a read token, then
   `python -m modal secret create huggingface HF_TOKEN=<your token>`
3. Upload the images once:
   ```bash
   python -m modal volume create nosh-egg
   python -m modal volume put nosh-egg data/images /data/images
   ```
   On Git Bash, set `MSYS_NO_PATHCONV=1` first.
4. Run and download:
   ```bash
   python -m modal run modal_app.py --out task1
   python -m modal volume get nosh-egg /outputs/task1 outputs/
   ```
5. Build the ground truth (SAM 3 text prompt "egg"; Grounding DINO + SAM 3 where SAM 3 finds nothing)
   and compare the methods:
   ```bash
   python scripts/make_gt.py --task1 outputs/task1 --out data/gt_masks
   python scripts/task1_compare.py
   ```

On a local GPU instead: install PyTorch for your CUDA version, `pip install -r requirements.txt`,
`huggingface-cli login`, then `python scripts/run_task1.py --images data/images --out outputs/task1` and step 5.

**Committed Task 1 results (`outputs/task1/`):**

| Path | Content |
|---|---|
| `method_comparison.csv` | per method: images with/without an egg, eggs found, count = GT, IoU vs GT |
| `pairwise_iou.csv` | mean union-mask IoU between every pair of methods |
| `summary.csv` | per image: egg count per method and pairwise IoU |
| `review_these.txt` | 38 images where the methods disagree |
| `gt_review/` | the 7 images where SAM 3 text and Grounding DINO + SAM 3 disagree (contact sheet + CSV) |
| `<method>/masks/`, `<method>/results.json` | masks and per-image counts for all 5 setups |
| `sam3_text/overlays/` | overlays of the chosen method |
| `comparison/` | per image, the original plus all 5 methods side by side |
| `pip_freeze.txt` | exact package versions of the GPU run |

## Task 2: classical segmentation (dev split only)

```bash
python scripts/make_split.py            # splits/dev.txt, splits/eval.txt + outputs/task2/split/ (leakage check)
python scripts/task21_eda.py            # outputs/task2_1/eda/ (EDA plots; image clusters used by the log)
python scripts/task21_experiments.py    # outputs/task2_1/experiments/experiment_log.csv (all trials)
python scripts/run_task2.py --split dev # final algorithm -> outputs/task2_1/final/dev/
python scripts/task21_figures.py        # outputs/task2_1/figures/
python scripts/run_task2.py --image path/to/image.jpg --out outputs/demo   # any single image
```

**Committed Task 2 results:**

| Path | Content |
|---|---|
| `outputs/task2/split/` | `split_summary.json` (sizes, sessions, scene counts), `leakage_check.png` |
| `outputs/task2_1/eda/` | EDA plots A–J, `per_image.csv`, `eda_summary.json` |
| `outputs/task2_1/experiments/` | `experiment_log.csv` (all trials), per-trial per-image results, `iou_by_cluster.csv` |
| `outputs/task2_1/figures/` | IoU over the trials, per cluster, final dev examples |
| `outputs/task2_1/final/dev/` | final algorithm on dev: masks, overlays (red = prediction, green = GT), `metrics.csv`, `summary.json` |

## Task 3: evaluation (run once, after Task 2 is frozen)

```bash
python scripts/run_task2.py --split eval   # outputs/task2_1/final/eval/
python scripts/task3_eval.py               # outputs/task3/ (metrics, per-scene table, figures, gallery)
```

## Task 4: video tracking

```bash
python scripts/run_task4.py               # outputs/task4/tracked.mp4, tracks.csv, summary.json (~3 min on 8 cores)
```

**Task 4 tracked video:** https://drive.google.com/file/d/1hbDwUgPEwV0BYkP3mvo0WxX8rCifEjkQ/view?usp=sharing
