# CondNet-SR

This repository contains the implementation of **CondNet-SR**,
a conductivity estimation framework based on a pretrained segmentation backbone
(**CondNet-TART**: UNet + Transformer) with the **Statistical-Rank (SR) loss**.

> Companion code for:
> Y. Kubota et al., *"MRI-Constrained Estimation of Layer-Dependent Gray Matter Conductivity:
> Implications for Brain Stimulation Electric Fields"* (under review).

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Upstream preprocessing  (NOT included in this repo)            │
│  - Co-registration of MRI (T1/T2) and label maps                │
│  - 99%ile intensity normalization                               │
│  → produces IXI{id:03d}_{T1,T2,label}_after.nii.gz              │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Local preprocessing        (tools/preprocess/)         │
│    01_stratified_split.py           IXI → train/val/test CSV    │
│    02_export_png_and_paths.py       NIfTI → slice PNGs + CSV    │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: Segmentation pretraining   (tools/pretrain/)           │
│    seg_pretrain.py                  UNet+Transformer → best.pth │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓ (set model.pretrained_unet_path)
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: CondNet-SR train / eval    (main.py)                   │
│    python main.py --config configs/config.yaml                  │
└─────────────────────────────────────────────────────────────────┘
```

**Each step is required** — skipping the pretraining step will cause CondNet-SR to fail
(the backbone weights `model.pretrained_unet_path` must exist before Step 3).

---

## Project Structure

```
CondNet-SR/
├── main.py                     # Entry point for train / eval (Step 3)
├── configs/
│   └── config.yaml             # Main config
├── datasets/
│   └── dataloader.py           # CondDataset (reads paths_all_slices.csv)
├── engine/
│   ├── builder.py              # Model / optimizer construction
│   └── trainer.py              # Training / evaluation loop
├── model/
│   └── condnet_tart.py         # UNet_2D backbone + CondNet_Transfer head
├── losses/
│   └── losses.py               # SR loss, MAE / logMAE, etc.
├── utils/
│   └── logger.py
├── tools/
│   ├── preprocess/             # Step 1 — see tools/preprocess/README.md
│   │   ├── 01_stratified_split.py
│   │   ├── 02_export_png_and_paths.py
│   │   └── README.md
│   └── pretrain/               # Step 2 — see tools/pretrain/README.md
│       ├── seg_pretrain.py
│       └── README.md
├── checkpoints/                # (gitignored)
├── logs/                       # (gitignored)
└── README.md                   # (this file)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install torch torchvision
pip install segmentation_models_pytorch
pip install opencv-python pillow pandas pyyaml SimpleITK nibabel
```

### 2. Prepare data (Step 1)

See [`tools/preprocess/README.md`](tools/preprocess/README.md) for details.

```bash
# Edit paths at the top of each script, then:
python tools/preprocess/01_stratified_split.py
python tools/preprocess/02_export_png_and_paths.py
```

### 3. Pretrain segmentation backbone (Step 2)

See [`tools/pretrain/README.md`](tools/pretrain/README.md) for details.

```bash
# Edit Config() in the script, then:
python tools/pretrain/seg_pretrain.py
```

The output `best.pth` must be specified in `configs/config.yaml` under
`model.pretrained_unet_path`.

### 4. Train CondNet-SR (Step 3)

Edit `configs/config.yaml`:

```yaml
run:
  mode: train
data:
  train_csv: "path/to/train_paths.csv"
  val_csv:   "path/to/val_paths.csv"
model:
  pretrained_unet_path: "path/to/best.pth"    # from Step 2
```

Then:

```bash
python main.py --config configs/config.yaml
```

### 5. Evaluate

Edit `configs/config.yaml`:

```yaml
run:
  mode: eval
eval:
  load_pth: "checkpoints/<timestamp>/best.pth"
  save_predictions: true
  pred_dir: "path/to/pred_output"
```

Then:

```bash
python main.py --config configs/config.yaml
```

---

## Outputs

Training automatically saves under `checkpoints/<timestamp>/`:

- `best.pth` — best validation epoch
- `last.pth` — final epoch
- `epoch_xxx.pth` — per-epoch checkpoints
- `loss_curve.png` — train/val loss plot
- `history.pkl` — full loss history

Logs are saved under `logs/<timestamp>.log`.

---

## Configuration Notes

The main config `configs/config.yaml` controls everything in Step 3.

Key fields:

| Field | Meaning |
|---|---|
| `run.mode` | `train` / `eval` / `test` / `predict` |
| `model.pretrained_unet_path` | **Required**. Path to `best.pth` from Step 2 |
| `model.freeze_keywords` | Backbone modules frozen during transfer (default: TCB1-5, Trans) |
| `model.multi_model` | `true`: original + ctnet separately; `false`: single model |
| `loss.name` | `mae` / `logmae` / `condnet_sr` (SR loss) |
| `loss.lambda_stat`, `lambda_rank`, `lambda_smooth` | SR loss weights |

---

## Notes

- Dataset and checkpoints are **not** included in this repository.
- All paths in `configs/config.yaml` and preprocessing scripts are hardcoded
  to the author's environment — please modify for your own use.
- Electric field simulation and layered-profile analysis pipelines are **not**
  included in this repository (those are handled separately).
- The `seg_pretrain.py` `UNet_2D` and `model/condnet_tart.py` `UNet_2D` must
  stay architecturally identical for weight transfer to work.

---

## Citation

This code accompanies the following paper (currently under review):

```
Y. Kubota et al.,
"MRI-Constrained Estimation of Layer-Dependent Gray Matter Conductivity:
 Implications for Brain Stimulation Electric Fields"
(Under review)
```

This work builds on our earlier transfer-learning framework:

```
Y. Kubota, S. Kodera, and A. Hirata,
"A novel transfer learning framework for non-uniform conductivity estimation
 with limited data in personalized brain stimulation,"
Physics in Medicine & Biology, vol. 70, no. 10, p. 105002, May 2025.
DOI: 10.1088/1361-6560/add105
```

(Full bibliographic details for the first paper will be added upon publication.)
