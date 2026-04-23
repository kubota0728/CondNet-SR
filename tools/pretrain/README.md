# Pretraining (Segmentation Backbone)

Script for **pretraining** the CondNet-SR backbone (UNet + Transformer) on a
segmentation task. This step is **required** — CondNet-SR cannot be trained
without the backbone weights produced here.

---

## Position in the Pipeline

```
Preprocessing  →  Pretraining (this step)  →  CondNet-SR training
                         │                              ▲
                         ↓                              │
                     best.pth ─────────────→  pretrained_unet_path
```

The `best.pth` produced by `seg_pretrain.py` must be specified via
`model.pretrained_unet_path` in `configs/config.yaml`, and is loaded when
`main.py` runs.

---

## Architecture

- **UNet + 6 Transformer blocks** (`TransformerBlock(embed_size=512, heads=4) × 6`)
- Input: T1 + T2 + T2-missing mask, concatenated along the channel dimension
- Output: segmentation logits with **15 classes** (labels 0–14)
- Loss: `0.5 × BCE + 0.5 × Tversky`

**Important**: The `UNet_2D` defined in `tools/pretrain/seg_pretrain.py` and the
`UNet_2D` defined in `model/condnet_tart.py` **must remain architecturally
identical**. Their `state_dict`s are compatible by design so that weights
transfer cleanly; changing one without the other will break weight transfer.

---

## How to Run

### 1. Preprocessing must be complete

The `paths_all_slices.csv` files (for train / val / test) produced by
`tools/preprocess/` must already exist.

### 2. Edit the `Config` dataclass at the top of `seg_pretrain.py`

```python
@dataclass
class Config:
    train_csv: str = "D:/kubota/Data/Model10test/filepath/train_df_20260226.csv"
    val_csv:   str = "D:/kubota/Data/Model10test/filepath/val_df_20260226.csv"
    test_csv:  str = "D:/kubota/Data/Model10test/filepath/test_df_20260226.csv"
    out_dir:   str = "D:/kubota/Data/Model10test/seg_result/"
    ...
    batch_size: int = 32
    epochs:     int = 50
    lr:         float = 1e-3
```

### 3. Execute

```bash
python tools/pretrain/seg_pretrain.py
```

### 4. Point the main config to the produced weights

Edit `configs/config.yaml`:

```yaml
model:
  pretrained_unet_path: "<out_dir>/best.pth"
```

---

## Outputs

Files written under `out_dir`:

| File | Content |
|---|---|
| `best.pth` | Weights from the epoch with the lowest validation loss (use this for transfer) |
| `last.pth` | Weights from the final epoch |
| `train_{epoch}.pth` | Weights at each epoch |
| `history.pkl` | Training / validation loss history (pickle) |
| `output.log` | Captured stdout log |

---

## Dependencies

In addition to standard PyTorch:

```bash
pip install segmentation_models_pytorch
pip install opencv-python pillow
```

The loss functions (`TverskyLoss`, `SoftBCEWithLogitsLoss`) come from
`segmentation_models_pytorch`.

---

## Input Data Format

The script reads `paths_all_slices.csv` produced by the preprocessing step.
Required columns:
`t1_img`, `t2_img`, `label`, `t2mask`.

Note: do **not** pass the raw split CSVs from `01_stratified_split.py` here —
pass the path CSVs produced by `02_export_png_and_paths.py`.

---

## References

The transfer-learning framework used here builds on our earlier work:

```
Y. Kubota, S. Kodera, and A. Hirata,
"A novel transfer learning framework for non-uniform conductivity estimation
 with limited data in personalized brain stimulation,"
Physics in Medicine & Biology, vol. 70, no. 10, p. 105002, May 2025.
DOI: 10.1088/1361-6560/add105
```
