# Preprocessing

Scripts for preparing training data.
Run them in the numbered order.

---

## Prerequisite: Upstream Preprocessing

The NIfTI files consumed by these scripts are **assumed to have already undergone**:

- Co-registration of MRI (T1/T2) with the label map
- Intensity normalization of T1/T2 images (99th percentile clipping)

These upstream steps were performed in a preceding project (Model8 pipeline) and
**are NOT included in this repository**.
The `_after` suffix in filenames (e.g. `IXI012_T1_after.nii.gz`) indicates this processed state.

If you wish to run CondNet-SR on your own data, you must perform an equivalent
preprocessing pipeline on your side.

---

## Execution Order

```
01_stratified_split.py   →   02_export_png_and_paths.py
```

### Step 1: `01_stratified_split.py`

**Purpose**: Split the IXI dataset into train / val / test using age and sex stratification.

**Input** (set `INPUT_CSV` at the top of the script):
A subject metadata CSV with the following columns:
- `ID` — subject identifier
- `AGE` — age
- `Sex` — `"Male"` / `"Female"`

**Output** (saved under `OUT_DIR`):
- `train.csv`
- `val.csv`
- `test.csv`

**Key settings** (edit at the top of the script):
- `N_TRAIN`, `N_VAL`, `N_TEST` — number of cases per split
- `AGE_BIN_EDGES` — age-bin boundaries
- `TRAIN_SEX_TARGET` — sex ratio target for the training set (set `None` to disable)
- `RANDOM_SEED` — reproducibility

**Run**:
```bash
python tools/preprocess/01_stratified_split.py
```

---

### Step 2: `02_export_png_and_paths.py`

**Purpose**: For each split CSV from Step 1, decompose NIfTI volumes into per-slice PNG files
and generate the **path CSV** (`paths_all_slices.csv`) that the training dataloader reads.

**Input**:
- A split CSV from Step 1 (set via `CSV_PATH`)
- T1/T2 NIfTI: `<IMAGE_DIR>/IXI{id:03d}_T1_after.nii.gz`, `_T2_after.nii.gz`
- Label NIfTI: `<LABEL_DIR>/IXI{id:03d}_label_after.nii.gz`

**Output** (per-case folders under `OUT_DIR/<IXI_prefix>/`):
- `img/IXI{id:03d}_z{zzz}_T1.png`
- `img/IXI{id:03d}_z{zzz}_T2.png`
- `label/IXI{id:03d}_z{zzz}_label.png` — stored as `raw_label × 18` so the range maps to 0–252;
  the dataloader divides by 18 at read time
- `t2mask/IXI{id:03d}_z{zzz}_T2mask.png` — T2-missing mask (T1>100 ∧ T2==0)
- `<OUT_DIR>/paths_all_slices.csv` — aggregated path CSV for all slices

**Key settings**:
- `CSV_PATH` — which split CSV to process (run the script 3 times: once each for train/val/test)
- `LABEL_DIR`, `IMAGE_DIR` — where the NIfTI files live
- `OUT_DIR` — where PNGs are written
- `T1_THRESH`, `T2_ZERO_EPS` — thresholds for T2-missing mask

**Run** (re-run for each split by swapping `CSV_PATH`):
```bash
python tools/preprocess/02_export_png_and_paths.py
```

**Note**:
The `normalize_volume_to_uint8()` function inside this script performs only a
**volume-wise min-max mapping to 0–255** (for uint8 PNG storage) — it is not an
intensity-correction step. Intensity 99%ile normalization is assumed to be
already done upstream.

---

## Output CSV Columns (format expected by the dataloader)

`paths_all_slices.csv` contains the following columns and is read by
`datasets/dataloader.py::CondDataset`:

| Column | Description |
|---|---|
| `number` | Global slice index |
| `IXI_ID` | Subject ID (integer) |
| `age` | Subject age |
| `PREFIX` | IXI prefix (e.g. `IXI012`) |
| `slice` | Slice index along z |
| `t1_img` | Absolute path to T1 PNG |
| `t2_img` | Absolute path to T2 PNG |
| `label` | Absolute path to label PNG |
| `t2mask` | Absolute path to T2-missing mask PNG |

These paths are assigned to `data.train_csv` / `val_csv` / `test_csv`
in `configs/config.yaml`.

---

## FAQ

**Q. Paths are hardcoded — why?**
A. To keep the research code minimal, each script defines its paths as constants
near the top. Please edit them to match your environment.

**Q. Why are labels stored as `raw_label × 18` in PNGs?**
A. Labels in the range 0–14 would render nearly black in a PNG viewer, which makes
visual inspection hard. Multiplying by 18 stretches them to 0–252. The dataloader
reverses this with an integer divide by 18 at read time.
