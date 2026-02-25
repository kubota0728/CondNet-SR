# CondNet-CR

This repository contains the implementation of CondNet-CR,  
a conductivity estimation framework based on a pretrained segmentation backbone (CondNet-TART).

---

## Overview

CondNet-CR consists of:

- Segmentation backbone (UNet + Transformer)
- Transfer head for conductivity regression
- Training / validation / evaluation pipeline
- Automatic checkpoint and loss curve saving

---

## Project Structure

```
CondNet-CR/
├── configs/
├── datasets/
├── engine/
├── model/
├── pretrain_seg/        # Segmentation pretraining scripts
├── checkpoints/         # (ignored by git)
├── logs/                # (ignored by git)
└── main.py
```

---

## Pretraining Requirement

The segmentation backbone must be pretrained before running CondNet-CR.

Pretraining scripts are provided in:

pretrain_seg/

After pretraining, specify the backbone weights in:

model:
  pretrained_unet_path: "path/to/pretrained_unet.pth"

---

## Training

Edit the configuration file:

configs/config.yaml

Set:

run:
  mode: train

Then run:

python main.py --config configs/config.yaml

---

## Evaluation

Set:

run:
  mode: eval
  load_pth: "checkpoints/exp01/best.pth"

Then run:

python main.py --config configs/config.yaml

---

## Output

Training automatically saves:

- best.pth
- last.pth
- epoch_xxx.pth
- loss_curve.png
- history.pkl

All outputs are saved under:

checkpoints/<exp_name>/

---

## Notes

- Dataset and checkpoints are not included in this repository.
- Modify paths in configs/config.yaml according to your environment.