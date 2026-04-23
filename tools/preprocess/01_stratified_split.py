# -*- coding: utf-8 -*-
"""
Split the IXI subject list into train / val / test with age-and-sex stratification.

Step 1 of the preprocessing pipeline.

Input:
  Subject-metadata CSV (set INPUT_CSV below) with columns:
    - ID   : subject identifier
    - AGE  : age in years
    - Sex  : "Male" / "Female"

Output:
  Three CSVs under OUT_DIR:
    - train.csv
    - val.csv
    - test.csv

Stratification:
  The training set is sampled to roughly match TRAIN_SEX_TARGET across the age
  bins defined by AGE_BIN_EDGES. Val / test are drawn randomly from the
  remaining subjects. Targets are relaxed automatically when a bin lacks
  enough samples.

Key settings (edit constants below):
  INPUT_CSV, OUT_DIR, N_TRAIN / N_VAL / N_TEST,
  AGE_BIN_EDGES, TRAIN_SEX_TARGET, RANDOM_SEED.

See tools/preprocess/README.md for the full pipeline context.

Created on Thu Feb 26 11:35:13 2026
@author: kubota
"""

from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd

# =========================
# Settings
# =========================
INPUT_CSV = r"D:\kubota\Data\Model10/20260226_data.csv"   # <- あなたのファイル
OUT_DIR = r"D:\kubota\Data\Model10/split_out"

N_TRAIN = 20
N_VAL   = 5
N_TEST  = 10

RANDOM_SEED = 42

# 年齢ビン：下限は 0 推奨（20にすると20未満が脱落します）
AGE_BIN_EDGES = [0, 30, 40, 50, 60, 70, 200]  # 例：～29, 30-39, ..., 70+

# trainの性別比を「目標」として指定（無理なら自動で緩める）
# None にすると性別は気にせず年齢だけ均等に取ります
TRAIN_SEX_TARGET = {"Male": 0.5, "Female": 0.5}  # or None

# trainの年齢ビンを均等にしたいなら None のままでOK（全ビン同重み）
TRAIN_AGE_BIN_WEIGHTS = None  # 例：{"(0, 30]":1, "(30, 40]":1, ...}

# =========================
# Helpers
# =========================
def make_age_bins(df: pd.DataFrame, edges):
    return pd.cut(df["AGE"], bins=edges, right=True, include_lowest=True).astype(str)

def normalize_weights(keys, w):
    if w is None:
        ww = {k: 1.0 for k in keys}
    else:
        ww = {k: float(w.get(k, 1.0)) for k in keys}
    s = sum(ww.values())
    return {k: ww[k] / s for k in keys}

def allocate_counts(total, keys, weights, avail):
    raw = {k: total * weights[k] for k in keys}
    base = {k: int(math.floor(raw[k])) for k in keys}
    rem = total - sum(base.values())

    frac_order = sorted(keys, key=lambda k: (raw[k] - base[k]), reverse=True)
    i = 0
    while rem > 0 and i < len(frac_order) * 3:
        k = frac_order[i % len(frac_order)]
        base[k] += 1
        rem -= 1
        i += 1

    # 上限超過を削る
    for k in keys:
        if base[k] > avail.get(k, 0):
            base[k] = avail.get(k, 0)

    # 足りない分を余裕のあるところに足す
    shortage = total - sum(base.values())
    while shortage > 0:
        candidates = [k for k in keys if base[k] < avail.get(k, 0)]
        if not candidates:
            break
        candidates = sorted(candidates, key=lambda k: (weights[k], avail.get(k, 0) - base[k]), reverse=True)
        base[candidates[0]] += 1
        shortage -= 1

    return base

def sample_train_stratified(df, n_train, rng):
    df = df.copy()
    df["AGE_BIN"] = make_age_bins(df, AGE_BIN_EDGES)
    bins = sorted(df["AGE_BIN"].unique().tolist())

    w_age = normalize_weights(bins, TRAIN_AGE_BIN_WEIGHTS)
    avail_bins = {b: int((df["AGE_BIN"] == b).sum()) for b in bins}
    bin_counts = allocate_counts(n_train, bins, w_age, avail_bins)

    picked = []
    for b in bins:
        n_b = bin_counts[b]
        if n_b <= 0:
            continue
        pool = df[df["AGE_BIN"] == b]

        # 性別ターゲットがある場合は、可能な範囲で近づける
        if TRAIN_SEX_TARGET is None:
            picked.append(pool.sample(n=n_b, replace=False, random_state=rng.integers(0, 2**32 - 1)))
        else:
            sexes = pool["Sex"].dropna().unique().tolist()
            w_sex = normalize_weights(sexes, TRAIN_SEX_TARGET)
            avail_sex = {s: int((pool["Sex"] == s).sum()) for s in sexes}
            sex_counts = allocate_counts(n_b, sexes, w_sex, avail_sex)

            part = []
            for s in sexes:
                k = sex_counts[s]
                if k > 0:
                    part.append(pool[pool["Sex"] == s].sample(
                        n=k, replace=False, random_state=rng.integers(0, 2**32 - 1)
                    ))
            part = pd.concat(part, axis=0) if part else pool.iloc[0:0]

            # まだ不足なら性別無視で埋める（ここが「自動で緩める」部分）
            if len(part) < n_b:
                remain = pool.drop(index=part.index)
                need = n_b - len(part)
                if need > 0 and len(remain) >= need:
                    part = pd.concat([part, remain.sample(
                        n=need, replace=False, random_state=rng.integers(0, 2**32 - 1)
                    )], axis=0)

            picked.append(part)

    train = pd.concat(picked, axis=0) if picked else df.iloc[0:0]

    # 念のため不足があれば全体から埋める（最終保険）
    if len(train) < n_train:
        need = n_train - len(train)
        rest = df.drop(index=train.index)
        if len(rest) < need:
            raise RuntimeError(f"Not enough samples for train: need {need}, remaining {len(rest)}")
        train = pd.concat([train, rest.sample(
            n=need, replace=False, random_state=rng.integers(0, 2**32 - 1)
        )], axis=0)

    return train.drop(columns=["AGE_BIN"], errors="ignore")

def summarize(df, name):
    print("\n" + "="*60)
    print(f"[{name}] n={len(df)}  AGE(min/mean/max)={df['AGE'].min():.2f}/{df['AGE'].mean():.2f}/{df['AGE'].max():.2f}")
    print("- Sex")
    print(df["Sex"].value_counts(dropna=False).to_string())
    # 年齢ビン分布（表示だけ）
    tmp = df.copy()
    tmp["AGE_BIN"] = make_age_bins(tmp, AGE_BIN_EDGES)
    print("- Age bins")
    print(tmp["AGE_BIN"].value_counts(dropna=False).sort_index().to_string())
    print("="*60)

# =========================
# Main
# =========================
def main():
    rng = np.random.default_rng(RANDOM_SEED)

    df = pd.read_csv(INPUT_CSV)
    for c in ["ID", "Sex", "AGE"]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'. Columns={df.columns.tolist()}")

    # trainだけ層化
    train = sample_train_stratified(df, N_TRAIN, rng=rng)

    # 残りから val, test はランダム（優先度低い想定）
    remain = df.drop(index=train.index).copy()

    if len(remain) < (N_VAL + N_TEST):
        raise RuntimeError(f"Not enough remaining for val+test: remaining={len(remain)}, need={N_VAL+N_TEST}")

    val = remain.sample(n=N_VAL, replace=False, random_state=rng.integers(0, 2**32 - 1))
    remain2 = remain.drop(index=val.index)

    test = remain2.sample(n=N_TEST, replace=False, random_state=rng.integers(0, 2**32 - 1))

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    train.sort_values("ID").to_csv(out_dir / "train.csv", index=False)
    val.sort_values("ID").to_csv(out_dir / "val.csv", index=False)
    test.sort_values("ID").to_csv(out_dir / "test.csv", index=False)

    summarize(df, "all")
    summarize(train, "train")
    summarize(val, "val")
    summarize(test, "test")

    print("\nSaved:", out_dir.resolve())

if __name__ == "__main__":
    main()