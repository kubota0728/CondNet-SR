# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:20:23 2026

@author: kubota
"""

import os
import pandas as pd


# 添付CSVのパス
input_csv = r"D:\kubota\Data\Model10test\profiles_local_sphere_grad\M1_sphereR50.0_top10_mean_gradTop10.0/all_cases_M1_local_sphere_shell_profiles_grad_long.csv"

# 出力CSV
output_csv = r"D:\kubota\Data\Model10test\profiles_local_sphere_grad\M1_sphereR50.0_top10_mean_gradTop10.0/M1.csv"


def pick_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"候補列が見つかりません: {candidates}\n実際の列: {list(df.columns)}")


def main():
    df = pd.read_csv(input_csv)

    case_col = pick_column(df, ["case_id", "Case", "case"])
    depth_col = pick_column(df, ["distance_mm", "depth_mm", "depth"])
    grad_col = pick_column(df, ["grad_mean", "gradient_mean", "grad", "gradient"])

    df = df[[case_col, depth_col, grad_col]].copy()
    df.columns = ["case_id", "distance_mm", "gradient"]

    # 数値化
    df["distance_mm"] = pd.to_numeric(df["distance_mm"], errors="coerce")
    df["gradient"] = pd.to_numeric(df["gradient"], errors="coerce")

    # wide形式へ変換
    # 行: 症例, 列: 深さ
    wide = df.pivot_table(
        index="case_id",
        columns="distance_mm",
        values="gradient",
        aggfunc="mean"
    )

    # 深さ順に並べる
    wide = wide.reindex(sorted(wide.columns), axis=1)

    # 列名をわかりやすくする
    wide.columns = [f"{c:.1f}mm" for c in wide.columns]

    # 症例名を列に戻す
    wide = wide.reset_index()

    # 保存
    wide.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("入力CSV :", input_csv)
    print("出力CSV :", output_csv)
    print(wide)


if __name__ == "__main__":
    main()