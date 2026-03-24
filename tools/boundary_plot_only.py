# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:19:45 2026

@author: kubota
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# ============================================================
# 設定
# ============================================================
CFG = {
    # 症例ごとの 5列CSV が入っているフォルダ
    "folder": r"D:/Kubota/data/Model10test/gm_wm_boundary_analysis/M3_logmae099_099_04",

    # 症例ID
    "pids": [46, 122, 254, 304, 362, 369, 401, 437, 441, 575],

    # 列順
    "order": ["gm_m2", "gm_m1", "boundary", "wm_p1", "wm_p2"],

    # 表示名
    "label_map": {
        "gm_m2": "-2",
        "gm_m1": "-1",
        "boundary": "0",
        "wm_p1": "1",
        "wm_p2": "2",
    },

    # 各症例・各領域から何点使うか
    "max_points_per_region_per_case": 10000,

    # y軸範囲
    "ymin": 0.0,
    "ymax": 0.20,

    # 表示・保存
    "show_figure": True,
    "save_png": False,
    "save_svg": True,

    # 出力ファイル名
    "out_png_name": "gm_wm_boundary_violin_all_cases.png",
    "out_svg_name": "gm_wm_boundary_violin_all_cases.svg",

    # 図サイズ
    "figsize": (8, 6),

    # 乱数seed
    "random_seed": 0,
}
# ============================================================


def load_sampled_data(cfg):
    folder = cfg["folder"]
    pids = cfg["pids"]
    order = cfg["order"]
    max_points = int(cfg["max_points_per_region_per_case"])
    rng = np.random.default_rng(cfg["random_seed"])

    region_values = {r: [] for r in order}

    for pid in pids:
        csv_path = os.path.join(folder, f"boundary_voxels_pid_{pid}.csv")

        if not os.path.exists(csv_path):
            print(f"Skip (not found): {csv_path}")
            continue

        df = pd.read_csv(
            csv_path,
            usecols=order,
            dtype={c: "float32" for c in order}
        )

        print(f"Loaded PID {pid}: {df.shape}")

        for r in order:
            vals = df[r].dropna().to_numpy(dtype=np.float32)

            if vals.size == 0:
                continue

            if vals.size > max_points:
                idx = rng.choice(vals.size, size=max_points, replace=False)
                vals = vals[idx]

            region_values[r].append(vals)

    data_list = []
    labels_present = []

    for r in order:
        if len(region_values[r]) == 0:
            continue

        vals_all = np.concatenate(region_values[r])
        data_list.append(vals_all)
        labels_present.append(r)

        print(f"{r}: {len(vals_all)} points")

    if len(data_list) == 0:
        raise ValueError("No data loaded. CSV path or filenames may be incorrect.")

    return data_list, labels_present


def plot_violin_box(data_list, labels_present, cfg):
    plt.rcParams["font.family"] = "Arial"

    label_map = cfg["label_map"]
    ymin = cfg["ymin"]
    ymax = cfg["ymax"]

    fig, ax = plt.subplots(figsize=cfg["figsize"])
    positions = np.arange(1, len(data_list) + 1)

    # ---- GM側背景（-1と0の中間まで） ----
    ax.axvspan(0.5, 2.5, color="#d9d9d9", alpha=0.4, zorder=0)

    # ---- バイオリン ----
    vp = ax.violinplot(
        data_list,
        widths=0.9,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    for body in vp["bodies"]:
        body.set_facecolor("#c6dbef")
        body.set_edgecolor("none")
        body.set_alpha(0.8)

    # ---- ボックス ----
    ax.boxplot(
        data_list,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="#3182bd", edgecolor="black", linewidth=1.0),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
        medianprops=dict(color="black", linewidth=1.2),
    )

    # ---- 軸設定 ----
    ax.set_xticks(positions)
    ax.set_xticklabels([label_map[r] for r in labels_present])

    ax.tick_params(labelsize=28)

    ax.set_ylim(ymin, ymax)

    ax.yaxis.set_major_locator(MultipleLocator(0.05))

    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.canvas.draw()

    # ---- 保存 ----
    if cfg["save_png"] or cfg["save_svg"]:
        out_dir = cfg["folder"]
        os.makedirs(out_dir, exist_ok=True)

        if cfg["save_png"]:
            png_path = os.path.join(out_dir, cfg["out_png_name"])
            plt.savefig(png_path, dpi=300, bbox_inches="tight")
            print(f"Saved PNG: {png_path}")

        if cfg["save_svg"]:
            svg_path = os.path.join(out_dir, cfg["out_svg_name"])
            plt.savefig(svg_path, bbox_inches="tight")
            print(f"Saved SVG: {svg_path}")

    # ---- 表示 ----
    if cfg["show_figure"]:
        plt.show(block=False)
        plt.pause(0.1)
    else:
        plt.close(fig)


def main():
    data_list, labels_present = load_sampled_data(CFG)
    plot_violin_box(data_list, labels_present, CFG)
    print("Done.")


if __name__ == "__main__":
    main()