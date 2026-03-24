# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:58:33 2026

@author: kubota
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# ============================================================
# 設定
# ============================================================
CFG = {
    # 入力CSV
    "csv_path": r"D:/Kubota/data/Model10test/csv/boxplot/GM_spearman.csv",

    # 使用する列名
    "columns": ["original", "mixed", "proposed"],

    # 表示名
    "label_map": {
        "original": "Original",
        "mixed": "Mixed",
        "proposed": "Proposed",
    },

    # 基準値（不要なら None）
    "reference_value": None,

    # 軸ラベル・タイトル
    "ylabel": "CV",
    "title": "Comparison of variability",

    # y軸範囲（自動にしたいなら None）
    "ymin": -0.4,
    "ymax": 1.0,

    # y軸目盛間隔（不要なら None）
    "y_major_step": 0.2,

    # フォント
    "font_family": "Arial",

    # 図サイズ
    "figsize": (6, 4),

    # 表示・保存
    "show_figure": True,
    "save_svg": True,
    "save_png": False,

    # 出力ファイル名
    "out_svg_name": "D:/Kubota/data/Model10test/csv/boxplot/GM_spearman.svg",
    "out_png_name": "D:/Kubota/data/Model10test/csv/boxplot/WM_CV.png",
}
# ============================================================


def load_plot_data(csv_path, columns):
    df = pd.read_excel(csv_path)

    missing_cols = [c for c in columns if c not in df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"CSV に必要な列がありません: {missing_cols}")

    data_list = []
    labels_present = []

    for c in columns:
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        data_list.append(vals.to_numpy())
        labels_present.append(c)

    if len(data_list) == 0:
        raise ValueError("有効なデータ列がありません。")

    return data_list, labels_present


def plot_box_comparison(data_list, labels_present, cfg):
    plt.rcParams["font.family"] = cfg["font_family"]

    fig, ax = plt.subplots(figsize=cfg["figsize"])

    positions = range(1, len(data_list) + 1)

    ax.boxplot(
        data_list,
        positions=positions,
        widths=0.7,
        patch_artist=True,
        showfliers=False,
        #showmeans=True,
        #meanprops=dict(
        #    marker="o",
        #    markerfacecolor="#08519c",   # 濃い青
        #    markeredgecolor="#08519c",
        #    markersize=6
        #),
        boxprops=dict(facecolor="#9ecae1", edgecolor="black", linewidth=1.0),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
        medianprops=dict(color="black", linewidth=1.2),
    )

    # 基準値ライン
    ref = cfg.get("reference_value", None)
    if ref is not None:
        ax.axhline(
            y=ref,
            color="orange",
            linestyle="-",
            linewidth=1.5,
            alpha=0.8
        )

    ax.set_xticks(list(positions))
    ax.set_xticklabels([cfg["label_map"].get(c, c) for c in labels_present])

    #ax.set_ylabel(cfg["ylabel"], fontsize=18)
    #ax.set_title(cfg["title"], fontsize=18)
    ax.tick_params(labelsize=24)

    # y軸範囲
    ymin = cfg.get("ymin", None)
    ymax = cfg.get("ymax", None)
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)

    # y軸目盛
    y_major_step = cfg.get("y_major_step", None)
    if y_major_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_major_step))

    ax.grid(True, axis="y", alpha=0.3)

    # 上右の枠を消す
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.canvas.draw()

    out_dir = os.path.dirname(cfg["csv_path"])
    os.makedirs(out_dir, exist_ok=True)
    ax = plt.gca()

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    if cfg.get("save_svg", False):
        out_svg = os.path.join(out_dir, cfg["out_svg_name"])
        plt.savefig(out_svg, bbox_inches="tight")
        print(f"Saved SVG: {out_svg}")

    if cfg.get("save_png", False):
        out_png = os.path.join(out_dir, cfg["out_png_name"])
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"Saved PNG: {out_png}")

    if cfg.get("show_figure", False):
        plt.show(block=False)
        plt.pause(0.1)
    else:
        plt.close(fig)


def main():
    data_list, labels_present = load_plot_data(
        csv_path=CFG["csv_path"],
        columns=CFG["columns"]
    )

    plot_box_comparison(
        data_list=data_list,
        labels_present=labels_present,
        cfg=CFG
    )

    print("Done.")


if __name__ == "__main__":
    main()