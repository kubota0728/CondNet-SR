# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 16:32:29 2026

@author: kubota
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def ensure_parent_dir(path):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def load_mean_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSVが見つかりません: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.sort_values("distance_mm").reset_index(drop=True)
    return df


def pick_existing_column(df, candidates, required=True):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(
            f"候補列が見つかりません。\n"
            f"candidates={candidates}\n"
            f"available={list(df.columns)}"
        )
    return None


def get_column_pair(df, mean_candidates, std_candidates=None, required=True):
    mean_col = pick_existing_column(df, mean_candidates, required=required)
    std_col = None
    if std_candidates is not None:
        std_col = pick_existing_column(df, std_candidates, required=False)
    return mean_col, std_col


def plot_line_with_sd(ax, x, y, sd=None, label=None, color=None, show_sd=True,
                      marker="o", linewidth=2, alpha_fill=0.18, zorder=3):
    ax.plot(
        x, y,
        marker=marker,
        linewidth=linewidth,
        label=label,
        color=color,
        zorder=zorder
    )
    if show_sd and (sd is not None):
        ax.fill_between(
            x, y - sd, y + sd,
            color=color,
            alpha=alpha_fill,
            linewidth=0,
            zorder=zorder - 1
        )


def plot_label_composition(ax, x, df, title=None, fontsize=16, show_legend=True):
    csf_col = pick_existing_column(df, ["csf_ratio_mean"])
    gm_col = pick_existing_column(df, ["gm_ratio_mean"])
    wm_col = pick_existing_column(df, ["wm_ratio_mean"])

    csf = df[csf_col].to_numpy()
    gm = df[gm_col].to_numpy()
    wm = df[wm_col].to_numpy()

    cum1 = csf
    cum2 = csf + gm
    cum3 = csf + gm + wm

    ax.fill_between(x, 0, cum1, step="mid", alpha=0.8, label="CSF")
    ax.fill_between(x, cum1, cum2, step="mid", alpha=0.8, label="GM")
    ax.fill_between(x, cum2, cum3, step="mid", alpha=0.8, label="WM")
    ax.set_ylabel("Tissue Ratio", fontsize=fontsize)
    ax.set_ylim(0, 1.0)
    ax.tick_params(labelsize=fontsize - 2)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    if show_legend:
        ax.legend(fontsize=fontsize - 2, loc="upper right")


def get_metric_columns(df):
    cols = {}

    cols["cond_mean"], cols["cond_std"] = get_column_pair(
        df,
        ["cond_mean_mean"],
        ["cond_mean_std"]
    )

    cols["ef_mean"], cols["ef_std"] = get_column_pair(
        df,
        [
            "ef_metric_value_mean",
            "ef_norm_top10_mean_mean",
            "ef_norm_p99_mean",
            "ef_summary_value_mean",
        ],
        [
            "ef_metric_value_std",
            "ef_norm_top10_mean_std",
            "ef_norm_p99_std",
            "ef_summary_value_std",
        ]
    )

    cols["grad_mean"], cols["grad_std"] = get_column_pair(
        df,
        [
            "grad_mean_mean",
            "gradient_mean_mean",
        ],
        [
            "grad_mean_std",
            "gradient_mean_std",
        ]
    )

    return cols


def plot_comparison_figure(
    df_uniform,
    df_original,
    df_mixed,
    df_proposed,
    output_fig=None,
    show_plot=True,
):
    plt.rcParams["font.family"] = "Arial"

    x = df_uniform["distance_mm"].to_numpy()

    cols_u = get_metric_columns(df_uniform)
    cols_o = get_metric_columns(df_original)
    cols_m = get_metric_columns(df_mixed)
    cols_p = get_metric_columns(df_proposed)

    # 色設定
    color_uniform = "black"
    color_original = "tab:blue"
    color_mixed = "tab:green"
    color_proposed = "tab:orange"
    color_proposed_fill = "#f6c28b"   # 薄いオレンジ

    color_csf = "#cfe8f3"   # 薄い青
    color_gm = "#bfbfbf"    # グレー
    color_wm = "#eeeeee"    # 白

    fig, axes = plt.subplots(
        4, 2,
        figsize=(13, 13),
        sharex="col",
        sharey="row",
        gridspec_kw={"height_ratios": [0.8, 1.1, 1.1, 1.1]}
    )

    # =========================
    # 1段目: Label（最上段）
    # =========================
    ax = axes[0, 0]
    csf = df_uniform["csf_ratio_mean"].to_numpy()
    gm = df_uniform["gm_ratio_mean"].to_numpy()
    wm = df_uniform["wm_ratio_mean"].to_numpy()
    cum1 = csf
    cum2 = csf + gm
    cum3 = csf + gm + wm
    ax.fill_between(x, 0, cum1, step="mid", alpha=1.0, color=color_csf, edgecolor="none")
    ax.fill_between(x, cum1, cum2, step="mid", alpha=1.0, color=color_gm, edgecolor="none")
    ax.fill_between(x, cum2, cum3, step="mid", alpha=1.0, color=color_wm, edgecolor="none")
    ax.set_ylim(0, 1.0)
    ax.tick_params(labelsize=22)

    ax = axes[0, 1]
    csf = df_proposed["csf_ratio_mean"].to_numpy()
    gm = df_proposed["gm_ratio_mean"].to_numpy()
    wm = df_proposed["wm_ratio_mean"].to_numpy()
    cum1 = csf
    cum2 = csf + gm
    cum3 = csf + gm + wm
    ax.fill_between(x, 0, cum1, step="mid", alpha=1.0, color=color_csf, edgecolor="none")
    ax.fill_between(x, cum1, cum2, step="mid", alpha=1.0, color=color_gm, edgecolor="none")
    ax.fill_between(x, cum2, cum3, step="mid", alpha=1.0, color=color_wm, edgecolor="none")
    ax.set_ylim(0, 1.0)

    # =========================
    # 2段目: Conductivity
    # =========================
    ax = axes[1, 0]
    plot_line_with_sd(
        ax, x,
        df_uniform[cols_u["cond_mean"]].to_numpy(),
        df_uniform[cols_u["cond_std"]].to_numpy() if cols_u["cond_std"] else None,
        label="Uniform",
        color=color_uniform,
        show_sd=True
    )
    ax.tick_params(labelsize=22)
    ax.legend(fontsize=22)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    ax = axes[1, 1]
    plot_line_with_sd(
        ax, x, df_original[cols_o["cond_mean"]].to_numpy(),
        sd=None, label="Original", color=color_original, show_sd=False
    )
    plot_line_with_sd(
        ax, x, df_mixed[cols_m["cond_mean"]].to_numpy(),
        sd=None, label="Mixed", color=color_mixed, show_sd=False
    )
    plot_line_with_sd(
        ax, x, df_proposed[cols_p["cond_mean"]].to_numpy(),
        sd=df_proposed[cols_p["cond_std"]].to_numpy() if cols_p["cond_std"] else None,
        label="Proposed", color=color_proposed, show_sd=True, alpha_fill=0.22
    )
    # fill_between の色を薄いオレンジにしたいので上書き
    if cols_p["cond_std"] is not None:
        y = df_proposed[cols_p["cond_mean"]].to_numpy()
        sd = df_proposed[cols_p["cond_std"]].to_numpy()
        ax.collections[-1].remove()
        ax.fill_between(x, y - sd, y + sd, color=color_proposed_fill, alpha=0.35, linewidth=0)
    ax.tick_params(labelsize=22)
    ax.legend(fontsize=22)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    # =========================
    # 3段目: E-field
    # =========================
    ax = axes[2, 0]
    plot_line_with_sd(
        ax, x,
        df_uniform[cols_u["ef_mean"]].to_numpy(),
        df_uniform[cols_u["ef_std"]].to_numpy() if cols_u["ef_std"] else None,
        label="Uniform",
        color=color_uniform,
        show_sd=True
    )
    ax.tick_params(labelsize=22)
    #ax.legend(fontsize=22)

    ax = axes[2, 1]
    plot_line_with_sd(
        ax, x, df_original[cols_o["ef_mean"]].to_numpy(),
        sd=None, label="Original", color=color_original, show_sd=False
    )
    plot_line_with_sd(
        ax, x, df_mixed[cols_m["ef_mean"]].to_numpy(),
        sd=None, label="Mixed", color=color_mixed, show_sd=False
    )
    plot_line_with_sd(
        ax, x, df_proposed[cols_p["ef_mean"]].to_numpy(),
        sd=df_proposed[cols_p["ef_std"]].to_numpy() if cols_p["ef_std"] else None,
        label="Proposed", color=color_proposed, show_sd=True, alpha_fill=0.22
    )
    if cols_p["ef_std"] is not None:
        y = df_proposed[cols_p["ef_mean"]].to_numpy()
        sd = df_proposed[cols_p["ef_std"]].to_numpy()
        ax.collections[-1].remove()
        ax.fill_between(x, y - sd, y + sd, color=color_proposed_fill, alpha=0.35, linewidth=0)
    ax.tick_params(labelsize=22)
    #ax.legend(fontsize=22)

    # =========================
    # 4段目: Gradient
    # =========================
    ax = axes[3, 0]
    plot_line_with_sd(
        ax, x,
        df_uniform[cols_u["grad_mean"]].to_numpy(),
        df_uniform[cols_u["grad_std"]].to_numpy() if cols_u["grad_std"] else None,
        label="Uniform",
        color=color_uniform,
        show_sd=True
    )
    ax.tick_params(labelsize=22)
    ax.legend(fontsize=22)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))

    ax = axes[3, 1]
    plot_line_with_sd(
        ax, x, df_original[cols_o["grad_mean"]].to_numpy(),
        sd=None, label="Original", color=color_original, show_sd=False
    )
    plot_line_with_sd(
        ax, x, df_mixed[cols_m["grad_mean"]].to_numpy(),
        sd=None, label="Mixed", color=color_mixed, show_sd=False
    )
    plot_line_with_sd(
        ax, x, df_proposed[cols_p["grad_mean"]].to_numpy(),
        sd=df_proposed[cols_p["grad_std"]].to_numpy() if cols_p["grad_std"] else None,
        label="Proposed", color=color_proposed, show_sd=True, alpha_fill=0.22
    )
    if cols_p["grad_std"] is not None:
        y = df_proposed[cols_p["grad_mean"]].to_numpy()
        sd = df_proposed[cols_p["grad_std"]].to_numpy()
        ax.collections[-1].remove()
        ax.fill_between(x, y - sd, y + sd, color=color_proposed_fill, alpha=0.35, linewidth=0)
    ax.tick_params(labelsize=22)
    #ax.legend(fontsize=22)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))

    # x軸目盛り
    for c in range(2):
        axes[3, c].set_xticks(x)

    # ===== 軸ラベルを最後に削除 =====
    for row in axes:
        for ax in row:
            ax.set_xlabel("")
            ax.set_ylabel("")

    plt.tight_layout()

    if output_fig is not None:
        ensure_parent_dir(output_fig)
        plt.savefig(output_fig, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def make_mean_csv_path(base_dir, model_name, folder_suffix):
    folder_name = f"{model_name}_{folder_suffix}"
    file_name = f"all_cases_{model_name}_local_sphere_shell_profile_grad_mean_sd.csv"
    return os.path.join(base_dir, folder_name, file_name)


def plot_from_csvs(
    base_dir,
    folder_suffix,
    uniform_name="uniform",
    original_name="M1",
    mixed_name="M2",
    proposed_name="M3",
    output_fig=None,
    show_plot=True,
):
    uniform_csv = make_mean_csv_path(base_dir, uniform_name, folder_suffix)
    original_csv = make_mean_csv_path(base_dir, original_name, folder_suffix)
    mixed_csv = make_mean_csv_path(base_dir, mixed_name, folder_suffix)
    proposed_csv = make_mean_csv_path(base_dir, proposed_name, folder_suffix)

    print("uniform_csv :", uniform_csv)
    print("original_csv:", original_csv)
    print("mixed_csv   :", mixed_csv)
    print("proposed_csv:", proposed_csv)

    df_uniform = load_mean_csv(uniform_csv)
    df_original = load_mean_csv(original_csv)
    df_mixed = load_mean_csv(mixed_csv)
    df_proposed = load_mean_csv(proposed_csv)

    plot_comparison_figure(
        df_uniform=df_uniform,
        df_original=df_original,
        df_mixed=df_mixed,
        df_proposed=df_proposed,
        output_fig=output_fig,
        show_plot=show_plot,
    )


if __name__ == "__main__":
    base_dir = r"D:\kubota\Data\Model10test\profiles_local_sphere_grad"

    folder_suffix = "sphereR50.0_top10_mean_gradTop10.0"

    output_fig = os.path.join(
        base_dir,
        f"comparison_{folder_suffix}.svg"
    )

    plot_from_csvs(
        base_dir=base_dir,
        folder_suffix=folder_suffix,
        uniform_name="uniform",
        original_name="M1",
        mixed_name="M2",
        proposed_name="M3",
        output_fig=output_fig,
        show_plot=True,
    )