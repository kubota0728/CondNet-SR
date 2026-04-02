# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:48:41 2026

@author: kubota
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation


def load_raw_data(file_path, shape, dtype=np.float32):
    raw_data = np.fromfile(file_path, dtype=dtype)
    total_elements = np.prod(shape)

    if raw_data.size != total_elements:
        raise ValueError(
            f"データサイズ不一致: {file_path}\n"
            f"read={raw_data.size}, expected={total_elements}"
        )

    # raw は (z, y, x) 順と仮定
    volume = raw_data.reshape((shape[2], shape[1], shape[0]))
    volume = np.transpose(volume, (2, 1, 0))  # -> (x, y, z)
    return volume


LABELS = {
    "csf_labels": (13,),
    "gm_labels":  (10, 91),
    "wm_labels":  (11, 92),
}


def make_6conn_structure():
    """3次元6近傍構造要素"""
    st = np.zeros((3, 3, 3), dtype=bool)
    st[1, 1, 1] = True
    st[0, 1, 1] = True
    st[2, 1, 1] = True
    st[1, 0, 1] = True
    st[1, 2, 1] = True
    st[1, 1, 0] = True
    st[1, 1, 2] = True
    return st


def find_surface_point_and_centroid_vector(volume, coil_center_xyz, labels=LABELS):
    """
    コイル位置から最短の CSF 境界点を求め、
    その点から GM+WM 重心へ向かうベクトルを返す
    """
    coil_center_xyz = np.asarray(coil_center_xyz, dtype=np.float64)

    csf_mask = np.isin(volume, labels["csf_labels"])
    gmwm_mask = np.isin(volume, labels["gm_labels"] + labels["wm_labels"])

    if not np.any(csf_mask):
        raise ValueError("CSFラベルが見つかりません。")
    if not np.any(gmwm_mask):
        raise ValueError("GM/WMラベルが見つかりません。")

    # 6近傍で GM/WM に接する CSF ボクセルを境界候補とする
    st = make_6conn_structure()
    gmwm_dil = binary_dilation(gmwm_mask, structure=st)
    boundary_csf_mask = csf_mask & gmwm_dil

    boundary_coords = np.argwhere(boundary_csf_mask)  # (N, 3), xyz順
    if boundary_coords.size == 0:
        raise ValueError("CSF-GM/WM 境界候補が見つかりません。")

    # コイル位置から最短の境界点
    diff = boundary_coords.astype(np.float64) - coil_center_xyz[None, :]
    dist2 = np.sum(diff ** 2, axis=1)
    min_idx = np.argmin(dist2)
    boundary_point = boundary_coords[min_idx].astype(np.float64)

    # GM+WM 重心
    gmwm_coords = np.argwhere(gmwm_mask).astype(np.float64)
    centroid_gmwm = gmwm_coords.mean(axis=0)

    inward_vector = centroid_gmwm - boundary_point
    inward_norm = np.linalg.norm(inward_vector)
    if inward_norm == 0:
        raise ValueError("重心と境界点が一致しているためベクトルを計算できません。")

    inward_unit_vector = inward_vector / inward_norm

    result = {
        "coil_center_xyz": coil_center_xyz,
        "boundary_point_xyz": boundary_point,
        "centroid_gmwm_xyz": centroid_gmwm,
        "inward_vector_xyz": inward_vector,
        "inward_unit_vector_xyz": inward_unit_vector,
        "distance_coil_to_boundary": float(np.sqrt(np.min(dist2))),
        "boundary_label": int(volume[tuple(np.rint(boundary_point).astype(int))]),
        "csf_mask": csf_mask,
        "gmwm_mask": gmwm_mask,
    }
    return result


def make_display_volume(volume, labels=LABELS):
    """
    表示用の簡易ラベル画像
      0: その他
      1: CSF
      2: GM/WM
    """
    disp = np.zeros_like(volume, dtype=np.uint8)
    disp[np.isin(volume, labels["csf_labels"])] = 1
    disp[np.isin(volume, labels["gm_labels"] + labels["wm_labels"])] = 2
    return disp


def plot_boundary_vector_3views(volume, result, arrow_length=40, figsize=(15, 5)):
    """
    3断面表示
    - 赤丸 : 読み込み点（脳表点）
    - 白矢印 : GM+WM重心方向ベクトル
    """
    disp = make_display_volume(volume)

    b = result["boundary_point_xyz"]
    v = result["inward_unit_vector_xyz"]

    bx, by, bz = b
    vx, vy, vz = v

    bx_i, by_i, bz_i = np.rint(b).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    cmap = plt.cm.get_cmap("viridis", 3)

    # Sagittal: x = bx, 面は (y, z)
    ax = axes[0]
    img = disp[bx_i, :, :].T
    ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=2)
    ax.scatter(by, bz, s=90, facecolors="none", edgecolors="red",
               linewidths=2, marker="o", label="Surface point")
    ax.arrow(by, bz, arrow_length * vy, arrow_length * vz,
             color="white", width=0.4, head_width=6, head_length=6,
             length_includes_head=True)
    ax.set_title(f"Sagittal (x={bx_i})")
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.legend(loc="upper right", fontsize=8)

    # Coronal: y = by, 面は (x, z)
    ax = axes[1]
    img = disp[:, by_i, :].T
    ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=2)
    ax.scatter(bx, bz, s=90, facecolors="none", edgecolors="red",
               linewidths=2, marker="o", label="Surface point")
    ax.arrow(bx, bz, arrow_length * vx, arrow_length * vz,
             color="white", width=0.4, head_width=6, head_length=6,
             length_includes_head=True)
    ax.set_title(f"Coronal (y={by_i})")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.legend(loc="upper right", fontsize=8)

    # Axial: z = bz, 面は (x, y)
    ax = axes[2]
    img = disp[:, :, bz_i].T
    ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=2)
    ax.scatter(bx, by, s=90, facecolors="none", edgecolors="red",
               linewidths=2, marker="o", label="Surface point")
    ax.arrow(bx, by, arrow_length * vx, arrow_length * vy,
             color="white", width=0.4, head_width=6, head_length=6,
             length_includes_head=True)
    ax.set_title(f"Axial (z={bz_i})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


def print_vector_result(result):
    print("=== 重心方向ベクトル ===")
    print(f"coil_center_xyz        : {result['coil_center_xyz']}")
    print(f"surface_point_xyz      : {result['boundary_point_xyz']}")
    print(f"centroid_gmwm_xyz      : {result['centroid_gmwm_xyz']}")
    print(f"distance coil->surface : {result['distance_coil_to_boundary']:.3f} voxel")
    print(f"inward_vector_xyz      : {result['inward_vector_xyz']}")
    print(f"inward_unit_vector_xyz : {result['inward_unit_vector_xyz']}")


# ===== 使用例 =====
if __name__ == "__main__":
    raw_path = r"D:\kubota\Data\Model10test\conductivity\freesurfer\IXI122_318_434_434.raw"
    shape = (318, 434, 434)  # (x, y, z)

    coil_center_xyz = (96, 162, 379)

    volume = load_raw_data(raw_path, shape)

    result = find_surface_point_and_centroid_vector(
        volume=volume,
        coil_center_xyz=coil_center_xyz,
        labels=LABELS
    )

    print_vector_result(result)
    plot_boundary_vector_3views(volume, result, arrow_length=40)