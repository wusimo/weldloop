"""Export a MuJoCo weld run to a Blender-ready scene description.

The split is deliberate: MuJoCo owns the physics and the kinematics, Blender
owns nothing but the pixels.  This module writes, per render frame, the world
transform of every visual mesh of the robot plus the process state, and points
at the mesh files in the MuJoCo Menagerie cache.  ``scripts/blender_render.py``
reads that file inside Blender and builds the scene.

Nothing here re-simulates anything.  If the numbers in the render disagree with
the numbers in the README, this file is the bug.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from weldloop.config import WeldConfig
from weldloop.sim.mujoco_cell import CellLayout
from weldloop.sim.seam import Seam
from weldloop.viz.mujoco_render import RunRecord

__all__ = ["export_scene"]


def export_scene(
    records: dict[str, RunRecord],
    cfg: WeldConfig,
    seam: Seam,
    layout: CellLayout,
    model,
    out_dir: Path | str,
    *,
    hero: str = "adaptive",
    n_frames: int = 160,
) -> Path:
    """Write ``scene.json`` + ``frames.npz`` for the Blender renderer."""
    import mujoco

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = records[hero]
    other = records.get("baseline" if hero != "baseline" else "adaptive")

    # --- which geoms are robot meshes, and where do the files live -------
    from robot_descriptions import ur10e_mj_description

    assets = Path(ur10e_mj_description.PACKAGE_PATH) / "assets"
    mesh_geoms = []
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if model.geom_group[gid] != 2:      # visual meshes only, not collision
            continue
        mid = model.geom_dataid[gid]
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid)
        matid = model.geom_matid[gid]
        mat_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, matid)
            if matid >= 0 else "linkgray"
        )
        rgba = model.geom_rgba[gid].tolist()
        if matid >= 0:
            rgba = model.mat_rgba[matid].tolist()
        mesh_geoms.append(
            {
                "geom_id": int(gid),
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                or f"geom{gid}",
                "mesh": mesh_name,
                "file": str(assets / f"{mesh_name}.obj"),
                "material": mat_name,
                "rgba": rgba,
            }
        )

    # --- replay the recorded joint trajectory and capture world poses ----
    data = mujoco.MjData(model)
    idx = np.linspace(0, len(rec.t) - 1, n_frames).astype(int)
    G = len(mesh_geoms)
    pos = np.zeros((n_frames, G, 3))
    quat = np.zeros((n_frames, G, 4))
    torch_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torch")
    tcp_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    torch_pos = np.zeros((n_frames, 3))
    torch_quat = np.zeros((n_frames, 4))
    tcp_pos = np.zeros((n_frames, 3))
    for f, k in enumerate(idx):
        data.qpos[:6] = rec.qpos[k]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for j, g in enumerate(mesh_geoms):
            gid = g["geom_id"]
            pos[f, j] = data.geom_xpos[gid]
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, data.geom_xmat[gid].flatten())
            quat[f, j] = q
        torch_pos[f] = data.xpos[torch_bid]
        torch_quat[f] = data.xquat[torch_bid]
        tcp_pos[f] = data.site_xpos[tcp_sid]

    def series(rc: RunRecord, field: str) -> np.ndarray:
        return np.asarray(getattr(rc, field))[idx]

    frames = {
        "geom_pos": pos,
        "geom_quat": quat,
        "torch_pos": torch_pos,
        "torch_quat": torch_quat,
        "tcp_pos": tcp_pos,
        "t": series(rec, "t"),
        "s": series(rec, "s"),
        "gap": series(rec, "gap"),
        "p": series(rec, "p"),
        "w": series(rec, "w"),
        "I": series(rec, "I"),
        "smoke": series(rec, "smoke"),
        "burn_through": series(rec, "burn_through"),
        "v_travel": series(rec, "v_travel"),
        "seam_s": seam.s,
        "seam_gap": seam.gap,
    }
    if other is not None:
        o = np.linspace(0, len(other.t) - 1, n_frames).astype(int)
        frames["other_p"] = np.asarray(other.p)[o]
        frames["other_s"] = np.asarray(other.s)[o]
        frames["other_bt"] = np.asarray(other.burn_through)[o]
    np.savez_compressed(out_dir / "frames.npz", **frames)

    scene = {
        "hero": hero,
        "n_frames": int(n_frames),
        "mesh_geoms": mesh_geoms,
        "layout": {
            "table_top": layout.table_top,
            "plate_top": layout.plate_top,
            "plate_thickness": layout.plate_thickness,
            "seam_x": layout.seam_x,
            "seam_y0": layout.seam_y0,
            "seam_dir": list(layout.seam_dir),
            "plate_half_width": layout.plate_half_width,
            "plate_margin": layout.plate_margin,
            "pedestal_height": layout.pedestal_height,
        },
        "seam_length": float(seam.length),
        "plate_thickness_mm": cfg.joint.thickness * 1e3,
        "p_lo_mm": cfg.control.p_lo * 1e3,
        "p_hi_mm": cfg.control.p_hi * 1e3,
        "metrics": {
            name: {
                "burn_through_length_mm": r.metrics.burn_through_length_mm,
                "penetration_std_mm": r.metrics.penetration_std_mm,
                "in_band_pct": r.metrics.penetration_in_band_pct,
                "duration_s": r.metrics.duration_s,
            }
            for name, r in records.items()
            if r.metrics is not None
        },
    }
    (out_dir / "scene.json").write_text(json.dumps(scene, indent=2))
    return out_dir
