#!/usr/bin/env python3
"""Photoreal render of a weldloop run, executed inside Blender.

    blender -b -noaudio -P scripts/blender_render.py -- \
        --scene out/blender_scene --out out/blender --frames 120

Blender does no simulation here.  It reads ``scene.json`` + ``frames.npz``
written by ``weldloop.viz.blender_export`` — world transforms for every visual
mesh of the UR10e, and the process state per frame — and turns them into
pixels.  Every number that appears on screen came from the weld simulation.

What the extra fidelity buys, beyond looking better:

* the **arc is a real light source**.  It is the brightest thing in the scene
  by three orders of magnitude, it casts the shadows, and it scatters through
  the fume volume.  That is the physical reason a visible-light camera is
  useless during welding, rendered rather than asserted.
* the **fume is a volume**, so it occludes the pool the way it does in a real
  cell, instead of being a grey overlay.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Quaternion, Vector


# --------------------------------------------------------------------------
def parse_args() -> dict:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = {
        "scene": "out/blender_scene", "out": "out/blender", "frames": 0,
        "res": 960, "samples": 48, "start": 0, "end": 0, "still": -1,
        "view": "wide", "bead_strength": 7.0, "arc_energy": 5.5,
        "exposure": 99.0, "plate_rough": 0.45, "plate_metal": 0.88,
        "fume_density": 70.0, "arc_emission": 2600.0, "cool_len": 0.026,
        "key": 90.0, "rim": 42.0,
    }
    i = 0
    while i < len(argv):
        key = argv[i].lstrip("-").replace("-", "_")
        if key in out:
            out[key] = type(out[key])(argv[i + 1]) if not isinstance(out[key], str) \
                else argv[i + 1]
            i += 2
        else:
            i += 1
    return out


def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def mat_principled(name, color, metallic=0.0, roughness=0.5, emission=None,
                   emission_strength=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if emission is not None:
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
            bsdf.inputs["Emission Strength"].default_value = emission_strength
    return m


UR_MATERIALS = {
    "black": ((0.035, 0.035, 0.038), 0.25, 0.38),
    "jointgray": ((0.115, 0.118, 0.125), 0.55, 0.35),
    "linkgray": ((0.62, 0.63, 0.65), 0.75, 0.22),
    "urblue": ((0.055, 0.20, 0.38), 0.30, 0.30),
    "torch": ((0.115, 0.118, 0.125), 0.60, 0.30),
    "copper": ((0.72, 0.41, 0.18), 0.95, 0.26),
}


def box(name, centre, size, material):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=centre)
    ob = bpy.context.object
    ob.name = name
    ob.scale = size
    ob.data.materials.append(material)
    return ob


def build(args) -> dict:
    scene_dir = Path(args["scene"])
    meta = json.loads((scene_dir / "scene.json").read_text())
    fr = np.load(scene_dir / "frames.npz")
    L = meta["layout"]
    seam_len = meta["seam_length"]
    n_frames = args["frames"] or int(meta["n_frames"])
    n_frames = min(n_frames, len(fr["t"]))

    clear_scene()
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = args["samples"]
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.02
    sc.cycles.use_denoising = True
    try:
        sc.cycles.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        pass
    sc.cycles.volume_step_rate = 2.0
    sc.cycles.volume_max_steps = 128
    sc.cycles.max_bounces = 6
    sc.cycles.transmission_bounces = 2
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.render.resolution_x = args["res"]
    sc.render.resolution_y = int(args["res"] * 9 / 16)
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Punchy"
    sc.view_settings.exposure = (
        args["exposure"] if args["exposure"] < 90.0
        else (-1.6 if args["view"] == "close" else -0.4)
    )
    sc.frame_start = 0
    sc.frame_end = n_frames - 1

    # --- world: dim workshop ------------------------------------------
    world = bpy.data.worlds.new("w")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.012, 0.014, 0.018, 1.0)
    bg.inputs[1].default_value = 1.0

    mats = {k: mat_principled(k, c, m, r) for k, (c, m, r) in UR_MATERIALS.items()}
    mats["steel"] = mat_principled(
        "steel", (0.085, 0.093, 0.104), args["plate_metal"], args["plate_rough"]
    )
    mats["bench"] = mat_principled("bench", (0.055, 0.060, 0.070), 0.60, 0.55)
    mats["floor"] = mat_principled("floor", (0.028, 0.030, 0.034), 0.10, 0.85)
    mats["clampm"] = mat_principled("clampm", (0.10, 0.105, 0.115), 0.85, 0.30)

    # --- room ----------------------------------------------------------
    bpy.ops.mesh.primitive_plane_add(size=14.0, location=(0, 0, 0))
    bpy.context.object.name = "floor"
    bpy.context.object.data.materials.append(mats["floor"])

    # --- bench and pedestal --------------------------------------------
    seam_x, seam_y0 = L["seam_x"], L["seam_y0"]
    top = L["table_top"]
    bx0, bx1 = seam_x - 0.26, seam_x + 0.30
    by0, by1 = seam_y0 - 0.22, seam_y0 + 0.42
    box("bench", ((bx0 + bx1) / 2, (by0 + by1) / 2, top - 0.018),
        (bx1 - bx0, by1 - by0, 0.036), mats["bench"])
    for lx in (bx0 + 0.05, bx1 - 0.05):
        for ly in (by0 + 0.05, by1 - 0.05):
            box(f"leg{lx:.2f}{ly:.2f}", (lx, ly, (top - 0.036) / 2),
                (0.036, 0.036, top - 0.036), mats["bench"])
    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.13, depth=L["pedestal_height"],
        location=(0, 0, L["pedestal_height"] / 2))
    bpy.context.object.name = "pedestal"
    bpy.context.object.data.materials.append(mats["bench"])

    # --- workpiece ------------------------------------------------------
    gap = float(np.mean(fr["seam_gap"]))
    half_w = L["plate_half_width"]
    thick = L["plate_thickness"]
    for sign, tag in ((+1.0, "a"), (-1.0, "b")):
        off = sign * (gap / 2.0 + half_w / 2.0)
        box(f"plate_{tag}",
            (seam_x - off, seam_y0 + seam_len / 2.0, top + thick / 2.0),
            (half_w, seam_len + 2 * L["plate_margin"], thick), mats["steel"])
    for frac in (0.12, 0.88):
        for sign in (+1.0, -1.0):
            box(f"clamp{frac}{sign}",
                (seam_x - sign * 0.052, seam_y0 + frac * seam_len, top + thick + 0.010),
                (0.020, 0.020, 0.020), mats["clampm"])

    # --- robot meshes ---------------------------------------------------
    objs = []
    for g in meta["mesh_geoms"]:
        bpy.ops.wm.obj_import(filepath=g["file"], forward_axis="Y", up_axis="Z")
        imported = [o for o in bpy.context.selected_objects]
        bpy.ops.object.join() if len(imported) > 1 else None
        ob = bpy.context.object
        ob.name = g["name"]
        ob.data.materials.clear()
        ob.data.materials.append(mats.get(g["material"], mats["linkgray"]))
        bpy.ops.object.shade_smooth()
        ob.rotation_mode = "QUATERNION"
        objs.append(ob)

    # --- torch (rebuilt as primitives; the MJCF torch is primitives too) --
    torch_parts = []
    # offsets are in the TORCH BODY frame, matching the MJCF exactly
    for nm, r, h, z, mat in (
        ("torch_body", 0.026, 0.110, 0.055, "torch"),
        ("torch_neck", 0.014, 0.090, 0.145, "torch"),
        ("torch_nozzle", 0.011, 0.044, 0.198, "copper"),
    ):
        bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=h, location=(0, 0, 0))
        ob = bpy.context.object
        ob.name = nm
        ob.data.materials.append(mats[mat])
        ob.rotation_mode = "QUATERNION"
        bpy.ops.object.shade_smooth()
        torch_parts.append((ob, z))

    # --- weld bead: one strip, shaded by how far the torch has passed ----
    bead = _make_bead(seam_x, seam_y0, top + thick, seam_len, gap,
                      bead_strength=args["bead_strength"],
                      cool_len=args["cool_len"])

    # --- arc: emissive sphere + a very bright light ----------------------
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.0035, location=(0, 0, 0))
    arc_ball = bpy.context.object
    arc_ball.name = "arc"
    arc_mat = bpy.data.materials.new("arcmat")
    arc_mat.use_nodes = True
    nt = arc_mat.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs[0].default_value = (1.0, 0.93, 0.80, 1.0)
    em.inputs[1].default_value = args["arc_emission"]
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    arc_ball.data.materials.append(arc_mat)

    arc_light_data = bpy.data.lights.new("arclight", type="POINT")
    arc_light_data.energy = args["arc_energy"]
    arc_light_data.color = (1.0, 0.72, 0.42)
    arc_light_data.shadow_soft_size = 0.0035
    arc_light = bpy.data.objects.new("arclight", arc_light_data)
    sc.collection.objects.link(arc_light)

    # --- fill lights ------------------------------------------------------
    for nm, loc, energy, size, col in (
        ("key", (1.5, -1.2, 2.3), args["key"], 1.1, (1.0, 0.97, 0.92)),
        ("rim", (-1.4, 1.3, 2.0), args["rim"], 1.4, (0.72, 0.80, 1.0)),
    ):
        ld = bpy.data.lights.new(nm, type="AREA")
        ld.energy = energy
        ld.size = size
        ld.color = col
        ob = bpy.data.objects.new(nm, ld)
        ob.location = loc
        direction = Vector((seam_x, seam_y0 + seam_len / 2, top)) - Vector(loc)
        ob.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        sc.collection.objects.link(ob)

    # --- fume volume ------------------------------------------------------
    fume = _make_fume(seam_x, seam_y0, top + thick, seam_len)

    # --- camera -----------------------------------------------------------
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 62.0 if args["view"] == "close" else 40.0
    cam_data.clip_start = 0.005
    cam = bpy.data.objects.new("cam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam

    return {
        "meta": meta, "fr": fr, "n_frames": n_frames, "objs": objs,
        "torch_parts": torch_parts, "arc_ball": arc_ball, "arc_light": arc_light,
        "bead": bead, "fume": fume, "cam": cam, "layout": L, "seam_len": seam_len,
    }


def _make_bead(seam_x, seam_y0, plate_top, seam_len, gap, bead_strength=7.0,
               cool_len=0.026):
    """A single strip along the seam whose shader knows where the torch is."""
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    ob = bpy.context.object
    ob.name = "weld_bead"
    # primitive_cube_add(size=1.0) spans +/-0.5, so scale is the FULL extent.
    # Getting this wrong made the bead cover half the seam and put the shader's
    # position mapping out by a factor of two.
    ob.scale = (0.019, seam_len, 0.0044)
    ob.location = (seam_x, seam_y0 + seam_len / 2.0, plate_top + 0.0013)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    m = bpy.data.materials.new("beadmat")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    texco = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(texco.outputs["Object"], sep.inputs[0])

    torch_s = nt.nodes.new("ShaderNodeValue")
    torch_s.name = "torch_s"
    torch_s.label = "torch_s"

    # After transform_apply the object coordinates are in METRES, spanning
    # +/- seam_len/2 along Y, so no rescaling -- only the shift to 0..seam_len.
    shift = nt.nodes.new("ShaderNodeMath")
    shift.operation = "ADD"
    shift.inputs[1].default_value = seam_len / 2.0
    nt.links.new(sep.outputs["Y"], shift.inputs[0])

    behind = nt.nodes.new("ShaderNodeMath")      # torch_s - pos_along
    behind.operation = "SUBTRACT"
    nt.links.new(torch_s.outputs[0], behind.inputs[0])
    nt.links.new(shift.outputs[0], behind.inputs[1])

    laid = nt.nodes.new("ShaderNodeMath")
    laid.operation = "GREATER_THAN"
    laid.inputs[1].default_value = 0.0
    nt.links.new(behind.outputs[0], laid.inputs[0])

    hot = nt.nodes.new("ShaderNodeMath")
    hot.operation = "DIVIDE"
    hot.inputs[1].default_value = cool_len       # metres of visible cooling
    nt.links.new(behind.outputs[0], hot.inputs[0])
    hot_inv = nt.nodes.new("ShaderNodeMath")
    hot_inv.operation = "SUBTRACT"
    hot_inv.inputs[0].default_value = 1.0
    nt.links.new(hot.outputs[0], hot_inv.inputs[1])
    hot_cl = nt.nodes.new("ShaderNodeClamp")
    nt.links.new(hot_inv.outputs[0], hot_cl.inputs[0])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0.10, 0.045, 0.020, 1)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (1.0, 0.26, 0.030, 1)
    nt.links.new(hot_cl.outputs[0], ramp.inputs[0])

    strength = nt.nodes.new("ShaderNodeMath")
    strength.operation = "POWER"
    strength.inputs[1].default_value = 1.9
    nt.links.new(hot_cl.outputs[0], strength.inputs[0])
    strength2 = nt.nodes.new("ShaderNodeMath")
    strength2.operation = "MULTIPLY"
    strength2.inputs[1].default_value = bead_strength
    nt.links.new(strength.outputs[0], strength2.inputs[0])

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.075, 0.062, 0.055, 1)
    bsdf.inputs["Metallic"].default_value = 0.55
    bsdf.inputs["Roughness"].default_value = 0.62
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Emission Color"])
    nt.links.new(strength2.outputs[0], bsdf.inputs["Emission Strength"])

    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(laid.outputs[0], mix.inputs[0])
    nt.links.new(transp.outputs[0], mix.inputs[1])
    nt.links.new(bsdf.outputs[0], mix.inputs[2])
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    ob.data.materials.append(m)
    return ob


def _make_fume(seam_x, seam_y0, plate_top, seam_len):
    """A volume whose density sits behind the arc and rises."""
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    ob = bpy.context.object
    ob.name = "fume"
    ob.scale = (0.16, seam_len + 0.10, 0.34)   # full extents; half is +/-0.17 in Z
    ob.location = (seam_x, seam_y0 + seam_len / 2.0, plate_top + 0.17)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    m = bpy.data.materials.new("fumemat")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    texco = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.name = "fume_map"
    nt.links.new(texco.outputs["Object"], mapping.inputs["Vector"])
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 17.0
    noise.inputs["Detail"].default_value = 6.0
    noise.inputs["Roughness"].default_value = 0.62
    nt.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

    # mask: only just behind the torch, thinning upward
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(texco.outputs["Object"], sep.inputs[0])
    torch_v = nt.nodes.new("ShaderNodeValue")
    torch_v.name = "fume_s"
    torch_v.label = "fume_s"
    dy = nt.nodes.new("ShaderNodeMath")
    dy.operation = "SUBTRACT"
    nt.links.new(sep.outputs["Y"], dy.inputs[0])
    nt.links.new(torch_v.outputs[0], dy.inputs[1])
    dy2 = nt.nodes.new("ShaderNodeMath")
    dy2.operation = "MULTIPLY"
    nt.links.new(dy.outputs[0], dy2.inputs[0])
    nt.links.new(dy.outputs[0], dy2.inputs[1])
    band = nt.nodes.new("ShaderNodeMath")
    band.operation = "MULTIPLY"
    band.inputs[1].default_value = -1.0 / (0.050 ** 2)   # metres: 50 mm plume
    nt.links.new(dy2.outputs[0], band.inputs[0])
    band_e = nt.nodes.new("ShaderNodeMath")
    band_e.operation = "POWER"
    band_e.inputs[0].default_value = 2.71828
    nt.links.new(band.outputs[0], band_e.inputs[1])

    # thin out with height: object Z spans +/- 0.17 m, dense at the bottom
    height = nt.nodes.new("ShaderNodeMath")
    height.operation = "SUBTRACT"
    height.inputs[0].default_value = 0.17
    nt.links.new(sep.outputs["Z"], height.inputs[1])
    height_n = nt.nodes.new("ShaderNodeMath")
    height_n.operation = "DIVIDE"
    height_n.inputs[1].default_value = 0.34
    nt.links.new(height.outputs[0], height_n.inputs[0])
    height_c = nt.nodes.new("ShaderNodeClamp")
    nt.links.new(height_n.outputs[0], height_c.inputs[0])

    dens_level = nt.nodes.new("ShaderNodeValue")
    dens_level.name = "fume_density"
    dens_level.label = "fume_density"

    m1 = nt.nodes.new("ShaderNodeMath"); m1.operation = "MULTIPLY"
    nt.links.new(noise.outputs["Fac"], m1.inputs[0])
    nt.links.new(band_e.outputs[0], m1.inputs[1])
    m2 = nt.nodes.new("ShaderNodeMath"); m2.operation = "MULTIPLY"
    nt.links.new(m1.outputs[0], m2.inputs[0])
    nt.links.new(height_c.outputs[0], m2.inputs[1])
    m3 = nt.nodes.new("ShaderNodeMath"); m3.operation = "MULTIPLY"
    nt.links.new(m2.outputs[0], m3.inputs[0])
    nt.links.new(dens_level.outputs[0], m3.inputs[1])

    scatter = nt.nodes.new("ShaderNodeVolumeScatter")
    scatter.inputs["Color"].default_value = (0.82, 0.80, 0.76, 1.0)
    scatter.inputs["Anisotropy"].default_value = 0.45
    nt.links.new(m3.outputs[0], scatter.inputs["Density"])
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(scatter.outputs[0], out.inputs["Volume"])
    ob.data.materials.append(m)
    return ob


# --------------------------------------------------------------------------
def animate(state, args) -> None:
    # Blender 5 replaced Action.fcurves with slotted actions; setting the
    # default interpolation before inserting keys is version-proof.
    try:
        bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    except Exception:
        pass
    fr = state["fr"]
    n = state["n_frames"]
    L = state["layout"]
    seam_len = state["seam_len"]
    seam_x, seam_y0 = L["seam_x"], L["seam_y0"]
    plate_top = L["table_top"] + L["plate_thickness"]

    bead_val = state["bead"].data.materials[0].node_tree.nodes["torch_s"]
    fume_s = state["fume"].data.materials[0].node_tree.nodes["fume_s"]
    fume_d = state["fume"].data.materials[0].node_tree.nodes["fume_density"]
    fume_map = state["fume"].data.materials[0].node_tree.nodes["fume_map"]


    for f in range(n):
        bpy.context.scene.frame_set(f)
        for j, ob in enumerate(state["objs"]):
            ob.location = Vector(fr["geom_pos"][f, j].tolist())
            ob.rotation_quaternion = Quaternion(fr["geom_quat"][f, j].tolist())
            ob.keyframe_insert("location", frame=f)
            ob.keyframe_insert("rotation_quaternion", frame=f)

        # the torch is placed from its own MuJoCo body pose, so it sits exactly
        # where the simulated arm is holding it
        s_now = float(fr["s"][f])
        tq = Quaternion(fr["torch_quat"][f].tolist())
        tp = Vector(fr["torch_pos"][f].tolist())
        for ob, z in state["torch_parts"]:
            ob.location = tp + tq @ Vector((0.0, 0.0, z))
            ob.rotation_quaternion = tq
            ob.keyframe_insert("location", frame=f)
            ob.keyframe_insert("rotation_quaternion", frame=f)

        state["arc_ball"].location = Vector(fr["tcp_pos"][f].tolist())
        state["arc_ball"].keyframe_insert("location", frame=f)
        state["arc_light"].location = Vector(fr["tcp_pos"][f].tolist()) + Vector(
            (0.0, 0.0, 0.016)
        )
        state["arc_light"].keyframe_insert("location", frame=f)
        duty = float(np.clip(fr["I"][f] / 240.0, 0.35, 1.5))
        state["arc_light"].data.energy = args["arc_energy"] * duty
        state["arc_light"].data.keyframe_insert("energy", frame=f)

        bead_val.outputs[0].default_value = s_now
        bead_val.outputs[0].keyframe_insert("default_value", frame=f)

        # fume: object coords run -0.5..0.5 across the volume
        fume_s.outputs[0].default_value = s_now - seam_len / 2.0
        fume_s.outputs[0].keyframe_insert("default_value", frame=f)
        fume_d.outputs[0].default_value = args["fume_density"] * float(fr["smoke"][f])
        fume_d.outputs[0].keyframe_insert("default_value", frame=f)
        fume_map.inputs["Location"].default_value = (0.0, 0.0, -0.22 * f / max(n, 1))
        fume_map.inputs["Location"].keyframe_insert("default_value", frame=f)

        # camera: slow arc around the workpiece
        u = f / max(n - 1, 1)
        cam = state["cam"]
        if args["view"] == "close":
            # ride just behind and above the torch, looking at the pool
            tcp = Vector(fr["tcp_pos"][f].tolist())
            ang = math.radians(-58.0 + 8.0 * math.sin(2 * math.pi * u))
            radius = 0.28
            target = tcp + Vector((0.0, 0.042, -0.004))
            cam.location = tcp + Vector(
                (radius * math.cos(ang), radius * math.sin(ang) - 0.10, 0.086)
            )
        else:
            # a fixed operator viewpoint with a slow drift, rather than an orbit
            # that swings behind the workpiece
            # lower and further back, so the seam and the bead stay visible
            # under the arm rather than behind it
            ang = math.radians(-62.0 + 10.0 * math.sin(2 * math.pi * u))
            radius = 1.34
            target = Vector((seam_x - 0.04, seam_y0 + seam_len * 0.5, plate_top + 0.045))
            cam.location = target + Vector(
                (radius * math.cos(ang), radius * math.sin(ang), 0.30)
            )
        cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_euler", frame=f)




def main() -> None:
    args = parse_args()
    state = build(args)
    animate(state, args)
    out = Path(args["out"])
    out.mkdir(parents=True, exist_ok=True)
    sc = bpy.context.scene
    sc.render.image_settings.file_format = "PNG"
    if args["still"] >= 0:
        sc.frame_set(args["still"])
        sc.render.filepath = str(out / f"still_{args['still']:04d}.png")
        bpy.ops.render.render(write_still=True)
        print("STILL", sc.render.filepath)
        return
    start = args["start"]
    end = args["end"] or state["n_frames"] - 1
    sc.render.filepath = str(out / "frame_")
    sc.frame_start, sc.frame_end = start, end
    bpy.ops.render.render(animation=True)
    print("DONE frames", start, end)


main()
