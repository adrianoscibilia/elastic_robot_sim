"""A2L_005 §2 (R-5): Fig. 2, offscreen renders of the two simulated platforms.

The scene is the r7 asset (``kuka_lbr_iiwa_14_r820_table``, ``ur10_table``;
r7 has ``payload.mode: none``, so the base ``*_table.urdf``), passed through
the runner's own ``_materialized_mujoco_urdf`` (xacro expansion, inertial-frame
normalisation, mesh conversion).  The compiled model is saved as MJCF and
only presentation elements are added: a plain light background, a plain
floor, and two lights with shadows.  The posture is the clean motor-side
state ``q_motor_clean`` at t = 5 s of the first test bag in file order.

    MUJOCO_GL=egl .venv/bin/python scripts/paper/fig2_renders.py
    MUJOCO_GL=egl .venv/bin/python scripts/paper/fig2_renders.py --preview   # renders only, nothing saved to figures/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

sys.path.insert(0, str(C.ERS / "src"))
from elastic_sim.assets import load_asset_spec  # noqa: E402
from elastic_sim.generic_mujoco_runner import _materialized_mujoco_urdf  # noqa: E402

WIDTH, HEIGHT = 2000, 1500
ELEVATION = -20.0
POSTURE_T = 5.0
SCENES = {
    "iiwa_drive": dict(asset="kuka_lbr_iiwa_14_r820_table", azimuth=125.0, distance=2.6, lookat_dz=-0.12,
                       title="(a) KUKA LBR iiwa 14 R820", out="iiwa",
                       labels=["A1", "A2", "A3", "A4", "A5", "A6", "A7"]),
    "ur10": dict(asset="ur10_table", azimuth=145.0, distance=3.0, lookat_dz=-0.12,
                 title="(b) UR10 CB3", out="ur10", labels=["pan", "lift", "elbow", "w1", "w2", "w3"],
                 label_offsets={"pan": (-22, -6), "lift": (20, -4), "elbow": (-24, 8), "w1": (-24, 10),
                                "w2": (30, 6), "w3": (24, -14)}),
}
SCENES["iiwa_drive"]["label_offsets"] = {"A1": (20, 0), "A2": (22, -10), "A3": (24, 2), "A4": (22, 12),
                                         "A5": (20, 8), "A6": (-24, 14), "A7": (20, 10)}
TABLE_BELOW_BASE = 0.30


def posture(variant):
    n = len(C.JOINT_NAMES[variant])
    path = C.FILES[(variant, "test")]
    d = pd.read_parquet(path, columns=["t", "bag", "split"] + [f"q_motor_clean{j}" for j in range(n)])
    test = d[d.split == "test"]
    bag = next(iter(dict.fromkeys(test.bag)))
    b = test[test.bag == bag]
    i = int(np.argmin(np.abs(b.t.to_numpy() - b.t.iloc[0] - POSTURE_T)))
    return bag, float(b.t.iloc[i] - b.t.iloc[0]), b.iloc[i][[f"q_motor_clean{j}" for j in range(n)]].to_numpy(float)


def dress(xml_text: str) -> str:
    """Add presentation only: background, plain floor, lights with shadows, shadow resolution."""
    root = ET.fromstring(xml_text)
    visual = root.find("visual") or ET.SubElement(root, "visual")
    for tag in ("headlight", "quality", "map"):
        for e in visual.findall(tag):
            visual.remove(e)
    ET.SubElement(visual, "headlight", ambient="0.30 0.30 0.30", diffuse="0.30 0.30 0.30", specular="0.05 0.05 0.05")
    ET.SubElement(visual, "quality", shadowsize="8192", offsamples="8")
    ET.SubElement(visual, "map", znear="0.01", haze="0")
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(0, asset)
    ET.SubElement(asset, "texture", type="skybox", builtin="flat", rgb1="1 1 1", rgb2="1 1 1", width="32", height="32")
    ET.SubElement(asset, "material", name="paper_floor", rgba="0.93 0.93 0.93 1", reflectance="0", specular="0")
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="paper_floor", type="plane", size="0 0 0.05", material="paper_floor",
                  contype="0", conaffinity="0")
    ET.SubElement(world, "light", name="paper_key", pos="1.5 -1.5 4.0", dir="-0.35 0.35 -1", directional="true",
                  castshadow="true", diffuse="0.55 0.55 0.55", specular="0.1 0.1 0.1")
    ET.SubElement(world, "light", name="paper_fill", pos="-2 2 3", dir="0.5 -0.5 -1", directional="true",
                  castshadow="false", diffuse="0.2 0.2 0.2", specular="0 0 0")
    return ET.tostring(root, encoding="unicode")


def keep_visuals(urdf_path: Path):
    """Ask MuJoCo's URDF importer to keep the ``<visual>`` geoms (it discards them by default)."""
    root = ET.parse(urdf_path).getroot()
    ext = root.find("mujoco")
    if ext is None:
        ext = ET.SubElement(root, "mujoco")
    compiler = ext.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(ext, "compiler")
    compiler.set("discardvisual", "false")
    ET.ElementTree(root).write(urdf_path, encoding="utf-8", xml_declaration=True)


def coloured_visuals(mjcf: Path, sources: list[Path]) -> str:
    """Drop the collision geoms; replace each visual mesh geom by its DAE's per-material submeshes and colours.

    The runner converts every mesh to one uncoloured OBJ (``_convert_mesh_to_obj``
    concatenates the DAE's sub-geometries), which is right for dynamics and
    loses the colours.  The submeshes are written in the same file frame (their
    union must have the converted mesh's bounds, checked here) and placed with
    the original geom's pos/quat.
    """
    import trimesh
    root = ET.parse(mjcf).getroot()
    asset = root.find("asset")
    mesh_source = {}
    for m in asset.findall("mesh"):
        name = m.get("name") or Path(m.get("file")).stem
        index = int(Path(m.get("file")).stem.split("_")[-1])
        mesh_source[name] = sources[index]
    parents = {child: parent for parent in root.iter() for child in parent}
    for geom in list(root.iter("geom")):
        name = geom.get("mesh")
        if name is None:
            continue
        source = mesh_source[name]
        parent = parents[geom]
        if "collision" in source.parts:
            parent.remove(geom)
            continue
        scene = trimesh.load(source, process=False)
        parts = scene.dump(concatenate=False) if isinstance(scene, trimesh.Scene) else [scene]
        runner_mesh = trimesh.load(mjcf.parent / f"{name}.obj", process=False, force="mesh")
        union = np.vstack([p.vertices for p in parts])
        if not np.allclose([union.min(0), union.max(0)], runner_mesh.bounds, atol=1e-6):
            raise SystemExit(f"{source}: submesh frame differs from the runner's converted mesh")
        slot = list(parent).index(geom)
        parent.remove(geom)
        for k, part in enumerate(parts):
            sub = f"{name}_part{k}"
            part.export(mjcf.parent / f"{sub}.obj")
            ET.SubElement(asset, "mesh", name=sub, file=f"{sub}.obj")
            material = getattr(part.visual, "material", None)
            colour = np.asarray(material.main_color if material is not None else part.visual.main_color, float) / 255
            attrs = {k2: v for k2, v in geom.attrib.items() if k2 in ("pos", "quat", "euler", "axisangle", "xyaxes",
                                                                       "zaxis", "class")}
            attrs.update(type="mesh", mesh=sub, rgba=" ".join(f"{c:.3f}" for c in colour), contype="0",
                         conaffinity="0", group="1")
            new = ET.Element("geom", attrs)
            parent.insert(slot + k, new)
    # The table: light grey (the URDF gives a visual and a collision copy of the same box).
    for geom in root.iter("geom"):
        if geom.get("type") == "box":
            geom.set("rgba", "0.50 0.50 0.49 1")
    return ET.tostring(root, encoding="unicode")


def render(variant, azimuth=None, distance=None):
    spec = SCENES[variant]
    asset = load_asset_spec(C.ERS / "assets" / "robots" / spec["asset"] / "asset.yaml")
    bag, t, q = posture(variant)
    sources = list(asset.validate_resources())
    with _materialized_mujoco_urdf(asset) as (urdf_path, proxies):
        if proxies:
            raise SystemExit(f"{spec['asset']}: a mesh fell back to a sphere proxy; not a faithful render")
        keep_visuals(urdf_path)
        model = mujoco.MjModel.from_xml_path(str(urdf_path))
        mjcf = urdf_path.parent / "scene_paper.xml"
        mujoco.mj_saveLastXML(str(mjcf), model)
        mjcf.write_text(coloured_visuals(mjcf, sources))
        mjcf.write_text(dress(mjcf.read_text()))
        model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    for name, value in zip(asset.joint_names, q):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = value
    mujoco.mj_forward(model, data)
    # Joint anchors (world) for the annotated version.
    anchors = {}
    for name, label in zip(asset.joint_names, spec["labels"]):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        anchors[label] = data.xanchor[jid].copy()
    # Look at the centre of the arm's box (joint anchors and the last body), shifted by lookat_dz.
    points = np.vstack(list(anchors.values()) + [data.xpos[-1]])
    centre = 0.5 * (points.min(0) + points.max(0))
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = centre + [0.0, 0.0, spec["lookat_dz"]]
    cam.distance = distance or spec["distance"]
    cam.azimuth = azimuth if azimuth is not None else spec["azimuth"]
    cam.elevation = ELEVATION
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, HEIGHT)
    model.vis.global_.fovy = 30.0
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    option = mujoco.MjvOption()
    renderer.update_scene(data, camera=cam, scene_option=option)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = True
    image = renderer.render().copy()
    # Project the joint anchors to pixels for the annotated figure.
    pixels = {k: project(renderer.scene, v) for k, v in anchors.items()}
    renderer.close()
    meta = dict(asset=spec["asset"], bag=bag, t=t, q_motor=q.tolist(), azimuth=cam.azimuth,
                elevation=cam.elevation, distance=cam.distance, lookat=list(cam.lookat), fovy=30.0)
    return image, pixels, meta


def project(scene, point):
    """World point -> pixel (x, y) through the scene's first GL camera."""
    c = scene.camera[0]
    pos, fwd, up = np.asarray(c.pos), np.asarray(c.forward), np.asarray(c.up)
    right = np.cross(fwd, up)
    rel = np.asarray(point) - pos
    x, y, z = rel @ right, rel @ up, rel @ fwd
    xs, ys = x / z * c.frustum_near, y / z * c.frustum_near
    # frustum_width == 0: MuJoCo takes the horizontal extent from the viewport aspect.
    half_w = c.frustum_width or 0.5 * (c.frustum_top - c.frustum_bottom) * WIDTH / HEIGHT
    u = (xs - (c.frustum_center - half_w)) / (2 * half_w)
    v = (ys - c.frustum_bottom) / (c.frustum_top - c.frustum_bottom)
    return float(u * WIDTH), float((1 - v) * HEIGHT)


def compose(images, pixels, metas, annotated: bool, out: Path):
    plt = C.setup_matplotlib()
    w = 7.16
    crop = []
    for img, px, variant in zip(images, pixels, C.VARIANTS):
        # Content = pixels away from the background colour (the corner pixel).
        background = img[0, 0].astype(int)
        ys, xs = np.nonzero(np.abs(img.astype(int) - background).max(axis=2) > 6)
        pad = 40
        # Trim the table: keep TABLE_BELOW_BASE of the arm's pixel height below the base joint.
        base_y = px[SCENES[variant]["labels"][0]][1]
        arm_h = base_y - min(y for _, y in px.values())
        crop.append([max(ys.min() - pad, 0), min(ys.max() + pad, int(base_y + TABLE_BELOW_BASE * arm_h)),
                     max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])])
    # Same panel aspect for both: widen (or heighten) symmetrically, with background fill.
    target = max((c[1] - c[0]) / (c[3] - c[2]) for c in crop)
    panels = []
    for img, c in zip(images, crop):
        y0, y1, x0, x1 = c
        h, wd = y1 - y0, x1 - x0
        H, W = (h, int(round(h / target))) if h / wd >= target else (int(round(wd * target)), wd)
        canvas = np.empty((H, W, 3), np.uint8)
        canvas[:] = img[0, 0]
        oy, ox = (H - h) // 2, (W - wd) // 2
        canvas[oy:oy + h, ox:ox + wd] = img[y0:y1, x0:x1]
        panels.append((canvas, x0 - ox, y0 - oy))
    # Exact panel sizes in inches, so each image fills its axes and the caption sits below it.
    gap, cap = 0.06, 0.20
    pw = (w - gap) / 2
    ph = pw * target
    H = ph + cap
    fig = plt.figure(figsize=(w, H))
    axes = [fig.add_axes([k * (pw + gap) / w, cap / H, pw / w, ph / H]) for k in range(2)]
    for ax, (panel, x0, y0), px, variant in zip(axes, panels, pixels, C.VARIANTS):
        ax.imshow(panel, interpolation="lanczos")
        ax.set_axis_off()
        ax.set_xlim(-0.5, panel.shape[1] - 0.5)
        ax.set_ylim(panel.shape[0] - 0.5, -0.5)
        ax.text(0.5, -0.02, SCENES[variant]["title"], transform=ax.transAxes, ha="center", va="top", fontsize=8)
        if annotated:
            offsets = SCENES[variant].get("label_offsets", {})
            for label, (x, y) in px.items():
                dx, dy = offsets.get(label, (18, 10))
                ax.annotate(label, xy=(x - x0, y - y0), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                            ha="left" if dx >= 0 else "right", va="bottom" if dy >= 0 else "top", color="black",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.5", lw=0.4, alpha=0.9),
                            arrowprops=dict(arrowstyle="-", lw=0.5, color="0.3"))
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="render to the scratch dir only")
    parser.add_argument("--azimuth", type=float, nargs=2, default=None)
    parser.add_argument("--out", default=None, help="output directory (default figures/v2)")
    args = parser.parse_args()
    out_dir = Path(args.out) if args.out else C.FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    images, pixels, metas = [], [], {}
    from PIL import Image
    for k, variant in enumerate(C.VARIANTS):
        az = args.azimuth[k] if args.azimuth else None
        img, px, meta = render(variant, azimuth=az)
        images.append(img)
        pixels.append(px)
        metas[variant] = meta
        Image.fromarray(img).save(out_dir / f"fig_robots_sim_{SCENES[variant]['out']}.png", dpi=(300, 300))
        print(variant, json.dumps(meta))
    if args.preview:
        return
    compose(images, pixels, metas, False, out_dir / "fig_robots_sim")
    compose(images, pixels, metas, True, out_dir / "fig_robots_sim_annotated")
    (C.DATA_OUT / "paper_r5_renders_meta.json").write_text(json.dumps(metas, indent=1))


if __name__ == "__main__":
    main()
