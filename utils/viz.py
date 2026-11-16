# -*- coding: utf-8 -*-
import os
import json
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

RENDER_DPI = 1000


def _save_render(fig, save_path, dpi=RENDER_DPI, save_pdf=True):
    """Save `fig` at print resolution, plus (optionally) a vector .pdf twin
    next to it so every render has a print-quality / infinitely-scalable
    companion. Set save_pdf=False (e.g. during inference) to skip the extra
    vector export and its I/O cost."""
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    if save_pdf:
        base, _ = os.path.splitext(save_path)
        fig.savefig(base + ".pdf", bbox_inches='tight', facecolor='white')

TYPE_COLORS = {
    0: "#F5C25C",  # lobby/corridor
    1: "#E5547D",  # restroom
    2: "#E8A33D",  # stairs
    3: "#7AB970",  # elevator
    4: "#6FA0D8",  # office
    5: "#A35BAA",  # mechanical
}

_FACE_LIGHT = [0.45, 1.00, 0.80, 0.55, 0.90, 0.65]


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _shade(hex_color, factor):
    r = min(255, int(int(hex_color[1:3], 16) * factor))
    g = min(255, int(int(hex_color[3:5], 16) * factor))
    b = min(255, int(int(hex_color[5:7], 16) * factor))
    return f"#{r:02x}{g:02x}{b:02x}"


def render_voxel_isometric_with_site(voxel_json_path, save_path, save_pdf=True):
    """Same as render_voxel_isometric, but also draws a faint, thin grid over
    the full ground-floor footprint (occupied + void voxels) so the site
    boundary is visible even where no room was generated."""
    vj = load_json(voxel_json_path)
    voxels = vj["voxel_node"]

    polys, colors = [], []
    xs_all, ys_all, zs_all = [], [], []

    ground_floor = min((int(v["location"][0]) for v in voxels), default=0)
    site_quads = []  # flat ground-floor cells, occupied or void

    for v in voxels:
        c0, c1, c2 = float(v["coordinate"][0]), float(v["coordinate"][1]), float(v["coordinate"][2])
        d0, d1, d2 = float(v["dimension"][0]),  float(v["dimension"][1]),  float(v["dimension"][2])
        if d0 <= 1e-6:
            d0 = 4.0

        x0, x1 = c1, c1 + d1
        y0, y1 = c2, c2 + d2
        z0, z1 = c0, c0 + d0

        xs_all += [x0, x1]; ys_all += [y0, y1]

        if int(v["location"][0]) == ground_floor:
            site_quads.append([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)])

        tcode = int(v["type"])
        if tcode < 0:
            continue
        color = TYPE_COLORS.get(tcode, "#888888")
        zs_all += [z0, z1]

        box_faces = [
            [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],
            [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],
            [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],
            [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],
            [(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)],
            [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],
        ]
        polys.extend(box_faces)
        colors.extend([_shade(color, f) for f in _FACE_LIGHT])

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(3.84, 3.84), dpi=100, facecolor='white')
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    # mplot3d normally orders whole collections by their own *average* depth,
    # which can put the (large, flat) site grid in front of the building when
    # the grid's near corner is closer to the camera than the building's
    # average depth. Disable that and use explicit zorder instead, so the
    # building is always painted on top of the ground grid.
    ax.computed_zorder = False

    if site_quads:
        grid_col = Poly3DCollection(site_quads, zsort='average', linewidths=0.3)
        grid_col.set_facecolor((0, 0, 0, 0))     # no fill -- outline only
        grid_col.set_edgecolor('#bbbbbb')         # light gray
        grid_col.set_alpha(0.6)
        grid_col.set_zorder(1)
        ax.add_collection3d(grid_col)

    if polys:
        col = Poly3DCollection(polys, zsort='average', linewidths=0.4)
        col.set_facecolor(colors)
        col.set_edgecolor('#333333')
        col.set_alpha(1.0)
        col.set_zorder(2)
        ax.add_collection3d(col)

    if xs_all:
        rx = max(xs_all) - min(xs_all)
        ry = max(ys_all) - min(ys_all)
        rz = (max(zs_all) - min(zs_all)) if zs_all else rx
        max_r = max(rx, ry, rz, 1.0)
        xmin, ymin = min(xs_all), min(ys_all)
        zmin = min(zs_all) if zs_all else 0.0
        xmax, ymax = max(xs_all), max(ys_all)
        ax.set_xlim(xmin, xmin + max_r)
        ax.set_ylim(ymin, ymin + max_r)
        ax.set_zlim(zmin, zmin + max_r)

        ox = max(rx * 0.18, 2.5)
        oy = max(ry * 0.18, 2.5)
        ax.text(xmin - ox, (ymin + ymax) / 2, zmin,
                "W", color="#CC0000", fontsize=13, fontweight="bold", ha="center", va="center")
        ax.text((xmin + xmax) / 2, ymin - oy, zmin,
                "S", color="#CC0000", fontsize=13, fontweight="bold", ha="center", va="center")

        ax.set_xlabel("X", fontsize=8, labelpad=2)
        ax.set_ylabel("Y", fontsize=8, labelpad=2)
        ax.set_zlabel("Z (floor)", fontsize=8, labelpad=2)
        ax.tick_params(labelsize=6)

    ax.view_init(elev=25, azim=215)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#cccccc')
    ax.grid(False)

    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    _save_render(fig, save_path, save_pdf=save_pdf)
    fig.clf()


def render_voxel_isometric(voxel_json_path, save_path, save_pdf=True):
    vj = load_json(voxel_json_path)
    voxels = vj["voxel_node"]

    polys, colors = [], []
    xs_all, ys_all, zs_all = [], [], []

    for v in voxels:
        tcode = int(v["type"])
        if tcode < 0:
            continue
        color = TYPE_COLORS.get(tcode, "#888888")

        c0, c1, c2 = float(v["coordinate"][0]), float(v["coordinate"][1]), float(v["coordinate"][2])
        d0, d1, d2 = float(v["dimension"][0]),  float(v["dimension"][1]),  float(v["dimension"][2])
        if d0 <= 1e-6:
            d0 = 4.0

        x0, x1 = c1, c1 + d1
        y0, y1 = c2, c2 + d2
        z0, z1 = c0, c0 + d0

        xs_all += [x0, x1]; ys_all += [y0, y1]; zs_all += [z0, z1]

        box_faces = [
            [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],
            [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],
            [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],
            [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],
            [(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)],
            [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],
        ]
        polys.extend(box_faces)
        colors.extend([_shade(color, f) for f in _FACE_LIGHT])

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(3.84, 3.84), dpi=100, facecolor='white')
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    if polys:
        col = Poly3DCollection(polys, zsort='average', linewidths=0.4)
        col.set_facecolor(colors)
        col.set_edgecolor('#333333')
        col.set_alpha(1.0)
        ax.add_collection3d(col)

    if xs_all:
        rx = max(xs_all) - min(xs_all)
        ry = max(ys_all) - min(ys_all)
        rz = max(zs_all) - min(zs_all)
        max_r = max(rx, ry, rz, 1.0)
        xmin, ymin, zmin = min(xs_all), min(ys_all), min(zs_all)
        xmax, ymax, zmax = max(xs_all), max(ys_all), max(zs_all)
        ax.set_xlim(xmin, xmin + max_r)
        ax.set_ylim(ymin, ymin + max_r)
        ax.set_zlim(zmin, zmin + max_r)

        ox = max(rx * 0.18, 2.5)
        oy = max(ry * 0.18, 2.5)
        ax.text(xmin - ox, (ymin + ymax) / 2, zmin,
                "W", color="#CC0000", fontsize=13, fontweight="bold", ha="center", va="center")
        ax.text((xmin + xmax) / 2, ymin - oy, zmin,
                "S", color="#CC0000", fontsize=13, fontweight="bold", ha="center", va="center")

        ax.set_xlabel("X", fontsize=8, labelpad=2)
        ax.set_ylabel("Y", fontsize=8, labelpad=2)
        ax.set_zlabel("Z (floor)", fontsize=8, labelpad=2)
        ax.tick_params(labelsize=6)

    ax.view_init(elev=25, azim=215)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#cccccc')
    ax.grid(False)

    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    _save_render(fig, save_path, save_pdf=save_pdf)
    fig.clf()
