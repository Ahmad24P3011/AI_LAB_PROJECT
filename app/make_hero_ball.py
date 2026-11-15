"""One-off: draw the banner football as data/app/hero_ball.svg  (python app/make_hero_ball.py).

An original drawing of a classic 32-panel ball (a truncated icosahedron: 12 dark pentagons, 20 light hexagons), seen from a
tilted angle, with soft shading. It is NOT any official match ball: those carry trademarked logos, so none is used.
"""
from itertools import permutations
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "data" / "app" / "hero_ball.svg"
PHI = (1 + 5 ** 0.5) / 2
R = 100          # ball radius in the SVG
PANEL, SEAM, RING = "#240029", "#B8A9C9", "#04F5FF"


def vertices():
    """The 60 corners of a truncated icosahedron: even permutations of three base points, with every sign choice."""
    corners = set()
    for base in [(0, 1, 3 * PHI), (1, 2 + PHI, 2 * PHI), (PHI, 2, 2 * PHI + 1)]:
        for perm in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:                         # the even permutations
            for signs in np.ndindex(2, 2, 2):
                point = tuple(round(base[perm[i]] * (-1 if signs[i] else 1), 6) + 0.0 for i in range(3))
                corners.add(point)
    return np.array(sorted(corners))


def rotation(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    X = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Z @ Y @ X


def to_screen(points, rot):
    """Rotate unit vectors, project straight down the z axis, and pull hidden points (z < 0) onto the rim so shapes stay closed."""
    p = points @ rot.T
    xy = p[:, :2].copy()
    hidden = p[:, 2] < 0
    norm = np.linalg.norm(xy[hidden], axis=1, keepdims=True)
    xy[hidden] = xy[hidden] / np.maximum(norm, 1e-9)
    return xy * R, p[:, 2]


def main():
    v = vertices()
    assert len(v) == 60
    unit = v / np.linalg.norm(v, axis=1, keepdims=True)
    dist = np.linalg.norm(v[:, None] - v[None], axis=2)
    edges = [(i, j) for i in range(60) for j in range(i + 1, 60) if abs(dist[i, j] - 2) < 1e-3]
    assert len(edges) == 90
    # a pentagon is a 5-cycle of corners joined by seams (hexagons are 6-cycles), so every 5-cycle found is a pentagon
    adjacency = {i: {j for a, b in edges for i2, j in ((a, b), (b, a)) if i2 == i} for i in range(60)}
    faces = set()
    for a in range(60):
        for b in adjacency[a]:
            for c in adjacency[b] - {a}:
                for d in adjacency[c] - {b, a}:
                    for e in adjacency[d] - {c, b, a}:
                        if a in adjacency[e]:
                            faces.add(frozenset((a, b, c, d, e)))
    pent = [list(f) for f in faces]
    assert len(pent) == 12, len(pent)

    rot = rotation(np.radians(-28), np.radians(32), np.radians(8))
    xy, z = to_screen(unit, rot)

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-R - 6} {-R - 6} {2 * R + 12} {2 * R + 12}" width="{2 * R + 12}" height="{2 * R + 12}">',
           "<!-- original, generated drawing of a classic football (app/make_hero_ball.py); not an official match ball -->",
           '<defs>',
           f'<clipPath id="c"><circle cx="0" cy="0" r="{R}"/></clipPath>',
           '<radialGradient id="shade" cx="35%" cy="30%" r="80%"><stop offset="0" stop-color="#FFFFFF"/><stop offset="0.55" stop-color="#F1ECF6"/>'
           '<stop offset="1" stop-color="#B9A8CC"/></radialGradient>',
           '<radialGradient id="light" cx="35%" cy="30%" r="55%"><stop offset="0" stop-color="#FFFFFF" stop-opacity="0.55"/>'
           '<stop offset="1" stop-color="#FFFFFF" stop-opacity="0"/></radialGradient>',
           '<radialGradient id="edge" cx="50%" cy="50%" r="50%"><stop offset="0.72" stop-color="#240029" stop-opacity="0"/>'
           '<stop offset="1" stop-color="#240029" stop-opacity="0.45"/></radialGradient>',
           '</defs>',
           f'<circle cx="0" cy="0" r="{R}" fill="url(#shade)"/>',
           '<g clip-path="url(#c)">']
    for i, j in edges:                                                           # seams between panels
        if z[i] > -0.05 or z[j] > -0.05:
            svg.append(f'<line x1="{xy[i][0]:.2f}" y1="{xy[i][1]:.2f}" x2="{xy[j][0]:.2f}" y2="{xy[j][1]:.2f}" stroke="{SEAM}" stroke-width="1.6" stroke-linecap="round"/>')
    for face in pent:                                                            # the 12 dark pentagons, corners in angular order
        pts = unit[face]
        centre = pts.mean(axis=0)
        if (centre @ rot.T)[2] < -0.45:
            continue
        axis = centre / np.linalg.norm(centre)
        u = np.cross(axis, [0.3, 0.9, 0.1]); u /= np.linalg.norm(u)
        w = np.cross(axis, u)
        order = np.argsort(np.arctan2(pts @ w, pts @ u))
        ring = [face[k] for k in order]
        coords = " ".join(f"{xy[k][0]:.2f},{xy[k][1]:.2f}" for k in ring)
        svg.append(f'<polygon points="{coords}" fill="{PANEL}" stroke="{PANEL}" stroke-width="1.6" stroke-linejoin="round"/>')
    svg += ['</g>', f'<circle cx="0" cy="0" r="{R}" fill="url(#light)"/>', f'<circle cx="0" cy="0" r="{R}" fill="url(#edge)"/>',
            f'<circle cx="0" cy="0" r="{R}" fill="none" stroke="{RING}" stroke-width="3"/>', '</svg>']
    OUT.write_text("\n".join(svg), encoding="utf-8")
    print(f"wrote {OUT}: {len(edges)} seams, {len(pent)} pentagons")


if __name__ == "__main__":
    main()
