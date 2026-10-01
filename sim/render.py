"""Controlled synthetic indoor sequences: texture level x dynamic level along identical trajectories.

CPU ray caster (Open3D RaycastingScene / Embree), Lambertian shading with point lights, 2x2 supersampling,
procedural tileable grayscale textures. Outputs per sequence (TUM-like layout):
  left/%06d.png right/%06d.png (8-bit gray), depth/%06d.png (uint16, z*5000, left camera),
  mask/%06d.png (255 = dynamic agent), times.txt, groundtruth.txt (TUM: t tx ty tz qx qy qz qw, left camera),
  frame_stats.csv (texture metrics on static pixels), settings.yaml (ORB-SLAM2).
Usage: python render.py --scene 1 --tex 0 --dyn 2 --out DIR [--frames 450]
"""
import os, sys, math, argparse, json, time
import numpy as np
import cv2
import open3d as o3d

W, H = 640, 480
FX = FY = 400.0
CX, CY = (W - 1) / 2.0, (H - 1) / 2.0
BASELINE = 0.10
FPS = 30.0
NFRAMES = 450
BETAS = [0.0, 0.4, 0.7, 1.0]          # texture levels L0..L3 (fixed after calibration pilot)
EXPOSURE = 0.65
NOISE_SIGMA = 1.0                      # additive Gaussian image noise [gray levels], same realisation in all conditions
FAST_T = 20                            # texture metric: FAST-9 threshold
GRAD_BINS, GRAD_MAX = 64, 512.0        # texture metric: Sobel-magnitude histogram

# ----------------------------------------------------------------------------- textures
def _value_noise(n, cells, rng):
    """Tileable value noise on an n x n texture with `cells` random lattice cells per side."""
    g = rng.random((cells, cells))
    x = np.arange(n) * cells / n
    i0 = np.floor(x).astype(int); f = x - i0; i1 = (i0 + 1) % cells
    f = f * f * (3 - 2 * f)
    a = g[i0][:, i0]; b = g[i0][:, i1]; c = g[i1][:, i0]; d = g[i1][:, i1]
    fy, fx = f[:, None], f[None, :]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy

def fractal(n, rng, base=4, octaves=5, gain=0.55):
    out = np.zeros((n, n)); amp = 1.0; tot = 0
    for o in range(octaves):
        out += amp * _value_noise(n, base * 2 ** o, rng); tot += amp; amp *= gain
    return out / tot

def make_texture(kind, seed, n=512):
    rng = np.random.default_rng(seed)
    u = np.arange(n) / n
    X, Y = np.meshgrid(u, u)
    if kind == 'plaster':
        t = 0.72 + 0.22 * (fractal(n, rng, 4, 6) - 0.5) + 0.05 * (rng.random((n, n)) - 0.5)
    elif kind == 'wood':
        warp = fractal(n, rng, 2, 4)
        t = 0.45 + 0.12 * np.sin(2 * np.pi * (6 * X + 3.0 * warp)) + 0.08 * (fractal(n, rng, 16, 3) - 0.5)
        t = t * (1 - 0.35 * (np.abs(((Y * 4) % 1) - 0.5) > 0.485))   # plank seams
    elif kind == 'tiles':
        k = 4; tile = rng.random((k, k)) * 0.18
        ti = (np.floor(Y * k).astype(int), np.floor(X * k).astype(int))
        t = 0.62 + tile[ti] + 0.10 * (fractal(n, rng, 8, 4) - 0.5)
        grout = (np.abs(((X * k) % 1) - 0.5) > 0.47) | (np.abs(((Y * k) % 1) - 0.5) > 0.47)
        t[grout] = 0.30
    elif kind == 'fabric':
        t = 0.5 + 0.10 * np.sin(2 * np.pi * 48 * X) * np.sin(2 * np.pi * 48 * Y) + 0.18 * (fractal(n, rng, 6, 5) - 0.5)
    elif kind == 'carpet':
        t = 0.40 + 0.40 * (fractal(n, rng, 8, 5, 0.7) - 0.5)
    elif kind == 'brick':
        rows = 8; r = np.floor(Y * rows); off = (r % 2) * 0.5
        c = np.floor(X * 4 + off)
        shade = rng.random((rows + 1, 10))[r.astype(int), (c.astype(int) % 10)]
        t = 0.42 + 0.20 * shade + 0.12 * (fractal(n, rng, 16, 3) - 0.5)
        mortar = (np.abs(((Y * rows) % 1) - 0.5) > 0.44) | (np.abs(((X * 4 + off) % 1) - 0.5) > 0.47)
        t[mortar] = 0.78
    elif kind == 'marble':
        t = 0.70 + 0.18 * np.sin(2 * np.pi * (3 * X + 2 * Y + 4 * fractal(n, rng, 3, 5)))
    elif kind == 'metal':
        t = 0.55 + 0.04 * (fractal(n, rng, 32, 2) - 0.5)
    elif kind in ('poster', 'screen', 'clothes'):
        t = np.full((n, n), rng.uniform(0.3, 0.8))
        m = 40 if kind != 'clothes' else 25
        for _ in range(m):
            v = rng.uniform(0.05, 0.95)
            if rng.random() < 0.5:
                x0, y0 = rng.random(2); w, h = rng.uniform(0.04, 0.3, 2)
                t[(X > x0) & (X < x0 + w) & (Y > y0) & (Y < y0 + h)] = v
            else:
                x0, y0 = rng.random(2); r = rng.uniform(0.03, 0.15)
                t[(X - x0) ** 2 + (Y - y0) ** 2 < r * r] = v
        t += 0.06 * (fractal(n, rng, 16, 3) - 0.5)
    elif kind == 'books':
        t = np.zeros((n, n)); x = 0.0
        while x < 1:
            w = rng.uniform(0.015, 0.05); v = rng.uniform(0.15, 0.9)
            t[:, (u >= x) & (u < x + w)] = v; x += w
        t[(Y % 0.25) > 0.23] = 0.35
        t += 0.05 * (fractal(n, rng, 16, 3) - 0.5)
    else:
        raise ValueError(kind)
    tp = np.pad(np.clip(t, 0.02, 0.98).astype(np.float32), 8, mode='wrap')   # tileable blur
    return np.ascontiguousarray(cv2.GaussianBlur(tp, (0, 0), 0.8)[8:-8, 8:-8])

# ----------------------------------------------------------------------------- geometry
class MeshBuilder:
    """Accumulates triangles with per-vertex UVs and per-triangle material ids."""
    def __init__(self):
        self.V = []; self.T = []; self.UV = []; self.M = []; self.nv = 0
    def _add(self, pts, tris, uvs, mat):
        self.V.append(np.asarray(pts, float)); b = self.nv; self.nv += len(pts)
        for tr, uv in zip(tris, uvs):
            self.T.append([b + tr[0], b + tr[1], b + tr[2]]); self.UV.append(uv); self.M.append(mat)
    def quad(self, p0, p1, p2, p3, mat, uv):
        self._add([p0, p1, p2, p3], [(0, 1, 2), (0, 2, 3)], [[uv[0], uv[1], uv[2]], [uv[0], uv[2], uv[3]]], mat)
    def box(self, c, s, mat, yaw=0.0, period=1.0, face_mats=None, poster_faces=()):
        """Box (optionally yawed). face_mats: dict face->material (None = omit face).
        UVs tile with `period` metres, except faces in poster_faces which map the texture once (0..1)."""
        c = np.asarray(c, float); hx, hy, hz = np.asarray(s, float) / 2
        R = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        faces = {
            'x-': [(-hx, -hy, -hz), (-hx, hy, -hz), (-hx, hy, hz), (-hx, -hy, hz)],
            'x+': [(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz)],
            'y-': [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)],
            'y+': [(-hx, hy, -hz), (hx, hy, -hz), (hx, hy, hz), (-hx, hy, hz)],
            'z-': [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz)],
            'z+': [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)],
        }
        for f, pts in faces.items():
            m = face_mats[f] if (face_mats and f in face_mats) else mat
            if m is None: continue
            P4 = [c + R @ np.array(p) for p in pts]
            e1 = np.linalg.norm(P4[1] - P4[0]); e2 = np.linalg.norm(P4[3] - P4[0])
            if f in poster_faces:
                uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], float)
            else:
                o = (P4[0][0] * 0.37 + P4[0][1] * 0.61 + P4[0][2] * 0.23) / period   # avoid identical repeats
                uv = np.array([[0, 0], [e1, 0], [e1, e2], [0, e2]], float) / period + o
            self.quad(*P4, m, uv)
    def cylinder(self, c, r, h, mat, seg=28, period=1.0):
        c = np.asarray(c, float)
        for k in range(seg):
            a0, a1 = 2 * math.pi * k / seg, 2 * math.pi * (k + 1) / seg
            p0 = c + [r * math.cos(a0), r * math.sin(a0), 0]; p1 = c + [r * math.cos(a1), r * math.sin(a1), 0]
            u0, u1 = a0 * r / period, a1 * r / period
            self.quad(p0, p1, p1 + [0, 0, h], p0 + [0, 0, h], mat, np.array([[u0, 0], [u1, 0], [u1, h / period], [u0, h / period]]))
            top = c + [0, 0, h]
            self._add([top, p0 + [0, 0, h], p1 + [0, 0, h]], [(0, 1, 2)],
                      [[[0.5, 0.5], [0.5 + math.cos(a0) / 2, 0.5 + math.sin(a0) / 2], [0.5 + math.cos(a1) / 2, 0.5 + math.sin(a1) / 2]]], mat)
    def arrays(self):
        return (np.concatenate(self.V).astype(np.float32), np.array(self.T, np.uint32),
                np.array(self.UV, np.float32), np.array(self.M, np.int64))

FACES = ('x-', 'x+', 'y-', 'y+', 'z-', 'z+')
def only(face, m):
    d = {k: None for k in FACES}; d[face] = m; return d

def room(mb, L, Wd, Hc, wall_mats, floor_mat, ceil_mat):
    """Room interior: floor z=0, ceiling z=Hc, walls x=0, x=L, y=0, y=Wd (only inner faces)."""
    mb.box((L / 2, Wd / 2, -0.05), (L, Wd, 0.1), None, face_mats=only('z+', floor_mat), period=2.0)
    mb.box((L / 2, Wd / 2, Hc + 0.05), (L, Wd, 0.1), None, face_mats=only('z-', ceil_mat), period=2.0)
    mb.box((-0.05, Wd / 2, Hc / 2), (0.1, Wd, Hc), None, face_mats=only('x+', wall_mats[0]), period=2.0)
    mb.box((L + 0.05, Wd / 2, Hc / 2), (0.1, Wd, Hc), None, face_mats=only('x-', wall_mats[1]), period=2.0)
    mb.box((L / 2, -0.05, Hc / 2), (L, 0.1, Hc), None, face_mats=only('y+', wall_mats[2]), period=2.0)
    mb.box((L / 2, Wd + 0.05, Hc / 2), (L, 0.1, Hc), None, face_mats=only('y-', wall_mats[3]), period=2.0)

def panel(mb, c, s, face, m):
    """Thin wall-mounted panel (poster, whiteboard) textured once on `face`."""
    mb.box(c, s, None, face_mats=only(face, m), poster_faces=(face,))

def mannequin(mat_body, mat_legs, mat_head):
    """Human-sized rigid agent in its local frame (origin on the floor, facing +x)."""
    mb = MeshBuilder()
    mb.box((0, 0.1, 0.43), (0.16, 0.15, 0.86), mat_legs, period=0.5)
    mb.box((0, -0.1, 0.43), (0.16, 0.15, 0.86), mat_legs, period=0.5)
    mb.box((0, 0, 1.17), (0.26, 0.46, 0.62), mat_body, period=0.6, poster_faces=('x+', 'x-'))
    mb.box((0, 0.29, 1.15), (0.11, 0.10, 0.58), mat_body, period=0.5)
    mb.box((0, -0.29, 1.15), (0.11, 0.10, 0.58), mat_body, period=0.5)
    mb.box((0, 0, 1.62), (0.21, 0.19, 0.25), mat_head, period=0.3)
    return mb.arrays()

# ----------------------------------------------------------------------------- scenes
def build_scene(sid):
    mats = []
    def mat(kind, seed):
        mats.append((kind, seed)); return len(mats) - 1
    mb = MeshBuilder()
    if sid == 1:  # office / living room, 7 x 6 x 2.8 m
        L, Wd, Hc = 7.0, 6.0, 2.8
        room(mb, L, Wd, Hc, [mat('plaster', 15), mat('plaster', 12), mat('brick', 16), mat('plaster', 11)], mat('wood', 13), mat('plaster', 14))
        books = mat('books', 21); wood = mat('wood', 22); fabric = mat('fabric', 23); metal = mat('metal', 24)
        mb.box((1.5, 5.8, 1.0), (2.0, 0.38, 2.0), wood, face_mats={'y-': books}, poster_faces=('y-',))   # bookshelf
        mb.box((4.8, 5.55, 0.375), (1.6, 0.7, 0.75), wood)                                              # desk
        mb.box((4.8, 5.75, 0.95), (0.6, 0.05, 0.38), metal, face_mats={'y-': mat('screen', 25)}, poster_faces=('y-',))
        mb.box((6.68, 2.6, 0.55), (0.6, 1.2, 1.1), wood)                                                # cabinet
        mb.box((0.45, 2.5, 0.22), (0.85, 2.0, 0.45), fabric, period=0.8)                                 # sofa
        mb.box((0.12, 2.5, 0.6), (0.24, 2.0, 0.8), fabric, period=0.8)
        mb.box((3.5, 2.9, 0.40), (1.0, 0.8, 0.06), wood)                                                # coffee table
        for dx in (-0.42, 0.42):
            for dy in (-0.32, 0.32): mb.box((3.5 + dx, 2.9 + dy, 0.19), (0.06, 0.06, 0.38), metal)
        mb.box((3.6, 3.0, 0.005), (2.4, 1.8, 0.01), mat('carpet', 26), period=1.2)                       # rug
        panel(mb, (3.8, 5.985, 1.6), (1.2, 0.03, 0.8), 'y-', mat('poster', 27))
        panel(mb, (6.985, 4.4, 1.5), (0.03, 1.0, 1.3), 'x-', mat('poster', 28))
        panel(mb, (0.015, 4.6, 1.5), (0.03, 1.1, 0.8), 'x+', mat('poster', 29))
        mb.box((6.98, 1.0, 1.05), (0.04, 0.95, 2.1), wood)                                              # door
        mb.cylinder((6.35, 5.45, 0), 0.22, 0.55, mat('marble', 30), period=0.6)                         # plant
        mb.cylinder((6.35, 5.45, 0.55), 0.34, 0.8, mat('carpet', 31), period=0.5)
        lights = [((2.0, 2.0, 2.6), 1.0), ((5.0, 4.0, 2.6), 1.0), ((3.5, 5.0, 2.6), 0.6)]
        ctrl = [(0.0, (1.2, 0.9, 1.45), (2.6, 5.5, 1.1)),
                (3.0, (2.3, 0.9, 1.45), (4.0, 6.0, 1.2)),
                (6.0, (3.6, 1.1, 1.50), (6.2, 5.2, 1.1)),
                (9.0, (5.0, 1.2, 1.45), (7.0, 3.5, 1.2)),
                (12.0, (4.8, 1.6, 1.40), (3.5, 6.0, 1.1)),
                (15.0, (3.4, 1.5, 1.45), (1.2, 5.2, 1.2))]
        agents = [dict(path=[(1.2, 3.9), (5.8, 3.9), (5.8, 4.6), (1.2, 4.6)], speed=1.0, phase=0.0),
                  dict(path=[(5.9, 2.4), (2.2, 4.3), (5.9, 4.9)], speed=0.9, phase=2.0),
                  dict(path=[(2.5, 5.0), (2.5, 2.2), (4.6, 2.0), (4.6, 5.0)], speed=1.1, phase=5.0)]
    elif sid == 2:  # lab / kitchen, 8 x 5 x 3 m
        L, Wd, Hc = 8.0, 5.0, 3.0
        room(mb, L, Wd, Hc, [mat('plaster', 41), mat('brick', 42), mat('plaster', 43), mat('plaster', 44)], mat('tiles', 45), mat('plaster', 46))
        wood = mat('wood', 51); marble = mat('marble', 52); metal = mat('metal', 53)
        mb.box((3.5, 4.7, 0.45), (5.0, 0.6, 0.9), wood, face_mats={'z+': marble})                      # counter
        mb.box((3.5, 4.82, 1.85), (5.0, 0.36, 0.7), wood)                                               # upper cabinets
        mb.box((7.78, 2.5, 0.95), (0.44, 3.0, 1.9), metal, face_mats={'x-': mat('books', 55)}, poster_faces=('x-',))
        for x0 in (2.8, 5.4):                                                                           # lab tables
            mb.box((x0, 2.2, 0.88), (1.6, 0.8, 0.05), marble)
            mb.box((x0, 2.2, 0.43), (1.5, 0.7, 0.05), metal)
            for dx in (-0.72, 0.72):
                for dy in (-0.32, 0.32): mb.box((x0 + dx, 2.2 + dy, 0.43), (0.05, 0.05, 0.86), metal)
        mb.box((4.1, 3.6, Hc / 2), (0.4, 0.4, Hc), mat('plaster', 56))                                  # column
        mb.box((6.8, 4.2, 0.3), (0.6, 0.5, 0.6), mat('poster', 54), poster_faces=('x-', 'y-'))          # boxes
        mb.box((6.8, 4.2, 0.8), (0.5, 0.45, 0.4), mat('poster', 57), poster_faces=('x-', 'y-'))
        panel(mb, (4.0, 0.015, 1.5), (2.2, 0.03, 1.1), 'y+', mat('poster', 58))
        panel(mb, (0.015, 2.5, 1.6), (0.03, 1.4, 0.9), 'x+', mat('poster', 59))
        for (x, y) in ((1.4, 3.3), (6.6, 1.2)): mb.cylinder((x, y, 0), 0.18, 0.65, mat('fabric', 60), period=0.5)
        lights = [((2.0, 2.5, 2.8), 1.0), ((6.0, 2.5, 2.8), 1.0), ((4.0, 1.0, 2.8), 0.5)]
        ctrl = [(0.0, (0.8, 0.7, 1.50), (3.0, 4.8, 1.2)),
                (3.0, (1.9, 0.8, 1.50), (5.0, 4.8, 1.0)),
                (6.0, (3.2, 0.9, 1.55), (7.5, 3.8, 1.1)),
                (9.0, (4.6, 0.9, 1.50), (7.8, 1.8, 1.2)),
                (12.0, (5.4, 1.1, 1.45), (4.5, 4.8, 1.1)),
                (15.0, (4.2, 1.0, 1.50), (0.5, 3.5, 1.2))]
        agents = [dict(path=[(1.0, 3.2), (7.0, 3.2), (7.0, 3.8), (1.0, 3.8)], speed=1.0, phase=1.0),
                  dict(path=[(6.9, 1.5), (4.8, 3.3), (1.5, 1.4), (4.8, 1.2)], speed=1.2, phase=0.0),
                  dict(path=[(3.4, 4.1), (3.4, 1.4), (6.8, 1.6), (6.4, 4.0)], speed=0.9, phase=3.0)]
    else:
        raise ValueError(sid)
    body = [mat('clothes', 90 + k) for k in range(3)]; legs = [mat('fabric', 95 + k) for k in range(3)]; head = mat('plaster', 99)
    return dict(materials=mats, static=mb.arrays(), lights=lights, ctrl=ctrl, agents=agents,
                agent_meshes=[mannequin(body[k], legs[k], head) for k in range(3)], agent_mats=set(body + legs + [head]))

# ----------------------------------------------------------------------------- trajectories
def catmull(ts, pts, t):
    ts = np.asarray(ts); P = np.asarray(pts, float)
    i = int(np.clip(np.searchsorted(ts, t, side='right') - 1, 0, len(ts) - 2))
    p0 = P[max(i - 1, 0)]; p1 = P[i]; p2 = P[i + 1]; p3 = P[min(i + 2, len(P) - 1)]
    u = (t - ts[i]) / (ts[i + 1] - ts[i])
    return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)

def camera_pose(ctrl, t):
    """Left camera pose T_wc (camera x right, y down, z forward; world z up)."""
    ts = [c[0] for c in ctrl]
    p = catmull(ts, [c[1] for c in ctrl], t) + np.array([0, 0, 0.02 * math.sin(2 * math.pi * 0.9 * t)])
    q = catmull(ts, [c[2] for c in ctrl], t)
    z = q - p; z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4); T[:3, 0] = x; T[:3, 1] = y; T[:3, 2] = z; T[:3, 3] = p
    return T

def agent_pose(agent, t):
    """Pose along the closed waypoint loop at arc length phase + speed*t; heading along the path."""
    P = np.asarray(agent['path'], float); P2 = np.vstack([P, P[:1]])
    seg = np.linalg.norm(np.diff(P2, axis=0), axis=1); cs = np.cumsum(seg)
    s = (agent['phase'] + agent['speed'] * t) % cs[-1]
    k = min(int(np.searchsorted(cs, s, side='right')), len(seg) - 1)
    d = (P2[k + 1] - P2[k]) / seg[k]
    pos = P2[k] + d * (s - (cs[k] - seg[k])); yaw = math.atan2(d[1], d[0])
    T = np.eye(4); T[:2, 3] = pos
    T[:3, :3] = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    return T

def rot_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2; return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; return (0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s)
    if R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; return ((R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s)
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; return ((R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s)

# ----------------------------------------------------------------------------- renderer
class Renderer:
    SUB = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]   # 2x2 supersampling offsets
    def __init__(self, scene, beta, n_agents):
        self.sc = scene; self.n_agents = n_agents
        tex = [make_texture(k, s) for (k, s) in scene['materials']]
        self.means = [float(t.mean()) for t in tex]
        stack = []
        for m, t in enumerate(tex):   # texture level: blend static albedo towards its per-material mean (agents unchanged)
            b = 0.0 if m in scene['agent_mats'] else beta
            stack.append(((1 - b) * t + b * self.means[m]).astype(np.float32))
        self.tex_stack = np.stack(stack); self.n = self.tex_stack.shape[1]; self.tex_flat = self.tex_stack.ravel()
        us, vs = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        self.dirs = np.stack([np.stack([(us + du - CX) / FX, (vs + dv - CY) / FY, np.ones_like(us)], -1)
                              for (du, dv) in [(0.0, 0.0)] + self.SUB]).astype(np.float32)   # (5,H,W,3), z = 1

    def build(self, t):
        rs = o3d.t.geometry.RaycastingScene(); geoms = []
        V, T, UV, M = self.sc['static']
        rs.add_triangles(o3d.core.Tensor(V), o3d.core.Tensor(T)); geoms.append((UV, M, False))
        for k in range(self.n_agents):
            Va, Ta, UVa, Ma = self.sc['agent_meshes'][k]
            Tw = agent_pose(self.sc['agents'][k], t)
            rs.add_triangles(o3d.core.Tensor((Va @ Tw[:3, :3].T + Tw[:3, 3]).astype(np.float32)), o3d.core.Tensor(Ta))
            geoms.append((UVa, Ma, True))
        return rs, geoms

    def render(self, rs, geoms, Twc, frame_seed):
        R, p = Twc[:3, :3].astype(np.float32), Twc[:3, 3].astype(np.float32)
        dw = self.dirs @ R.T
        rays = np.ascontiguousarray(np.concatenate([np.broadcast_to(p, dw.shape), dw], -1))
        ans = rs.cast_rays(o3d.core.Tensor(rays))
        th = ans['t_hit'].numpy(); gid = ans['geometry_ids'].numpy()
        hit = np.isfinite(th)
        # global primitive table (static + agents)
        UVs = np.concatenate([g[0] for g in geoms]); Ms = np.concatenate([g[1] for g in geoms])
        offs = np.cumsum([0] + [len(g[1]) for g in geoms])[:-1]
        dynflag = np.array([g[2] for g in geoms])
        sub = slice(1, None)                                  # supersamples used for shading
        h = hit[sub].ravel(); g = gid[sub].ravel()[h].astype(np.int64)
        pr = offs[g] + ans['primitive_ids'].numpy()[sub].ravel()[h].astype(np.int64)
        b = ans['primitive_uvs'].numpy()[sub].reshape(-1, 2)[h]
        nrm = ans['primitive_normals'].numpy()[sub].reshape(-1, 3)[h]
        tri = UVs[pr]; w0 = 1.0 - b[:, 0] - b[:, 1]
        uv = tri[:, 0] * w0[:, None] + tri[:, 1] * b[:, 0:1] + tri[:, 2] * b[:, 1:2]
        n = self.n
        x = (uv[:, 0] % 1.0) * n - 0.5; y = (uv[:, 1] % 1.0) * n - 0.5
        x0f = np.floor(x); y0f = np.floor(y); fx = (x - x0f).astype(np.float32); fy = (y - y0f).astype(np.float32)
        x0 = x0f.astype(np.int64) % n; y0 = y0f.astype(np.int64) % n; x1 = (x0 + 1) % n; y1 = (y0 + 1) % n
        base = Ms[pr] * (n * n); flat = self.tex_flat
        alb = ((flat.take(base + y0 * n + x0) * (1 - fx) + flat.take(base + y0 * n + x1) * fx) * (1 - fy)
               + (flat.take(base + y1 * n + x0) * (1 - fx) + flat.take(base + y1 * n + x1) * fx) * fy)
        ray = rays[sub].reshape(-1, 6)[h]
        P = ray[:, :3] + ray[:, 3:] * th[sub].ravel()[h][:, None]
        shade = np.full(len(P), 0.28, np.float32)
        for (lp, I) in self.sc['lights']:
            Lv = np.asarray(lp, np.float32) - P
            d2 = np.einsum('ij,ij->i', Lv, Lv); d = np.sqrt(d2) + 1e-6
            shade += (0.85 * I) * np.abs(np.einsum('ij,ij->i', nrm, Lv)) / d / (1.0 + d2 / 9.0)
        val = np.zeros(h.shape, np.float32); val[h] = alb * shade
        img = val.reshape(4, H, W).mean(0)
        img = np.clip(img * EXPOSURE, 0, 1) ** (1 / 2.2) * 255.0
        rng = np.random.default_rng(frame_seed)
        img = np.clip(img + rng.normal(0, NOISE_SIGMA, img.shape), 0, 255).round().astype(np.uint8)
        z = np.where(hit[0], th[0], 0.0).astype(np.float32)    # ray dir has unit z in camera frame -> t = depth
        dyn = hit & dynflag[np.where(hit, gid, 0).astype(np.int64)]
        mask = dyn.any(0).astype(np.uint8) * 255
        return img, z, mask

# ----------------------------------------------------------------------------- metrics
_fast = cv2.FastFeatureDetector_create(threshold=FAST_T, nonmaxSuppression=True, type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
def texture_metrics(img, mask):
    kps = _fast.detect(img, None)
    n_fast = 0
    for k in kps:
        x = min(int(round(k.pt[0])), W - 1); y = min(int(round(k.pt[1])), H - 1)
        n_fast += mask[y, x] == 0
    g = cv2.magnitude(cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3))
    gs = np.clip(g[mask == 0], 0, GRAD_MAX - 1e-3)
    h, _ = np.histogram(gs, bins=GRAD_BINS, range=(0, GRAD_MAX)); p = h / max(h.sum(), 1); p = p[p > 0]
    return int(n_fast), float(-(p * np.log2(p)).sum()), float((mask > 0).mean())

def write_settings(path):
    open(path, 'w').write(f"""%YAML:1.0
Camera.fx: {FX}
Camera.fy: {FY}
Camera.cx: {CX}
Camera.cy: {CY}
Camera.k1: 0.0
Camera.k2: 0.0
Camera.p1: 0.0
Camera.p2: 0.0
Camera.width: {W}
Camera.height: {H}
Camera.fps: {FPS}
Camera.bf: {FX * BASELINE}
Camera.RGB: 1
ThDepth: 40.0
DepthMapFactor: 5000.0
ORBextractor.nFeatures: 1000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
""")

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', type=int, required=True); ap.add_argument('--tex', type=int, required=True)
    ap.add_argument('--dyn', type=int, required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--frames', type=int, default=NFRAMES); ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--beta', type=float, default=None); ap.add_argument('--no-right', action='store_true')
    a = ap.parse_args()
    sc = build_scene(a.scene)
    beta = BETAS[a.tex] if a.beta is None else a.beta
    n_agents = [0, 1, 3][a.dyn]
    R = Renderer(sc, beta, n_agents)
    for d in ('left', 'right', 'depth', 'mask'): os.makedirs(os.path.join(a.out, d), exist_ok=True)
    write_settings(os.path.join(a.out, 'settings.yaml'))
    ft = open(os.path.join(a.out, 'times.txt'), 'w'); fg = open(os.path.join(a.out, 'groundtruth.txt'), 'w')
    fs = open(os.path.join(a.out, 'frame_stats.csv'), 'w'); fs.write('frame,n_fast,h_grad,rho\n')
    t0 = time.time(); j = 0
    for i in range(0, a.frames, a.stride):
        t = i / FPS
        Twc = camera_pose(sc['ctrl'], t)
        seed = 1000003 * a.scene + 2 * i
        rs, geoms = R.build(t)
        I, z, m = R.render(rs, geoms, Twc, seed)
        cv2.imwrite(os.path.join(a.out, 'left', f'{j:06d}.png'), I, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        cv2.imwrite(os.path.join(a.out, 'depth', f'{j:06d}.png'), np.clip(z * 5000.0, 0, 65535).round().astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        cv2.imwrite(os.path.join(a.out, 'mask', f'{j:06d}.png'), m, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not a.no_right:
            Tr = Twc.copy(); Tr[:3, 3] = Twc[:3, 3] + Twc[:3, 0] * BASELINE
            Ir, _, _ = R.render(rs, geoms, Tr, seed + 1)
            cv2.imwrite(os.path.join(a.out, 'right', f'{j:06d}.png'), Ir, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        nf, hg, rho = texture_metrics(I, m)
        fs.write(f'{j},{nf},{hg:.4f},{rho:.5f}\n'); ft.write(f'{t:.6f}\n')
        qx, qy, qz, qw = rot_to_quat(Twc[:3, :3])
        fg.write(f'{t:.6f} {Twc[0,3]:.6f} {Twc[1,3]:.6f} {Twc[2,3]:.6f} {qx:.8f} {qy:.8f} {qz:.8f} {qw:.8f}\n')
        j += 1
    for f in (ft, fg, fs): f.close()
    json.dump(dict(scene=a.scene, tex=a.tex, dyn=a.dyn, beta=beta, n_agents=n_agents, frames=j, fps=FPS, W=W, H=H,
                   fx=FX, baseline=BASELINE, noise_sigma=NOISE_SIGMA, render_s=round(time.time() - t0, 1)),
              open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)

if __name__ == '__main__':
    main()
