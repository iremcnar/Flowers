import math
import sys
import time

import numpy as np
from PIL import Image

SEED = 5
TIME_SCALE = 1.0
TICK_MS = 10
WIN_FRAC = 0.92
LOOP = False

KEYS = np.array([
    [0.0, 0.00, 0.00, 0.00, 0.0, 0.00, 0.00],
    [5.0, 0.42, 0.22, 0.00, 0.0, 0.25, 0.00],
    [10.0, 0.62, 0.42, 0.00, 0.0, 0.50, 0.35],
    [16.0, 0.80, 0.68, 0.50, 0.0, 0.72, 0.55],
    [24.0, 1.00, 1.00, 1.00, 0.0, 1.00, 1.00],
    [30.0, 1.00, 1.00, 1.00, 0.0, 1.00, 1.00],
    [39.0, 0.98, 0.85, 0.55, 1.0, 0.92, 0.00],
    [45.0, 0.98, 0.85, 0.55, 1.0, 0.90, 0.00],
])
T_END = float(KEYS[-1, 0])

SOIL_PAL = [(34, 25, 20), (50, 36, 28), (68, 50, 37), (88, 65, 47), (108, 82, 60), (64, 58, 52)]
STEM_PAL = [(62, 38, 120), (100, 58, 165), (140, 84, 200), (60, 160, 140)]
STEM_WILT = [(120, 88, 66), (96, 66, 52), (140, 108, 76)]
LEAF_PAL = [(60, 150, 140), (80, 120, 175), (130, 80, 195), (175, 105, 215)]
LEAF_WILT = [(130, 100, 70), (104, 76, 58), (146, 116, 84)]
RIB_PAL = [(210, 180, 255), (170, 230, 220)]
WILT_PAL = [(150, 86, 128), (116, 64, 108), (88, 52, 88), (126, 94, 104), (104, 72, 56)]
RIM_PAL = [(205, 165, 255), (255, 190, 250), (170, 140, 255)]
HI_PAL = [(235, 215, 255), (255, 225, 250)]
GOLD_PAL = [(255, 214, 110), (255, 238, 170), (255, 184, 90)]
STAMEN_PAL = [(255, 240, 255), (255, 215, 235), (255, 230, 150), (255, 190, 80)]
STAMEN_WILT = [(140, 112, 84), (120, 96, 72)]

LAYERS = [
    dict(n=8, ac=0.60, ao=1.00, kc=-1.20, ko=0.45, L=1.00, wc=0.20, wo=0.34, d=1150, off=0.00,
         pal=[(210, 80, 225), (170, 58, 225), (128, 50, 218), (96, 44, 205), (78, 42, 185)]),
    dict(n=6, ac=0.48, ao=0.68, kc=-1.05, ko=0.15, L=0.88, wc=0.19, wo=0.31, d=950, off=0.40,
         pal=[(255, 150, 235), (245, 92, 218), (205, 62, 218), (150, 52, 222)]),
    dict(n=5, ac=0.36, ao=0.46, kc=-0.85, ko=-0.05, L=0.72, wc=0.17, wo=0.27, d=760, off=0.20,
         pal=[(255, 205, 245), (255, 135, 232), (238, 84, 216), (195, 62, 226)]),
    dict(n=4, ac=0.22, ao=0.32, kc=-0.50, ko=-0.10, L=0.46, wc=0.14, wo=0.22, d=480, off=0.55,
         pal=[(255, 238, 252), (255, 185, 242), (255, 125, 228)]),
]
LEAF_DEFS = [
    (0.10, +1, 1.00, 0.55, -0.80),
    (0.16, -1, 0.90, 0.50, -0.85),
    (0.40, +1, 0.70, 0.95, -0.70),
    (0.33, -1, 0.62, 0.90, -0.65),
]

_OFF = sorted(((dx, dy) for dy in range(-3, 4) for dx in range(-3, 4) if dx * dx + dy * dy <= 9),
              key=lambda p: (p[0] ** 2 + p[1] ** 2, p))
OFF = np.array(_OFF, np.int32)
D2 = (OFF ** 2).sum(1)


def smoothstep(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def sinc(x):
    return np.sinc(x / np.pi)


def profile(u):
    return np.sin(np.pi * np.clip(u, 0, 1) ** 0.75) ** 0.85


def palette_pick(rng, pal, g, spread=0.55, jitter=0.07):
    pal = np.asarray(pal, np.float32)
    g = np.asarray(g, np.float32)
    x = np.clip(g, 0, 1) * (len(pal) - 1) + rng.normal(0, spread, g.shape)
    idx = np.clip(np.rint(x), 0, len(pal) - 1).astype(int)
    return (pal[idx] * (1 + rng.normal(0, jitter, (len(idx), 1)))).astype(np.float32)


def area_u(rng, n):
    out = np.empty(0)
    while out.size < n:
        u = rng.random(n * 3)
        out = np.concatenate([out, u[rng.random(n * 3) < profile(u)]])
    return out[:n]


def sample_v(rng, n, rim=0.14):
    v = rng.uniform(-1, 1, n)
    r = rng.random(n) < rim
    v[r] = np.sign(v[r]) * rng.uniform(0.88, 1.0, int(r.sum()))
    return v


def permute(d, rng):
    n = len(next(iter(d.values())))
    p = rng.permutation(n)
    return {k: v[p] for k, v in d.items()}


def cat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def splat(canvas, x, y, col, rad, nmax=13):
    H, W = canvas.shape[:2]
    xi = np.rint(x).astype(np.int32)
    yi = np.rint(y).astype(np.int32)
    off = OFF[:nmax]
    xs = xi[:, None] + off[None, :, 0]
    ys = yi[:, None] + off[None, :, 1]
    m = (D2[None, :nmax] <= (rad * rad)[:, None] + 0.01) & (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    rows = np.nonzero(m)[0]
    canvas[ys[m], xs[m]] = col[rows]


class Batch:
    def __init__(self):
        self.p = []

    def add(self, x, y, col, rad, z):
        if len(x):
            self.p.append((x, y, col, rad, z))

    def draw(self, canvas):
        if not self.p:
            return
        x, y, col, rad, z = [np.concatenate([q[i] for q in self.p]) for i in range(5)]
        o = np.argsort(z, kind="stable")
        c = np.clip(col[o], 0, 255).astype(np.uint8)
        splat(canvas, x[o], y[o], c, rad[o])


def state(t):
    t = float(min(max(t, 0.0), T_END))
    ts = KEYS[:, 0]
    i = min(max(int(np.searchsorted(ts, t, side="right")) - 1, 0), len(KEYS) - 2)
    u = (t - ts[i]) / (ts[i + 1] - ts[i])
    p0 = KEYS[max(i - 1, 0), 1:]
    p1 = KEYS[i, 1:]
    p2 = KEYS[i + 1, 1:]
    p3 = KEYS[min(i + 2, len(KEYS) - 1), 1:]
    v = 0.5 * (2 * p1 + (p2 - p0) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
               + (3 * p1 - p0 - 3 * p2 + p3) * u ** 3)
    return np.clip(v, 0, 1)


def bez(P, s):
    P0, P1, P2, P3 = P
    q = 1 - s
    x = q ** 3 * P0[0] + 3 * q * q * s * P1[0] + 3 * q * s * s * P2[0] + s ** 3 * P3[0]
    y = q ** 3 * P0[1] + 3 * q * q * s * P1[1] + 3 * q * s * s * P2[1] + s ** 3 * P3[1]
    dx = 3 * q * q * (P1[0] - P0[0]) + 6 * q * s * (P2[0] - P1[0]) + 3 * s * s * (P3[0] - P2[0])
    dy = 3 * q * q * (P1[1] - P0[1]) + 6 * q * s * (P2[1] - P1[1]) + 3 * s * s * (P3[1] - P2[1])
    return x, y, dx, dy


def mixcol(live, wilt, w):
    return live + (wilt - live) * w[:, None]


class Plant:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.rng = np.random.default_rng(SEED)
        self.cx = W / 2
        self.y0 = H * 0.80
        self.Ls = H * 0.38
        self.Lf = H * 0.235
        self.build_soil()
        self.build_stem()
        self.build_head()
        self.build_stamens()
        self.build_leaves()
        self.build_fx()

    def surface(self, x):
        H = self.H
        return (self.y0 - 0.022 * H * np.exp(-((x - self.cx) / (0.2 * self.W)) ** 2)
                - 0.003 * H * np.sin(x * 0.045) - 0.002 * H * np.sin(x * 0.13 + 1.3))

    def build_soil(self):
        W, H, rng = self.W, self.H, self.rng
        n = 26000
        x = rng.uniform(0, W, n)
        d = (rng.random(n) ** 1.5) * 0.14 * H
        y = self.surface(x) + d
        depth = d / (0.14 * H)
        edge = np.clip(np.minimum(x, W - x) / (0.2 * W), 0, 1) ** 0.8
        clump = 0.5 + 0.5 * np.sin(x * 0.07 + y * 0.05) * np.sin(y * 0.11 - x * 0.03)
        bright = (1 - depth) ** 0.8 * edge * (0.55 + 0.35 * clump + 0.35 * rng.random(n))
        col = palette_pick(rng, SOIL_PAL, rng.random(n), spread=1.0) * bright[:, None]
        rad = rng.choice([1.0, 1.5, 2.0], n, p=[0.4, 0.4, 0.2])
        self.base = np.zeros((H, W, 3), np.uint8)
        splat(self.base, x, y, np.clip(col, 0, 255).astype(np.uint8), rad)
        k = 170
        px = rng.uniform(0.05 * W, 0.95 * W, k)
        py = self.surface(px) + rng.random(k) ** 1.3 * 0.1 * H
        pc = palette_pick(rng, [(58, 52, 48), (84, 74, 64), (104, 90, 76), (46, 38, 32)], rng.random(k), spread=0.7)
        pc = pc * (np.clip(np.minimum(px, W - px) / (0.2 * W), 0, 1) ** 0.8)[:, None]
        splat(self.base, px, py, np.clip(pc, 0, 255).astype(np.uint8), rng.choice([2.0, 3.0], k, p=[0.6, 0.4]), nmax=29)
        m = 520
        cx = self.cx + rng.normal(0, 0.03 * H, m)
        self.cr = dict(
            x=cx,
            y=self.surface(cx) - rng.random(m) * 0.006 * H,
            amp=H * 0.05 * rng.random(m) ** 2.2,
            delay=rng.uniform(0, 2.5, m),
            col=(palette_pick(rng, SOIL_PAL, rng.uniform(0.3, 1, m), spread=0.8) * 0.8).astype(np.float32),
            rad=rng.choice([1.0, 1.5, 2.0], m, p=[0.4, 0.4, 0.2]),
        )

    def build_stem(self):
        rng, n = self.rng, 3600
        s = rng.random(n)
        lat = rng.uniform(-1, 1, n)
        rim = rng.random(n) < 0.18
        lat[rim] = np.sign(lat[rim]) * rng.uniform(0.85, 1.0, int(rim.sum()))
        live = palette_pick(rng, STEM_PAL[:3], s * 0.8 + rng.random(n) * 0.2)
        gr = rng.random(n) < 0.12
        live[gr] = np.asarray(STEM_PAL[3], np.float32) * (1 + rng.normal(0, 0.08, (int(gr.sum()), 1)))
        self.stem = permute(dict(
            s=s, lat=lat, live=live,
            wilt=palette_pick(rng, STEM_WILT, rng.random(n), spread=0.8),
            rn=rng.random(n), rad=rng.choice([1.0, 1.5, 2.0], n, p=[0.35, 0.45, 0.2])), rng)

    def ctrl(self, L, wilt, sway):
        cx, y0 = self.cx, self.y0
        P0 = (cx, y0 + 4.0)
        P1 = (cx - 0.06 * L * (1 - wilt) + 0.02 * L * wilt, y0 - 0.33 * L)
        P2 = (cx + 0.05 * L * (1 - wilt) + 0.08 * L * wilt + sway * 0.5, y0 - L * (0.74 + 0.30 * wilt))
        P3 = (cx + sway + 0.32 * L * wilt, y0 - L * (1 - 0.15 * wilt))
        return P0, P1, P2, P3

    def build_head(self):
        rng = self.rng
        parts = []
        for Ly in LAYERS:
            d = Ly["d"]
            for i in range(Ly["n"]):
                th = 2 * math.pi * i / Ly["n"] + Ly["off"] + rng.normal(0, 0.10)
                aj, lj = rng.normal(0, 0.06), rng.uniform(0.92, 1.08)
                pj, wj = rng.random(), rng.uniform(-0.5, 0.5)
                u, v = area_u(rng, d), sample_v(rng, d)
                col = palette_pick(rng, Ly["pal"], u + rng.normal(0, 0.10, d))
                sel = (np.abs(v) > 0.86) & (rng.random(d) < 0.55)
                col[sel] = palette_pick(rng, RIM_PAL, rng.random(int(sel.sum())))
                hl = rng.random(d) < 0.04
                col[hl] = palette_pick(rng, HI_PAL, rng.random(int(hl.sum())))
                fl = (rng.random(d) < 0.02) & (u < 0.55)
                col[fl] = palette_pick(rng, GOLD_PAL, rng.random(int(fl.sum())))
                f = lambda val: np.full(d, val, np.float32)
                parts.append(dict(
                    th=f(th), aj=f(aj), pj=f(pj), wj=f(wj), u=u.astype(np.float32), v=v.astype(np.float32),
                    ac=f(Ly["ac"]), ao=f(Ly["ao"]), kc=f(Ly["kc"]), ko=f(Ly["ko"]),
                    L=f(Ly["L"] * lj), wc=f(Ly["wc"]), wo=f(Ly["wo"]),
                    live=col, wilt=palette_pick(rng, WILT_PAL, 0.4 * u + 0.6 * rng.random(d), spread=0.7),
                    rn=rng.random(d).astype(np.float32),
                    rad=rng.choice([1.0, 1.5, 2.0], d, p=[0.3, 0.45, 0.25])))
        h = cat(parts)
        h["sp"] = profile(h["u"]).astype(np.float32)
        self.head = permute(h, rng)

    def build_stamens(self):
        rng, n, nf = self.rng, 700, 60
        fid = rng.integers(0, nf, n)
        t = 1 - rng.random(n) ** 1.6
        live = palette_pick(rng, STAMEN_PAL, t, spread=0.35)
        self.st = permute(dict(
            th=rng.uniform(0, 2 * math.pi, nf)[fid].astype(np.float32),
            beta=rng.uniform(0.2, 0.85, nf)[fid].astype(np.float32),
            ell=rng.uniform(0.30, 0.55, nf)[fid].astype(np.float32),
            t=t.astype(np.float32), live=live,
            wilt=palette_pick(rng, STAMEN_WILT, rng.random(n), spread=0.6),
            rad=np.where(t > 0.88, 2.0, rng.choice([1.0, 1.5], n)).astype(np.float32)), rng)

    def build_leaves(self):
        rng, parts = self.rng, []
        d = 900
        for j in range(len(LEAF_DEFS)):
            u, v = area_u(rng, d), sample_v(rng, d, 0.10)
            rib = rng.random(d) < 0.09
            v[rib] = rng.normal(0, 0.03, int(rib.sum()))
            live = palette_pick(rng, LEAF_PAL, u + rng.normal(0, 0.1, d))
            live[rib] = palette_pick(rng, RIB_PAL, rng.random(int(rib.sum())))
            parts.append(dict(
                lid=np.full(d, j), u=u.astype(np.float32), v=v.astype(np.float32), live=live,
                wilt=palette_pick(rng, LEAF_WILT, rng.random(d), spread=0.8),
                rn=rng.random(d).astype(np.float32),
                rad=rng.choice([1.0, 1.5, 2.0], d, p=[0.35, 0.45, 0.2])))
        self.leaf = permute(cat(parts), rng)

    def build_fx(self):
        rng = self.rng
        n = 1800
        r = np.minimum(np.abs(rng.normal(0, 0.55, n)), 1.6)
        self.halo = permute(dict(
            a=rng.uniform(0, 2 * math.pi, n), r=r,
            b=((1 - r / 1.6) ** 1.5 * 0.55).astype(np.float32),
            fl=rng.random(n).astype(np.float32)), rng)
        cf, cj, rho, a = [], [], [], []
        for j in range(7):
            m = 700 if j == 0 else 170
            cf += [0.0 if j == 0 else 0.5] * m
            cj += [max(j - 1, 0)] * m
            rho += [1.0 if j == 0 else 0.5] * m
            a += list(rng.uniform(0, 2 * math.pi, m))
        self.ring = permute(dict(cf=np.array(cf), cj=np.array(cj), rho=np.array(rho), a=np.array(a),
                                 fl=rng.random(len(a))), rng)
        m = 280
        self.spark = dict(
            a=rng.uniform(0, 2 * math.pi, m), rho=rng.random(m), ph=rng.random(m),
            sp=rng.uniform(0.6, 1.4, m), rad=rng.choice([1.0, 2.0], m, p=[0.7, 0.3]),
            hue=rng.random(m))
        m = 260
        xl = self.cx + rng.normal(0, 0.12 * self.H, m)
        self.fall = dict(
            r0=rng.random(m), ox=rng.normal(0, 1, m), oy=rng.normal(0, 1, m),
            xl=xl, yl=self.surface(xl) - rng.random(m) * 0.012 * self.H,
            ph=rng.uniform(0, 6.28, m),
            col=palette_pick(rng, WILT_PAL, rng.random(m), spread=0.8),
            rad=rng.choice([1.5, 2.0], m))

    def render(self, t):
        H, W = self.H, self.W
        stem, size, bloom, wilt, leaf, glow = state(t)
        canvas = self.base.copy()
        B = Batch()
        nodes = Batch()
        Lf = self.Lf

        L = self.Ls * stem
        sway = 0.012 * H * math.sin(t * 0.9) * stem * (1 - 0.6 * wilt)
        P = self.ctrl(L, wilt, sway)
        hx, hy = P[3]
        if stem > 0.01:
            S = self.stem
            nv = int(len(S["s"]) * np.clip(stem, 0.08, 1))
            s = S["s"][:nv]
            x, y, dx, dy = bez(P, s)
            nn = np.hypot(dx, dy) + 1e-6
            wst = 0.0085 * H * (1 - 0.3 * s) * (1 - 0.25 * wilt) * np.clip(0.4 + stem * 1.2, 0.4, 1)
            lat = S["lat"][:nv] * wst
            wi = np.clip(wilt * (0.75 + 0.5 * s) * (0.7 + 0.5 * S["rn"][:nv]), 0, 1)
            B.add(x - dy / nn * lat, y + dx / nn * lat,
                  mixcol(S["live"][:nv], S["wilt"][:nv], wi), S["rad"][:nv], np.full(nv, -500.0))

        if leaf > 0.02 and stem > 0.05:
            S = self.leaf
            nv = int(len(S["u"]) * np.clip(leaf, 0.15, 1))
            sa = np.array([d[0] for d in LEAF_DEFS])
            ax, ay, _, _ = bez(P, sa)
            lid = S["lid"][:nv]
            m = np.array([d[1] for d in LEAF_DEFS], np.float32)[lid]
            lfac = np.array([d[2] for d in LEAF_DEFS], np.float32)[lid]
            a0 = np.array([d[3] for d in LEAF_DEFS], np.float32)[lid] + (1 - leaf) * 0.8 - 0.6 * wilt
            c = np.array([d[4] for d in LEAF_DEFS], np.float32)[lid] - 0.5 * wilt
            u, v = S["u"][:nv], S["v"][:nv]
            LL = 0.19 * H * lfac * leaf * (1 - 0.25 * wilt)
            am = a0 + c * u * 0.5
            al = a0 + c * u
            sc = sinc(c * u * 0.5)
            lat = v * 0.27 * LL * profile(u)
            X = ax[lid] + LL * u * m * np.cos(am) * sc - lat * np.sin(al)
            Yup = LL * u * np.sin(am) * sc + lat * m * np.cos(al)
            wi = np.clip(wilt * (0.9 + 0.5 * u + 0.3 * S["rn"][:nv]) - 0.1, 0, 1)
            B.add(X, ay[lid] - Yup, mixcol(S["live"][:nv], S["wilt"][:nv], wi),
                  S["rad"][:nv], np.full(nv, -450.0))

        if stem > 0.01 and size > 0.05:
            phi = wilt * 2.75 + 0.035 * math.sin(t * 0.9 + 1.0) * (1 - wilt)
            tau = 0.20 + 0.36 * bloom - 0.40 * wilt
            cphi, sphi, ct, st_ = math.cos(phi), math.sin(phi), math.cos(tau), math.sin(tau)
            Ls_ = Lf * size

            def project(xl, yl, zl):
                yp = yl * ct - zl * st_
                zp = yl * st_ + zl * ct
                return hx + xl * cphi + yp * sphi, hy - (-xl * sphi + yp * cphi), zp, yp

            vis = float(np.clip(size ** 2 * 1.15, 0.2, 1.0))
            h = self.head
            nv = int(len(h["u"]) * vis)
            g = {k: val[:nv] for k, val in h.items()}
            u, v = g["u"], g["v"]
            A0 = g["ac"] + (g["ao"] - g["ac"]) * bloom + g["aj"]
            A0 = A0 * (1 - 0.55 * wilt) + wilt * g["wj"] * 0.3
            K = g["kc"] + (g["ko"] - g["kc"]) * bloom + wilt * (0.6 + 0.6 * g["pj"])
            Lp = Ls_ * g["L"] * (1 - 0.28 * wilt * g["pj"])
            am = A0 + K * u * 0.5
            sc = sinc(K * u * 0.5)
            rr = Lp * u * np.sin(am) * sc
            hh = Lp * u * np.cos(am) * sc
            lat = v * (g["wc"] + (g["wo"] - g["wc"]) * bloom) * Lp * g["sp"]
            rr = rr + 0.025 * Ls_ - 0.5 * lat * lat / (Lp + 1.0)
            cth, sth = np.cos(g["th"]), np.sin(g["th"])
            X, Y, zp, yp = project(cth * rr - sth * lat, hh, sth * rr + cth * lat)
            shade = np.clip(0.72 + 0.28 * np.clip(zp / Ls_ * 0.5 + 0.5, 0, 1) + 0.12 * (hh / Ls_), 0.5, 1.2)
            wi = np.clip(wilt * (1.0 + 0.7 * u + 0.4 * g["rn"]) - 0.2, 0, 1)
            col = mixcol(g["live"], g["wilt"], wi) * shade[:, None]
            B.add(X, Y, col, g["rad"], zp)

            ss = float(smoothstep(bloom, 0.30, 0.85))
            if ss > 0.02:
                S = self.st
                nv2 = int(len(S["t"]) * np.clip(size ** 2, 0.25, 1))
                g = {k: val[:nv2] for k, val in S.items()}
                ell = Ls_ * ss * g["ell"]
                beta = g["beta"] * (0.6 + 0.4 * bloom) + wilt * 0.3
                r = ell * g["t"] * np.sin(beta) + 0.02 * Ls_
                yl = 0.05 * Ls_ + ell * g["t"] * np.cos(beta)
                X, Y, zp, _ = project(np.cos(g["th"]) * r, yl, np.sin(g["th"]) * r)
                w = np.full(nv2, wilt, np.float32)
                B.add(X, Y, mixcol(g["live"], g["wilt"], w), g["rad"], zp + 3.0)

            if glow > 0.02:
                axis = np.array([math.sin(phi), -math.cos(phi)])
                ccx, ccy = hx + axis[0] * 0.5 * Ls_, hy + axis[1] * 0.5 * Ls_
                Hh = self.halo
                nh = int(len(Hh["a"]) * glow)
                rh = Hh["r"][:nh] * Ls_ * 1.1
                br = Hh["b"][:nh] * glow * (0.85 + 0.3 * np.sin(t * 1.3 + Hh["fl"][:nh] * 6.28))
                B.add(ccx + np.cos(Hh["a"][:nh]) * rh, ccy + np.sin(Hh["a"][:nh]) * rh,
                      np.asarray([150, 70, 240], np.float32)[None, :] * br[:, None],
                      np.ones(nh), np.full(nh, -2000.0))

                R = self.ring
                nr = int(len(R["a"]) * glow)
                Rs = Ls_ * 1.25
                cang = 2 * math.pi * R["cj"][:nr] / 6 + t * 0.15
                a = R["a"][:nr] + t * 0.05
                rx = ccx + Rs * (R["cf"][:nr] * np.cos(cang) + R["rho"][:nr] * np.cos(a))
                ry = ccy + Rs * (R["cf"][:nr] * np.sin(cang) + R["rho"][:nr] * np.sin(a))
                bb = (0.40 + 0.60 * glow) * (0.65 + 0.35 * np.sin(t * 2 + R["fl"][:nr] * 20))
                B.add(rx, ry, np.asarray([210, 145, 255], np.float32)[None, :] * bb[:, None],
                      np.ones(nr), np.full(nr, -1500.0))
                ja = 2 * math.pi * np.arange(6) / 6 + t * 0.15
                nodes.add(ccx + Rs * np.cos(ja), ccy + Rs * np.sin(ja),
                          np.tile(np.asarray([255, 215, 255], np.float32) * glow, (6, 1)),
                          np.full(6, 3.0), np.zeros(6))

                Sp = self.spark
                life = (t * 0.10 * Sp["sp"] + Sp["ph"]) % 1.0
                rho = (0.25 + 1.15 * Sp["rho"]) * Ls_ * (0.6 + 0.6 * life)
                px = ccx + np.cos(Sp["a"]) * rho
                py = ccy + np.sin(Sp["a"]) * rho - life * 0.25 * Ls_
                bs = (np.sin(np.pi * life) ** 0.8 * glow)[:, None]
                cs = np.where(Sp["hue"][:, None] < 0.5, np.asarray([255, 215, 255], np.float32),
                              np.asarray([255, 150, 235], np.float32))
                B.add(px, py, cs * bs, Sp["rad"], np.full(len(px), 2000.0))

            if wilt > 0.02:
                F = self.fall
                fp = np.clip((wilt * 1.05 - F["r0"] * 0.7) / 0.35, 0, 1)
                mk = fp > 0
                if mk.any():
                    f = fp[mk]
                    sx = hx + F["ox"][mk] * 0.18 * Ls_
                    sy = hy + F["oy"][mk] * 0.18 * Ls_ + 0.25 * Ls_
                    x = sx + (F["xl"][mk] - sx) * f + np.sin(f * 6.28 + F["ph"][mk]) * 0.03 * H * (1 - f)
                    y = sy + (F["yl"][mk] - sy) * f * f
                    B.add(x, y, F["col"][mk], F["rad"][mk], np.full(int(mk.sum()), 500.0))

        B.draw(canvas)
        if nodes.p:
            x, y, col, rad, _ = [np.concatenate([q[i] for q in nodes.p]) for i in range(5)]
            splat(canvas, x, y, np.clip(col, 0, 255).astype(np.uint8), rad, nmax=29)

        C = self.cr
        e = np.clip((t - C["delay"]) / 3.0, 0, 1)
        sy = C["y"] - C["amp"] * np.sin(np.pi * e)
        splat(canvas, C["x"], sy, np.clip(C["col"], 0, 255).astype(np.uint8), C["rad"])
        return canvas


def save_png():
    H = 1000
    plant = Plant(H, H)
    times = [3.0, 9.0, 16.0, 26.0, 41.0]
    frames = []
    for i, tt in enumerate(times):
        a = plant.render(tt)
        Image.fromarray(a).save(f"flower_{i + 1}.png")
        frames.append(Image.fromarray(a).resize((500, 500), Image.LANCZOS))
    strip = Image.new("RGB", (500 * len(frames), 500))
    for i, f in enumerate(frames):
        strip.paste(f, (500 * i, 0))
    strip.save("flower_stages.png")
    print("Saved: flower_stages.png and stage frames")


def save_gif(fps=12, size=640):
    plant = Plant(size, size)
    frames = [Image.fromarray(plant.render(i / fps)).convert("P", palette=Image.ADAPTIVE)
              for i in range(int(T_END * fps) + 1)]
    frames[0].save("flower.gif", save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=False)
    print("Saved: flower.gif")


class App:
    def __init__(self):
        import tkinter as tk
        from PIL import ImageTk

        self.root = tk.Tk()
        self.root.title("Pointillism Flower")
        self.root.configure(bg="black")
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        H = int(min(sh * WIN_FRAC, sw * 0.96))
        W = H
        self.root.geometry(f"{W}x{H}+{(sw - W) // 2}+{max(0, (sh - H) // 2 - 20)}")
        self.cv = tk.Canvas(self.root, width=W, height=H, bg="black", highlightthickness=0)
        self.cv.pack(fill="both", expand=True)
        self.cv.create_text(W // 2, H // 2, text="preparing...", fill="#777777")
        self.root.update()

        self.plant = Plant(W, H)
        self.photo = ImageTk.PhotoImage(Image.fromarray(self.plant.render(0)))
        self.cv.delete("all")
        self.cv.create_image(0, 0, anchor="nw", image=self.photo)
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.bind("<space>", lambda e: self.toggle())
        self.root.bind("<r>", lambda e: self.restart())
        self.root.bind("<s>", lambda e: Image.fromarray(self.plant.render(self.clock / TIME_SCALE)).save("flower_frame.png"))
        self.clock, self.last, self.paused = 0.0, time.perf_counter(), False
        self.tick()

    def toggle(self):
        self.paused = not self.paused

    def restart(self):
        self.clock = 0.0

    def tick(self):
        now = time.perf_counter()
        if not self.paused:
            self.clock += now - self.last
        self.last = now
        t = self.clock / TIME_SCALE
        if LOOP and t > T_END + 4:
            self.clock, t = 0.0, 0.0
        self.photo.paste(Image.fromarray(self.plant.render(t)))
        self.root.after(TICK_MS, self.tick)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    if "--png" in sys.argv:
        save_png()
    elif "--gif" in sys.argv:
        save_gif()
    else:
        App().run()