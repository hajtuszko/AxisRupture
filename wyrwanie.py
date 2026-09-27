# Etap 1: wyrwanie desek przez teren (wyginanie przybitych desek, stan koncowy)
import bpy, bmesh, math, random, zlib
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

UPW = Vector((0, 0, 1))


# ---------------------------------------------------------------- geometria 2D deski
def upper_hull(pts):
    pts = sorted(pts); h = []
    for p in pts:
        while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) -
                               (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0])) >= 0:
            h.pop()
        h.append(p)
    return h


def hull_at(h, x):
    for a, b in zip(h, h[1:]):
        if a[0] <= x <= b[0]:
            return a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]) if b[0] > a[0] else max(a[1], b[1])
    return 0.0


def simplify_loops(nodes, tol):
    """Douglas-Peucker na profilu deski: zwraca indeksy wezlow (petli), bez ktorych
    deska odchylilaby sie od ksztaltu o wiecej niz tol"""
    keep = {0, len(nodes) - 1}
    stack = [(0, len(nodes) - 1)]
    pts = [Vector(p) for p in nodes]
    while stack:
        i, j = stack.pop()
        a, seg = pts[i], pts[j] - pts[i]
        ln2 = seg.length_squared or 1e-24
        best, bk = 0.0, None
        for k in range(i + 1, j):
            v = pts[k] - a
            d = (v - seg * (v.dot(seg) / ln2)).length       # odleglosc od cieciwy (x, z, skret krawedzi)
            if d > best:
                best, bk = d, k
        if bk is not None and best > tol:
            keep.add(bk); stack += [(i, bk), (bk, j)]
    return keep


def arc_nodes(zs, anchor, x_anchor, ds):
    """wysokosci w wezlach -> wezly o stalej dlugosci odcinkow, liczone od gwozdzia (anchor)"""
    n = len(zs); xs = [0.0] * n; xs[anchor] = x_anchor
    for k in range(anchor + 1, n):
        dz = max(-0.95 * ds, min(0.95 * ds, zs[k] - zs[k - 1]))
        xs[k] = xs[k - 1] + math.sqrt(ds * ds - dz * dz)
    for k in range(anchor - 1, -1, -1):
        dz = max(-0.95 * ds, min(0.95 * ds, zs[k + 1] - zs[k]))
        xs[k] = xs[k + 1] - math.sqrt(ds * ds - dz * dz)
    return list(zip(xs, zs))


def canon(nodes, ds):
    """wezly -> lancuch o dokladnie stalej dlugosci odcinkow (deska sie nie rozciaga)"""
    x, z = nodes[0]; out = [(x, z)]
    for k in range(len(nodes) - 1):
        phi = math.atan2(nodes[k + 1][1] - nodes[k][1], nodes[k + 1][0] - nodes[k][0])
        x += ds * math.cos(phi); z += ds * math.sin(phi)
        out.append((x, z))
    return out


# ---------------------------------------------------------------- deska
class Board:
    def __init__(self, o):
        self.o = o
        mw = o.matrix_world
        R = mw.to_3x3()
        axes = [R.col[k].copy() for k in range(3)]
        sc = [a.length for a in axes]
        axes = [a.normalized() for a in axes]
        vs = [v.co for v in o.data.vertices]
        mn = Vector([min(v[k] for v in vs) for k in range(3)])
        mx = Vector([max(v[k] for v in vs) for k in range(3)])
        size = [(mx[k] - mn[k]) * sc[k] for k in range(3)]
        a_len = 0 if size[0] >= size[1] else 1
        self.es, self.ew, self.up = axes[a_len], axes[1 - a_len], axes[2]
        self.L, self.W, self.T = size[a_len], size[1 - a_len], size[2]
        if self.up.z < 0:
            self.up = -self.up
        if self.es.cross(self.ew).dot(self.up) < 0:
            self.ew = -self.ew
        self.c = mw @ ((mn + mx) / 2)
        self.F = Matrix((self.es, self.ew, self.up)).transposed().to_4x4()
        self.F.translation = self.c
        self.Finv = self.F.inverted()
        self.flat = self.up.z > 0.99
        self.z0 = self.c.z - self.T / 2          # poziom podlogi pod deska
        self.shift = Vector()                    # przesuniecie w swiecie (rozsuwanie, podniesienie)
        self.phase = 0
        self.nodes = None
        self.kinks = []                          # zalamania (s, kat, promien bevela)
        self.poly = None
        self.roll = None                         # skret deski w poprzek w kazdym wezle (rad)

    def setup(self, seg_len, nail_frac):
        self.N = max(6, int(round(self.L / seg_len)))
        self.ds = self.L / self.N
        self.sk = [-self.L / 2 + k * self.ds for k in range(self.N + 1)]
        self.kA = max(1, int(round(nail_frac * self.N)))
        self.kB = self.N - self.kA
        # gwozdzie wykryte na teksturze (Przygotowanie desek): (wezel k, polozenie w poprzek w)
        self.tex_nails = []
        d = self.o.get("deski_gwozdzie_tex")
        if d:
            mw = self.o.matrix_world
            for i in range(0, len(d), 3):
                q = self.Finv @ (mw @ Vector(d[i:i + 3]))
                k = min(self.N - 1, max(1, int(round((q.x + self.L / 2) / self.ds))))
                self.tex_nails.append((k, max(-self.W / 2, min(self.W / 2, q.y))))
            ks = sorted(k for k, w in self.tex_nails)
            # deska trzyma sie na skrajnych gwozdziach (najblizej koncow)
            if ks and ks[0] < self.N / 2:
                self.kA = ks[0]
            if ks and ks[-1] > self.N / 2:
                self.kB = ks[-1]
            if self.kB - self.kA < 2:
                self.kA = max(1, int(round(nail_frac * self.N))); self.kB = self.N - self.kA
        ns = max(21, int(math.ceil(self.L / 0.02)) + 1)
        self.sg = [-self.L / 2 + self.L * j / (ns - 1) for j in range(ns)]
        self.wg = [-self.W / 2, -self.W / 4, 0.0, self.W / 4, self.W / 2]

    def rest_nodes(self):
        return [(s, 0.0) for s in self.sk]

    def frame_point(self, x, w, z):
        """(x wzdluz, w w poprzek, z = wysokosc spodu nad spoczynkiem) -> punkt swiata"""
        return self.F @ Vector((x, w, z - self.T / 2)) + self.shift

    def section(self, k):
        """przekroj deski w wezle k: punkt spodu (x,z), normalna 2D (srednia sasiednich odcinkow), skret"""
        nodes = self.nodes or self.rest_nodes()
        N = len(nodes) - 1
        def seg_n(j):
            (x0, z0), (x1, z1) = nodes[j], nodes[j + 1]
            d = Vector((x1 - x0, z1 - z0)).normalized()
            return Vector((-d.y, d.x))
        n = seg_n(0) if k == 0 else seg_n(N - 1) if k == N else (seg_n(k - 1) + seg_n(k)).normalized()
        psi = self.roll[k] if self.roll else 0.0
        return Vector(nodes[k]), n, psi

    def node_point(self, k, w, h=0.0):
        """punkt swiata na przekroju k: w = w poprzek, h = wysokosc nad spodem (0..T)"""
        P, n, psi = self.section(k)
        u = w * math.sin(psi) + h * math.cos(psi)
        y = w * math.cos(psi) - h * math.sin(psi)
        q = P + n * u
        return self.F @ Vector((q.x, y, q.y - self.T / 2)) + self.shift


# ---------------------------------------------------------------- rdzen (bez operatora - wolany tez z podgladu na zywo)
class Settings:
    """ustawienia z nadpisaniami (np. jakosc podgladu): Settings(p, bevel_segments=2).bevel_segments"""
    def __init__(self, p, **over):
        self.__dict__["_p"] = p; self.__dict__["_o"] = over

    def __getattr__(self, k):
        o = self.__dict__["_o"]
        return o[k] if k in o else getattr(self.__dict__["_p"], k)


class DeformCore:
    # --- teren
    def terrain_bvh(self, context, terrain):
        dg = context.evaluated_depsgraph_get()
        te = terrain.evaluated_get(dg)
        me = te.to_mesh()
        verts = [te.matrix_world @ v.co for v in me.vertices]
        polys = [tuple(p.vertices) for p in me.polygons]
        te.to_mesh_clear()
        self.TB = BVHTree.FromPolygons(verts, polys)
        self.ttop = max(v.z for v in verts)
        self.tbot = min(v.z for v in verts)

    def th(self, x, y):
        hit = self.TB.ray_cast(Vector((x, y, self.ttop + 1.0)), Vector((0, 0, -1)))
        return hit[0].z + getattr(self, "zoff", 0.0) if hit[0] else -1e9

    # --- symulacja jednej deski
    def simulate(self, b, p):
        rnd = random.Random(p.seed * 1000003 + zlib.crc32(b.o.name.encode()))
        tau = rnd.uniform(p.bulge_min, p.bulge_max)
        slope_max = rnd.uniform(p.slope_min, p.slope_max)
        th_max = rnd.uniform(p.angle2_min, p.angle2_max)
        EPS, ds, kA, kB = p.clearance, b.ds, b.kA, b.kB
        a = b.sk[kB]
        Hw = [[self.th(*(b.c + b.es * s + b.ew * w).xy) for w in b.wg] for s in b.sg]
        H = [max(r) for r in Hw]
        if max(H) <= b.z0 and not getattr(self, "min_phase", {}).get(b.o.name):
            return False                                   # teren nie dochodzi do deski
        D = max(0.0, self.ttop - b.z0) + 0.01             # teren startuje ponizej deski
        phase, side, w_last = 1, 0, None
        for t in range(p.steps + 1):
            dz = -D * (1 - t / p.steps)
            g = [(s, max(h + dz - b.z0 + EPS, 0.0)) for s, h in zip(b.sg, H)]
            if phase == 1:
                pts = [(-a, 0.0), (a, 0.0)] + [(s, u) for s, u in g if -a < s < a and u > EPS]
                hl = upper_hull(pts)
                w = [hull_at(hl, s) if -a <= s <= a else 0.0 for s in b.sk]
                for s, u in g:                             # odcinki miedzy wezlami nad terenem
                    if not (-a < s < a) or u <= 0:
                        continue
                    k = min(b.N - 1, int((s - b.sk[0]) / ds)); f = (s - b.sk[k]) / ds
                    deficit = u - (w[k] * (1 - f) + w[k + 1] * f)
                    if deficit > 0:
                        if k == kA:
                            w[k + 1] += deficit / max(f, 0.05)
                        elif k + 1 == kB:
                            w[k] += deficit / max(1 - f, 0.05)
                        else:
                            w[k] += deficit; w[k + 1] += deficit
                slA = math.atan2(w[kA + 1], ds); slB = math.atan2(w[kB - 1], ds)
                outA = max(u for s, u in g if s <= -a + 1e-6)
                outB = max(u for s, u in g if s >= a - 1e-6)
                if max(w) > tau or max(slA, slB) > slope_max or max(outA, outB) > EPS:
                    phase = 2
                    if max(outA, outB) > EPS:
                        side = 1 if outB >= outA else -1
                    else:
                        side = 1 if slB >= slA else -1
                else:
                    w_last = w
            if phase == 2:
                kH = kA if side == 1 else kB
                sH = b.sk[kH]
                ang = max([math.atan2(u, side * (s - sH)) for s, u in g
                           if 1e-4 < side * (s - sH)] or [0.0])
                under = max(u for s, u in g if side * (s - sH) < 0.03)
                if ang > th_max or under > EPS:
                    phase = 3
                    break
        mp = getattr(self, "min_phase", {}).get(b.o.name)
        if mp and mp[0] > phase:
            phase, side = mp
        # przeszkody (teren/podloga) nad spodem deski w spoczynku, dla kazdego w w poprzek
        gfw = [[max(h - b.z0 + EPS, 0.0) for h in r] for r in Hw]
        wg = b.wg
        ssum = sum(w * w for w in wg)
        psi_max = math.radians(p.roll_max)
        bw = max(p.bevel_width, 1e-4)                  # szerokosc (Offset) bevela petli zalamania
        R = bw

        def g_eff(gw, psi):
            """wysokosc spodu na srodku deski potrzebna, by caly przekroj (skrecony o psi) byl nad terenem"""
            t = math.tan(psi)
            return max(g - w * t for g, w in zip(gw, wg))

        def cross_slope(gw):
            """przechyl w poprzek, przy ktorym deska musi sie najmniej podniesc nad teren/podloge:
            na spadku terenu - wzdluz spadku; gdy teren jest tylko pod jedna krawedzia -
            deska obraca sie wokol drugiej krawedzi i podnosi tylko te strone"""
            if max(gw) <= EPS * 1.01 or psi_max <= 0:
                return 0.0
            lim = math.tan(psi_max)
            lo, hi = -lim, lim
            for _ in range(60):                       # funkcja wypukla -> szukanie trojdzielne
                m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
                f1 = max(g - w * m1 for g, w in zip(gw, wg))
                f2 = max(g - w * m2 for g, w in zip(gw, wg))
                if f1 <= f2:
                    hi = m2
                else:
                    lo = m1
            t = (lo + hi) / 2
            if max(g - w * t for g, w in zip(gw, wg)) >= max(gw) - 1e-5:
                return 0.0                            # przechyl nic nie daje
            return math.atan(t)

        def smooth01(t):
            t = max(0.0, min(1.0, t)); return t * t * (3 - 2 * t)

        TH_MAX = math.radians(80)

        def rise(sh, d):
            """deska od gwozdzia: luk o promieniu r do kata th, dalej prosta (d = odleglosc pozioma)"""
            th, r = sh
            xe = r * math.sin(th)
            if d <= xe:
                return r - math.sqrt(max(r * r - d * d, 0.0))
            return r * (1 - math.cos(th)) + (d - xe) * math.tan(th)

        def rise_param(sh, u):
            """to samo po dlugosci deski u -> (odleglosc pozioma, wysokosc)"""
            th, r = sh
            la = r * th
            if u <= la:
                return r * math.sin(u / r), r * (1 - math.cos(u / r))
            return r * math.sin(th) + (u - la) * math.cos(th), r * (1 - math.cos(th)) + (u - la) * math.sin(th)

        def min_angle(samples, reach, psi_c, ramp):
            """najmniejszy kat (i najwiekszy promien <= R), przy ktorym deska (luk + prosta)
            przechodzi nad terenem. Gdy teren tuz za gwozdziem jest za wysoko na luk o promieniu R,
            promien jest zmniejszany - zgiecie robi sie ostrzejsze tylko tam, gdzie musi."""
            def ok(sh):
                for d, gw in samples:
                    if d > reach:
                        continue
                    ps = psi_c * smooth01(d / ramp)
                    if rise(sh, d) < g_eff(gw, ps) - 1e-6:
                        return False
                return True
            r = R
            for _ in range(16):
                if ok((0.0, r)):
                    return (0.0, r)
                if ok((TH_MAX, r)):
                    break
                r *= 0.6
            lo, hi = 0.0, TH_MAX
            for _ in range(40):
                mid = (lo + hi) / 2
                if ok((mid, r)):
                    hi = mid
                else:
                    lo = mid
            return (hi, r)

        N = b.N
        roll = [0.0] * (N + 1)
        def poly_nodes(poly):
            """lamana (s_spoczynkowe, x, z) -> wezly deski w punktach b.sk (sztywne odcinki)"""
            out = []
            for s in b.sk:
                for (s0, x0, z0), (s1, x1, z1) in zip(poly, poly[1:]):
                    if s0 - 1e-9 <= s <= s1 + 1e-9:
                        f = (s - s0) / (s1 - s0) if s1 > s0 else 0.0
                        out.append((x0 + (x1 - x0) * f, z0 + (z1 - z0) * f))
                        break
            return out

        def lift_angle(samples, reach):
            """najmniejszy kat prostej deski od gwozdzia, przy ktorym przechodzi nad terenem"""
            th = 0.0
            for d, gw in samples:
                if 1e-6 < d <= reach:
                    th = max(th, math.atan2(max(gw), d))
            return min(th, math.radians(80))

        L2 = b.L / 2
        kinks = []                                     # (s spoczynkowe, kat zalamania, promien bevela)
        if phase == 2:
            # najwieksze naprezenie przy trzymajacym gwozdziu: tam petla, dalej sztywna prosta
            kH = kA if side == 1 else kB
            sH = b.sk[kH]
            reach = side * ((L2 if side == 1 else -L2) - sH)
            samples = [(side * (s - sH), gfw[j]) for j, s in enumerate(b.sg) if side * (s - sH) > 1e-6]
            th = lift_angle(samples, reach)
            if side == 1:
                poly = [(-L2, -L2, 0.0), (sH, sH, 0.0),
                        (L2, sH + (L2 - sH) * math.cos(th), (L2 - sH) * math.sin(th))]
            else:
                poly = [(-L2, sH - (sH + L2) * math.cos(th), (sH + L2) * math.sin(th)),
                        (sH, sH, 0.0), (L2, L2, 0.0)]
            if th > 1e-4:
                kinks.append((sH, th, bw))
            nodes = poly_nodes(poly)
        elif phase == 3:
            # wyrwana: prosta, sztywna deska oparta o teren pod swoim srodkiem (przechylona w poprzek)
            def rest_line(psi):
                gf = [(s, g_eff(gw, psi)) for s, gw in zip(b.sg, gfw)]
                hl = upper_hull(gf)
                e = next(((u, v) for u, v in zip(hl, hl[1:]) if u[0] <= 0 <= v[0]), (hl[0], hl[-1]))
                m = (e[1][1] - e[0][1]) / (e[1][0] - e[0][0]) if e[1][0] > e[0][0] else 0.0
                return e, m, e[0][1] + m * (0 - e[0][0])
            e, m, z_c = rest_line(0.0)
            psi_c = 0.0
            if psi_max > 0:
                cs = [cross_slope(gfw[min(range(len(b.sg)), key=lambda j: abs(b.sg[j] - pt[0]))]) for pt in e]
                psi_c = sum(cs) / len(cs)
                e, m, z_c = rest_line(psi_c)
            phi = math.atan(m)
            nodes = [((k - N / 2) * ds * math.cos(phi), z_c + (k - N / 2) * ds * math.sin(phi))
                     for k in range(N + 1)]
            roll = [psi_c] * (N + 1)
        else:
            # przybita na obu gwozdziach: petle przy gwozdziach i na wierzcholku nad terenem
            smp = [(s, gfw[j]) for j, s in enumerate(b.sg) if -a < s < a]
            tA = max([math.atan2(max(gw), s + a) for s, gw in smp] + [0.0])
            tB = max([math.atan2(max(gw), a - s) for s, gw in smp] + [0.0])
            tA, tB = min(tA, math.radians(80)), min(tB, math.radians(80))
            if tA + tB < 1e-4:
                poly = [(-L2, -L2, 0.0), (L2, L2, 0.0)]
            else:
                ta, tb = math.tan(tA), math.tan(tB)
                xap = a * (tb - ta) / (ta + tb); zap = (xap + a) * ta
                LA = math.hypot(xap + a, zap); LB = math.hypot(a - xap, zap)
                sap, sB = -a + LA, -a + LA + LB
                poly = [(-L2, -L2, 0.0), (-a, -a, 0.0), (sap, xap, zap)]
                if sB < L2:
                    poly += [(sB, a, 0.0), (L2, a + (L2 - sB), 0.0)]
                else:
                    u = L2 - sap
                    poly += [(L2, xap + u * math.cos(tB), zap - u * math.sin(tB))]
                # bevel na wierzcholku obnizy go - zmniejsz promien, az luk bedzie nad terenem
                alpha = tA + tB
                dA = Vector((math.cos(tA), math.sin(tA))); dB = Vector((math.cos(tB), -math.sin(tB)))
                bis = (dB - dA).normalized()
                r_ap = bw / max(math.tan(alpha / 2), 1e-6)
                for _ in range(16):
                    c = Vector((xap, zap)) + bis * (r_ap / math.cos(alpha / 2))
                    tl = r_ap * math.tan(alpha / 2)
                    ok = True
                    for s, gw in smp:
                        if xap - tl * dA.x <= s <= xap + tl * dB.x and abs(s - c.x) < r_ap:
                            if c.y + math.sqrt(r_ap * r_ap - (s - c.x) ** 2) < max(gw) - 1e-6:
                                ok = False; break
                    if ok:
                        break
                    r_ap *= 0.6
                kinks = [(-a, tA, bw), (sap, alpha, r_ap * math.tan(alpha / 2))] + ([(sB, tB, bw)] if sB < L2 else [])
                kinks = [kk for kk in kinks if kk[1] > 1e-4]
            nodes = poly_nodes(poly)
        b.kinks = kinks
        b.poly = poly if phase in (1, 2) else None
        b.roll = roll
        b.nodes = canon(nodes, ds)
        b.phase = phase
        b.side = side
        b.H = H
        return True

    def lift(self, b, EPS):
        """podniesienie w pionie, zeby deska nie wchodzila w teren ani w podloge (albo w dno / sciane dziury)"""
        lift = 0.0
        F = getattr(self, "F", None)
        void = F is not None and F.mode in ('ZAPADNIECIE', 'WYBUCH')
        for k in range(b.N + 1):
            for w in (-b.W / 2, -b.W / 4, 0.0, b.W / 4, b.W / 2):
                q = b.node_point(k, w)
                if void:
                    lift = max(lift, b.z0 + F.sup_rel(q.x, q.y) - q.z)
                else:
                    lift = max(lift, self.th(q.x, q.y) + EPS - q.z, b.z0 - q.z)
        return lift

    def end_extent(self, b, axis):
        ext = []
        for k in (0, b.N):
            for h in (0.0, b.T):
                for w in (-b.W / 2, b.W / 2):
                    ext.append(b.node_point(k, w, h).dot(axis))
        return min(ext), max(ext)

    # --- zastosowanie wyniku do obiektu
    def apply(self, b, smooth, segments):
        """Deska prosta: tylko transformacja. Wygieta: petla w kazdym miejscu zalamania
        (najwieksze naprezenie), sztywne odcinki wygiete wzgledem petli, petla zbevelowana."""
        o, N = b.o, b.N
        roll = b.roll or [0.0] * (N + 1)
        Tw = Matrix.Translation(b.shift)
        mw = o.matrix_world.copy()

        def rigid(s0, x0, z0, phi, psi=0.0):
            """sztywny ruch: punkt spodu s0 -> (x0, z0), pochylenie phi, przechyl w poprzek psi"""
            G = Matrix.Translation(Vector((x0, 0, z0 - b.T / 2))) @ Matrix.Rotation(-phi, 4, 'Y') \
                @ Matrix.Rotation(psi, 4, 'X') @ Matrix.Translation(Vector((-s0, 0, b.T / 2)))
            return Tw @ b.F @ G @ b.Finv

        if not b.kinks:
            (x0, z0), (x1, z1) = b.nodes[0], b.nodes[-1]
            o.matrix_world = rigid(b.sk[0], x0, z0, math.atan2(z1 - z0, x1 - x0), roll[0]) @ mw
            return "prosta"

        poly = b.poly                                  # (s spoczynkowe, x, z) - wierzcholki lamanej
        phis = [math.atan2(q[2] - p_[2], q[1] - p_[1]) for p_, q in zip(poly, poly[1:])]
        nrm = [Vector((-math.sin(ph), math.cos(ph))) for ph in phis]
        # nowa transformacja obiektu = odcinek, na ktorym lezy srodek deski
        ic = next(i for i in range(len(phis)) if poly[i][0] <= 0 <= poly[i + 1][0] + 1e-9)
        new_mw = rigid(poly[ic][0], poly[ic][1], poly[ic][2], phis[ic]) @ mw
        if o.data.users > 1:
            o.data = o.data.copy()
        bm = bmesh.new(); bm.from_mesh(o.data)
        bm.transform(b.Finv @ mw)                      # -> uklad deski (x wzdluz, z w gore)
        loops = []
        for s_k, ang, rad in b.kinks:                  # petla w miejscu zalamania
            r = bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
                                       plane_co=Vector((s_k, 0, 0)), plane_no=Vector((1, 0, 0)))
            loops.append(([g for g in r["geom_cut"] if isinstance(g, (bmesh.types.BMEdge, bmesh.types.BMVert))],
                          ang, rad))
        kink_idx = {i for i in range(1, len(poly) - 1)}
        for v in bm.verts:
            s, w, z = v.co
            h = z + b.T / 2
            j = next((i for i in kink_idx if abs(s - poly[i][0]) < 1e-5), None)
            if j is not None:                          # wierzcholek petli: styk dwoch odcinkow (ukos)
                nb = (nrm[j - 1] + nrm[j]).normalized()
                q = Vector((poly[j][1], poly[j][2])) + nb * (h / max(nrm[j].dot(nb), 0.2))
            else:
                i = next((i for i in range(len(phis)) if s <= poly[i + 1][0] + 1e-9), len(phis) - 1)
                u = s - poly[i][0]
                q = Vector((poly[i][1] + math.cos(phis[i]) * u, poly[i][2] + math.sin(phis[i]) * u)) + nrm[i] * h
            v.co = Vector((q.x, w, q.y - b.T / 2))
        # bevel petli zalamania (jak Ctrl+B na zaznaczonej petli: Offset, Loop Slide, bez Clamp Overlap,
        # Miter Sharp). Szerokosc przycieta tak, by nie nachodzila na sasiednia petle ani koniec deski.
        stops = sorted([-b.L / 2, b.L / 2] + [k[0] for k in b.kinks])
        for (s_k, ang, width), (geom, _a, _w) in zip(b.kinks, loops):
            edges = [g for g in geom if isinstance(g, bmesh.types.BMEdge) and g.is_valid]
            if not edges or abs(ang) < 1e-4:
                continue
            gap = min(abs(s_k - t) for t in stops if abs(s_k - t) > 1e-6)
            off = min(width, 0.45 * gap)
            bmesh.ops.bevel(bm, geom=edges, offset=off, offset_type='OFFSET', segments=segments,
                            profile=0.5, affect='EDGES', clamp_overlap=False, loop_slide=True,
                            miter_outer='SHARP', miter_inner='SHARP')
        # bevel na krotkiej/skosnej desce potrafi zostawic 5-kat -> czworokat + trojkat
        ngons = [f for f in bm.faces if len(f.verts) > 4]
        if ngons:
            r = bmesh.ops.triangulate(bm, faces=ngons, ngon_method='BEAUTY')
            bmesh.ops.join_triangles(bm, faces=r["faces"], angle_face_threshold=math.radians(10),
                                     angle_shape_threshold=math.pi, cmp_uvs=True, cmp_materials=True)
        if smooth:
            # gladkie cieniowanie zgiecia; krawedzie deski (> 30 st.) zostaja ostre
            for e in bm.edges:
                if len(e.link_faces) == 2 and e.calc_face_angle(0.0) > math.radians(30):
                    e.smooth = False
            for f in bm.faces:
                f.smooth = True
        bm.transform(new_mw.inverted() @ Tw @ b.F)
        bm.to_mesh(o.data); bm.free(); o.data.update()
        o.matrix_world = new_mw
        return "wygieta"

    # --- kolizje deska-deska: pary z glebokoscia przenikania (m)
    def collisions(self, context, objs, changed, thr=0.002):
        dg = context.evaluated_depsgraph_get()
        geo = {}
        for o in objs:
            oe = o.evaluated_get(dg); me = oe.to_mesh()
            vs = [oe.matrix_world @ v.co for v in me.vertices]
            geo[o.name] = (vs, BVHTree.FromPolygons(vs, [tuple(q.vertices) for q in me.polygons]))
            oe.to_mesh_clear()
        dirs = [Vector((0.31, 0.17, 1)).normalized(), Vector((-0.23, 0.41, -1)).normalized(),
                Vector((1, -0.37, 0.13)).normalized()]

        def inside(v, bvh):
            votes = 0
            for d in dirs:
                n, o_ = 0, v.copy()
                while n < 50:
                    hit = bvh.ray_cast(o_, d)
                    if hit[0] is None:
                        break
                    n += 1; o_ = hit[0] + d * 1e-6
                votes += n % 2
            return votes >= 2

        def depth(vs, bvh):
            dd = 0.0
            for v in vs:
                r = bvh.find_nearest(v, 0.3)
                if r[0] is not None and r[3] > dd and inside(v, bvh):
                    dd = r[3]
            return dd
        out = []
        names = [o.name for o in objs]
        for i_, a_ in enumerate(names):
            for b_ in names[i_ + 1:]:
                if a_ not in changed and b_ not in changed:
                    continue
                if geo[a_][1].overlap(geo[b_][1]):
                    d = max(depth(geo[a_][0], geo[b_][1]), depth(geo[b_][0], geo[a_][1]))
                    if d > thr:
                        out.append((a_, b_, d))
        return out

    def settle(self, context, boards, moved, rounds=10):
        """rozsuniecie kolizji: swobodniejsza deska (mniej trzymajacych gwozdzi) podnoszona o glebokosc"""
        objs = [b.o for b in boards if b.phase != 4]
        rank = {b.o.name: b.phase for b in boards}
        changed = {b.o.name for b in moved}
        byname = {o.name: o for o in objs}
        lifted = 0
        for _ in range(rounds):
            context.view_layer.update()
            pairs = self.collisions(context, objs, changed)
            if not pairs:
                break
            done = set()
            for a_, b_, d in sorted(pairs, key=lambda t: -t[2]):
                mv = a_ if rank.get(a_, 0) >= rank.get(b_, 0) else b_
                if rank.get(mv, 0) == 0 or mv in done:
                    continue
                o = byname[mv]
                o.matrix_world = Matrix.Translation((0, 0, d + 0.0015)) @ o.matrix_world
                done.add(mv); lifted += 1
        return lifted

    # --- kontrola wyniku
    def verify(self, context, boards, changed):
        dg = context.evaluated_depsgraph_get()
        geo = {}
        for b in boards:
            oe = b.o.evaluated_get(dg); me = oe.to_mesh()
            vs = [oe.matrix_world @ v.co for v in me.vertices]
            geo[b.o.name] = (vs, BVHTree.FromPolygons(vs, [tuple(p.vertices) for p in me.polygons]))
            oe.to_mesh_clear()
        dirs = [Vector((0.31, 0.17, 1)).normalized(), Vector((-0.23, 0.41, -1)).normalized(),
                Vector((1, -0.37, 0.13)).normalized()]

        def inside(v, bvh):
            votes = 0
            for d in dirs:
                n, o_ = 0, v.copy()
                while n < 50:
                    hit = bvh.ray_cast(o_, d)
                    if hit[0] is None:
                        break
                    n += 1; o_ = hit[0] + d * 1e-6
                votes += n % 2
            return votes >= 2

        def depth(vs, bvh):
            dd = 0.0
            for v in vs:
                r = bvh.find_nearest(v, 0.2)
                if r[0] is not None and inside(v, bvh):
                    dd = max(dd, r[3])
            return dd
        bad, worst = 0, 0.0
        names = [b.o.name for b in boards]
        for i_, a_ in enumerate(names):
            for b_ in names[i_ + 1:]:
                if a_ not in changed and b_ not in changed:
                    continue
                if geo[a_][1].overlap(geo[b_][1]):
                    d = max(depth(geo[a_][0], geo[b_][1]), depth(geo[b_][0], geo[a_][1]))
                    worst = max(worst, d)
                    if d > 0.002:
                        bad += 1
        tbad = 0
        for n in (changed if self.TB is not None else ()):
            for v in geo[n][0]:
                if self.th(v.x, v.y) - v.z > 0.002:
                    # wierzcholek pod powierzchnia terenu - policz tylko wyrazne przypadki
                    r = self.TB.find_nearest(v, 0.2)
                    if r[0] is not None and (v - r[0]).dot(r[1]) < 0 and r[3] > 0.002:
                        tbad += 1; break
        return bad, worst, tbad

    def solve(self, context, p, boards_in=None):
        """stany desek (bez zmiany obiektow): (boards, moved, thrown, skipped) albo (None, poziom, komunikat)"""
        self.moved = []
        terrain, coll = p.terrain, p.boards
        scen = getattr(p, "scenario", 'TEREN')
        self.F = None
        if scen == 'TEREN':
            self.terrain_bvh(context, terrain)
        elif p.emitter is None:
            return 'ERROR', "Wskaz obiekt sterujacy scenariusza (Empty albo siatke)"
        boards, skipped = [], []
        for o in (boards_in if boards_in is not None else coll.all_objects):
            if o.type != 'MESH' or o == terrain or o == p.emitter or not o.data.vertices:
                continue
            if o.data.shape_keys:
                skipped.append(o.name + " (shape keys / animacja)"); continue
            b = Board(o)
            if not b.flat or b.T < 1e-4:
                skipped.append(o.name + (" (nie lezy plasko)" if not b.flat else " (brak grubosci)")); continue
            b.setup(p.segment_length, p.nail_position)
            boards.append(b)
        if not boards:
            return None, 'ERROR', "Brak desek do przetworzenia"
        if scen != 'TEREN':
            from . import scenariusze
            z0s = sorted(b.z0 for b in boards)
            self.F = scenariusze.Field(p, z0s[len(z0s) // 2])
            self.TB = None
            self.th = self.F.th; self.ttop = self.F.ttop; self.tbot = self.F.tbot
        if scen in ('TEREN', 'WILGOC'):
            moved = [b for b in boards if self.simulate(b, p)]
        else:
            moved = [b for b in boards if scenariusze.simulate_void(self, b, p, self.F)]
        self.moved = [b.o for b in moved]
        if not moved:
            return None, 'WARNING', ("Teren nie dochodzi do zadnej deski" if scen == 'TEREN' else
                                     "Scenariusz nie obejmuje zadnej deski (sprawdz polozenie i rozmiar obiektu sterujacego)")
        EPS = self._eps = p.clearance
        thrown = [b for b in moved if b.phase == 4]
        moved = [b for b in moved if b.phase != 4]
        for b in moved:
            if b.phase >= 2:
                b.shift = Vector((0, 0, self.lift(b, EPS)))
        # deski stykajace sie koncami: rozsun wzdluz dlugosci
        pairs = []
        live = [b for b in boards if b.phase != 4]
        for a_ in live:
            for b_ in live:
                if a_ is b_ or (a_.nodes is None and b_.nodes is None):
                    continue
                if abs(a_.es.dot(b_.es)) < 0.995:
                    continue
                if abs((b_.c - a_.c).dot(a_.ew)) > 0.5 * min(a_.W, b_.W):
                    continue
                gap = (b_.c - a_.c).dot(a_.es) - (a_.L + b_.L) / 2
                if -0.002 < gap < 0.03:
                    pairs.append((a_, b_))
        for it in range(10):
            moved_any = False
            for a_, b_ in pairs:
                over = self.end_extent(a_, a_.es)[1] + 0.0004 - self.end_extent(b_, a_.es)[0]
                if over <= 0 or (a_.phase <= 1 and b_.phase <= 1):
                    continue
                if a_.phase > b_.phase:
                    a_.shift -= a_.es * over
                elif b_.phase > a_.phase:
                    b_.shift += a_.es * over
                else:
                    a_.shift -= a_.es * (over / 2); b_.shift += a_.es * (over / 2)
                moved_any = True
            if not moved_any:
                break
        for b in moved:
            if b.phase >= 2:
                b.shift += Vector((0, 0, self.lift(b, EPS)))
        return boards, moved, thrown, skipped

    def run(self, context, p, remove_rigidbody=True):
        """caly etap 1; zwraca (poziom 'INFO'/'WARNING'/'ERROR', komunikat)"""
        res = self.solve(context, p)
        if res[0] is None:
            return res[1], res[2]
        boards, moved, thrown, skipped = res
        scen = getattr(p, "scenario", 'TEREN')
        coll = p.boards
        EPS = p.clearance
        if scen in ('ZAPADNIECIE', 'WYBUCH'):
            from . import scenariusze
        # usun Rigid Body (inaczej symulacja nadpisze ulozenie)
        if p.remove_rigidbody and remove_rigidbody:
            remove_rigid_bodies(context, [b.o for b in boards])
        stats = {"prosta": 0, "wygieta": 0}
        for b in moved:
            # gwozdzie (dla modulu Detale): gora i spod deski w miejscu gwozdzia + czy puscil
            if b.tex_nails:
                # prawdziwe gwozdzie z tekstury; puszczaja wszystkie poza trzymajacym koncem
                def rel(k):
                    if b.phase == 3:
                        return True
                    if b.phase == 2:
                        return k > b.kA if b.side == 1 else k < b.kB
                    return False
                src = [(k, w, rel(k)) for k, w in b.tex_nails]
            else:
                released = {1: set(), 2: {b.kB if b.side == 1 else b.kA}, 3: {b.kA, b.kB}}[b.phase]
                src = [(k, 0.0, k in released) for k in (b.kA, b.kB)]
            nails = [(b.node_point(k, w, b.T), b.node_point(k, w, 0.0), r) for k, w, r in src]
            stats[self.apply(b, p.smooth_shading, p.bevel_segments)] += 1
            inv = b.o.matrix_world.inverted()
            data = []
            for top, bot, rel in nails:
                data += list(inv @ top) + list(inv @ bot) + [1.0 if rel else 0.0]
            b.o["deski_faza"] = ("przybita", "jeden gwozdz", "wyrwana")[b.phase - 1]
            b.o["deski_gwozdzie"] = data                # [gora xyz, spod xyz, puscil] x 2, lokalnie
            b.o["deski_podloga"] = getattr(b, "floor", b.z0)   # najnizsze podparcie pod deska (swiat)
        n_settle = 0
        if getattr(p, "resolve_collisions", False):
            n_settle = self.settle(context, boards, moved)
        if scen in ('ZAPADNIECIE', 'WYBUCH') and p.make_ground:
            scenariusze.build_ground(context, p, self.F, coll)
        for b in thrown:                                # wyrzucone podmuchem: ukryte (do animacji / gruzu)
            b.o.hide_set(True); b.o.hide_render = True
            b.o["deski_faza"] = "wyrzucona"
        msg = (f"Deski: {len(moved)} ruszone (proste: {stats['prosta']}, wygiete: {stats['wygieta']}), "
               f"nietkniete: {len(boards) - len(moved) - len(thrown)}")
        if thrown:
            msg += f", wyrzucone (ukryte): {len(thrown)}"
        if n_settle:
            msg += f", podniesione (kolizje): {n_settle}"
        if skipped:
            msg += f", pominiete: {len(skipped)}"
        if p.verify:
            context.view_layer.update()
            bad, worst, tbad = self.verify(context, boards, {b.o.name for b in moved})
            msg += f" | kontrola: przenikania deska-deska >2mm: {bad} (max {worst * 1000:.1f} mm), w terenie: {tbad}"
        for s in skipped:
            print("Deski - pominieta:", s)
        return ('WARNING' if (skipped or (p.verify and (bad or tbad))) else 'INFO'), msg


def remove_rigid_bodies(context, objs):
    """usun Rigid Body (inaczej symulacja nadpisze ulozenie); dziala tez poza operatorem"""
    vl = context.view_layer
    for o in objs:
        if o.rigid_body and o.name in vl.objects:
            hs = o.hide_get(); o.hide_set(False)
            with context.temp_override(object=o, active_object=o, selected_objects=[o]):
                bpy.ops.rigidbody.object_remove()
            o.hide_set(hs)


class DESKI_OT_deform(DeformCore, bpy.types.Operator):
    bl_idname = "object.deski_teren_deform"
    bl_label = "Wyrwij deski terenem"
    bl_description = ("Wygina i wyrywa deski z kolekcji tak, jak zrobilby to teren wypychajacy "
                      "je od spodu do swojej obecnej (docelowej) pozycji")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        p = context.scene.deski_teren
        pg = getattr(context.scene, "deski_podglad", None)
        src = p.terrain if p.scenario == 'TEREN' else p.emitter
        return src is not None and p.boards is not None and not (pg and pg.active)

    def execute(self, context):
        lvl, msg = self.run(context, context.scene.deski_teren)
        self.report({lvl}, msg)
        return {'CANCELLED'} if lvl == 'ERROR' or not self.moved else {'FINISHED'}


# ---------------------------------------------------------------- ustawienia + panel

def _live(self, context):
    from . import podglad
    podglad.live_update(self, context)


def add_live_updates(cls, skip=()):
    """kazda zmiana ustawienia -> przeliczenie podgladu na zywo (jesli wlaczony)"""
    ann = cls.__annotations__
    for k, d in list(ann.items()):
        if k in skip or not hasattr(d, "keywords") or "update" in d.keywords:
            continue
        ann[k] = d.function(**dict(d.keywords, update=_live))


def mesh_poll(self, o):
    return o.type == 'MESH'


class DeskiSettings(bpy.types.PropertyGroup):
    scenario: bpy.props.EnumProperty(name="Scenariusz", default='TEREN', items=(
        ('TEREN', "Teren", "Teren w pozycji docelowej wypycha deski od spodu"),
        ('WILGOC', "Wilgoc", "Wybrzuszenie od wilgoci: kopula / grzbiet pod deskami (Empty: srodek i promien, "
                             "skala niejednorodna = elipsa / grzbiet)"),
        ('ZAPADNIECIE', "Zapadniecie", "Dziura w podlodze: deski zwisaja z krawedzi albo wpadaja do srodka"),
        ('WYBUCH', "Wybuch", "Krater i podmuch: deski odgiete w gore, najblizsze srodka wyrzucone")))
    emitter: bpy.props.PointerProperty(name="Obszar", type=bpy.types.Object,
                                       description="Empty (srodek, promien = rozmiar Empty) albo siatka (obrys); "
                                                   "przesuwanie / skalowanie widac od razu w Podgladzie na zywo")
    bulge_height: bpy.props.FloatProperty(name="Wysokosc wybrzuszenia", default=0.08, min=0.0, unit='LENGTH')
    hole_depth: bpy.props.FloatProperty(name="Glebokosc", default=0.35, min=0.0, unit='LENGTH',
                                        description="Glebokosc dziury / krateru pod podloga")
    hang_max: bpy.props.FloatProperty(name="Maks. zwis (st.)", default=65.0, min=0.0, max=89.0,
                                      description="Najwiekszy kat, o jaki deska zlamana na krawedzi opada w dziure")
    crater_size: bpy.props.FloatProperty(name="Krater (czesc promienia)", default=0.45, min=0.0, max=1.0,
                                         subtype='FACTOR')
    blast_strength: bpy.props.FloatProperty(name="Sila podmuchu", default=1.0, min=0.0, max=3.0)
    peel_max: bpy.props.FloatProperty(name="Maks. odgiecie (st.)", default=70.0, min=0.0, max=89.0)
    throw_zone: bpy.props.FloatProperty(name="Strefa wyrzucenia", default=0.3, min=0.0, max=1.0, subtype='FACTOR',
                                        description="Deski, na ktore podmuch dziala silniej niz (1 - to), sa wyrzucone (ukryte)")
    anim_step: bpy.props.IntProperty(name="Co ile klatek", default=2, min=1, max=20,
                                     description="Probkowanie symulacji w czasie (pomiedzy - interpolacja liniowa)")
    anim_loops: bpy.props.IntProperty(name="Petle animacji", default=4, min=0, max=16,
                                      description="Dodatkowe rownomierne petle w deskach wyginanych w animacji "
                                                  "(luk przesuwajacy sie po desce potrzebuje geometrii)")
    make_ground: bpy.props.BoolProperty(name="Utworz dno dziury / krateru", default=True,
                                        description="Siatka dna (ten sam ksztalt, o ktory opieraja sie deski) - obiekt "
                                                    "'<kolekcja> - dno', podmieniany przy kazdym przeliczeniu")
    terrain: bpy.props.PointerProperty(name="Teren", type=bpy.types.Object, poll=mesh_poll,
                                       description="Teren w pozycji docelowej (wypycha deski od spodu)")
    boards: bpy.props.PointerProperty(name="Deski", type=bpy.types.Collection,
                                      description="Kolekcja z deskami (kazda deska osobnym obiektem)")
    seed: bpy.props.IntProperty(name="Seed", default=11)
    nail_position: bpy.props.FloatProperty(name="Gwozdzie od konca", default=0.1, min=0.02, max=0.45,
                                           subtype='FACTOR', description="Polozenie gwozdzi jako ulamek dlugosci deski")
    bulge_min: bpy.props.FloatProperty(name="Ugiecie min", default=0.03, min=0.0, unit='LENGTH')
    bulge_max: bpy.props.FloatProperty(name="Ugiecie max", default=0.08, min=0.0, unit='LENGTH',
                                       description="Max wybrzuszenie przybitej deski, po ktorym puszcza gwozdz")
    slope_min: bpy.props.FloatProperty(name="Kat przy gwozdziu min", default=math.radians(18), subtype='ANGLE')
    slope_max: bpy.props.FloatProperty(name="Kat przy gwozdziu max", default=math.radians(32), subtype='ANGLE')
    angle2_min: bpy.props.FloatProperty(name="Kat zerwania 2. gwozdzia min", default=math.radians(22), subtype='ANGLE')
    angle2_max: bpy.props.FloatProperty(name="Kat zerwania 2. gwozdzia max", default=math.radians(40), subtype='ANGLE')
    segment_length: bpy.props.FloatProperty(name="Dlugosc segmentu", default=0.02, min=0.005, unit='LENGTH',
                                            description="Gestosc ciec do obliczen wygiecia (zbedne petle sa potem usuwane)")
    bevel_width: bpy.props.FloatProperty(name="Bevel: szerokosc (Offset)", default=0.25, min=0.001, unit='LENGTH',
                                         description="Offset bevela petli zalamania. Automatycznie zmniejszany, "
                                                     "zeby bevel nie nachodzil na sasiednia petle ani koniec deski")
    bevel_segments: bpy.props.IntProperty(name="Bevel: segmenty", default=6, min=1, max=64)
    steps: bpy.props.IntProperty(name="Kroki symulacji", default=120, min=10, max=2000)
    clearance: bpy.props.FloatProperty(name="Luz", default=0.0015, min=0.0, unit='LENGTH')
    roll_max: bpy.props.FloatProperty(name="Max skret w poprzek (st.)", default=35.0, min=0.0, max=60.0,
                                      description="Deska przechyla sie w poprzek do spadku terenu tam, gdzie na nim lezy. 0 = bez skretu")
    loop_tolerance: bpy.props.FloatProperty(name="Tolerancja petli", default=0.0005, min=0.00001, unit='LENGTH',
                                            description="Petla zostaje tylko, jesli bez niej ksztalt deski zmienilby sie "
                                                        "o wiecej niz tyle. Mniej = gladszy luk, wiecej petli")
    smooth_shading: bpy.props.BoolProperty(name="Gladkie cieniowanie luku", default=True,
                                           description="Wygiete deski: Shade Smooth, krawedzie deski (> 30 st.) ostre")
    straight_tolerance: bpy.props.FloatProperty(name="Prosta ponizej (st.)", default=0.2, min=0.0,
                                                description="Deska wygieta mniej niz tyle stopni zostaje nietknieta siatkowo")
    remove_rigidbody: bpy.props.BoolProperty(name="Usun Rigid Body z desek", default=True)
    resolve_collisions: bpy.props.BoolProperty(name="Rozsun kolizje", default=True,
                                               description="Deski przenikajace sie po wyginaniu: swobodniejsza (mniej "
                                                           "trzymajacych gwozdzi) jest podnoszona o glebokosc przenikania")
    verify: bpy.props.BoolProperty(name="Sprawdz przenikania", default=True)


add_live_updates(DeskiSettings, skip=("boards", "remove_rigidbody", "anim_step", "anim_loops"))


class DESKI_PT_panel(bpy.types.Panel):
    bl_label = "Wyrwanie desek"
    bl_idname = "DESKI_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Deski"
    bl_order = 1

    def draw(self, context):
        from . import podglad
        p = context.scene.deski_teren
        col = self.layout.column()
        col.prop(p, "scenario")
        if p.scenario == 'TEREN':
            col.prop(p, "terrain")
        else:
            box = col.box()
            box.prop(p, "emitter")
            if p.scenario == 'WILGOC':
                box.prop(p, "bulge_height")
            elif p.scenario == 'ZAPADNIECIE':
                box.prop(p, "hole_depth"); box.prop(p, "hang_max"); box.prop(p, "make_ground")
            else:
                box.prop(p, "blast_strength"); box.prop(p, "peel_max"); box.prop(p, "throw_zone")
                box.prop(p, "crater_size"); box.prop(p, "hole_depth"); box.prop(p, "make_ground")
        row = col.row(); row.enabled = not context.scene.deski_podglad.active   # w trakcie podgladu - zablokowane
        row.prop(p, "boards")
        podglad.draw_live(col, context)
        col.separator()
        col.prop(p, "seed"); col.prop(p, "nail_position")
        box = col.box(); box.label(text="Progi zerwania gwozdzi")
        r = box.row(align=True); r.prop(p, "bulge_min"); r.prop(p, "bulge_max")
        r = box.row(align=True); r.prop(p, "slope_min"); r.prop(p, "slope_max")
        r = box.row(align=True); r.prop(p, "angle2_min"); r.prop(p, "angle2_max")
        box = col.box(); box.label(text="Dokladnosc")
        box.prop(p, "segment_length"); box.prop(p, "bevel_width"); box.prop(p, "bevel_segments"); box.prop(p, "roll_max"); box.prop(p, "smooth_shading"); box.prop(p, "steps"); box.prop(p, "clearance")
        col.prop(p, "remove_rigidbody"); col.prop(p, "resolve_collisions"); col.prop(p, "verify")
        from . import animacja
        animacja.draw_anim(col, context)
        col.separator()
        col.operator(DESKI_OT_deform.bl_idname, icon='MOD_SIMPLEDEFORM')


classes = (DeskiSettings, DESKI_OT_deform, DESKI_PT_panel)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.deski_teren = bpy.props.PointerProperty(type=DeskiSettings)


def unregister():
    del bpy.types.Scene.deski_teren
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

