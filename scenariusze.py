# Scenariusze zniszczen dla etapu 1 (oprocz terenu):
#   WILGOC      - wybrzuszenie od wilgoci: kopula / grzbiet pod deskami (dalej jak teren - ten sam algorytm)
#   ZAPADNIECIE - dziura w podlodze: gwozdzie nad dziura puszczaja; deska trzymana z jednej strony
#                 lamie sie na krawedzi i zwisa, bez gwozdzi - spada i opiera sie o dno / krawedz
#   WYBUCH      - krater + podmuch: deski odginane w gore (zawias na gwozdziu dalej od srodka),
#                 najblizsze srodka wyrzucone (ukryte), bez podparcia - wpadaja do krateru
# Obszar = obiekt sterujacy (Empty lub siatka): srodek, promien (rozmiar Empty / wymiary siatki),
# skala niejednorodna daje elipse / grzbiet.
import math, random, zlib
from mathutils import Vector


def _smooth(t):
    t = max(0.0, min(1.0, t)); return t * t * (3 - 2 * t)


class Field:
    """th(x,y) - powierzchnia pchajaca deski od spodu (swiat); sup_rel(x,y) - podparcie wzgledem podlogi
    (0 = podloga, < 0 = dziura / krater); blast(x,y) - sila podmuchu 0..1"""

    def __init__(self, p, floor_z):
        self.mode = p.scenario
        e = p.emitter
        self.Minv = e.matrix_world.inverted()
        if e.type == 'EMPTY':
            self.rx = self.ry = max(e.empty_display_size, 1e-4)
        else:
            vs = [v.co for v in e.data.vertices] if e.type == 'MESH' and e.data.vertices else [Vector()]
            self.rx = max((max(v.x for v in vs) - min(v.x for v in vs)) / 2, 1e-4)
            self.ry = max((max(v.y for v in vs) - min(v.y for v in vs)) / 2, 1e-4)
        self.floor = floor_z
        self.H = p.bulge_height
        self.D = p.hole_depth
        self.crater = p.crater_size
        self.strength = p.blast_strength
        self.ttop = floor_z + (self.H if self.mode == 'WILGOC' else 0.0)
        self.tbot = floor_z - self.D

    def d(self, x, y):
        q = self.Minv @ Vector((x, y, 0.0))
        return math.hypot(q.x / self.rx, q.y / self.ry)

    def th(self, x, y):
        if self.mode == 'WILGOC':
            d = self.d(x, y)
            return self.floor + (self.H * (1 - d * d) ** 2 if d < 1 else 0.0)
        return -1e9

    def sup_rel(self, x, y):
        d = self.d(x, y)
        if self.mode == 'ZAPADNIECIE':                       # plaskie dno, stroma sciana (20% promienia)
            return -self.D * _smooth((1 - d) / 0.2)
        if self.mode == 'WYBUCH':                            # misa krateru
            dc = d / max(self.crater, 1e-3)
            return -self.D * (1 - dc * dc) if dc < 1 else 0.0
        return 0.0

    def blast(self, x, y):
        d = self.d(x, y)
        return min(1.0, self.strength * max(0.0, 1 - d) ** 1.3) if self.mode == 'WYBUCH' else 0.0


def simulate_void(core, b, p, F):
    """ZAPADNIECIE / WYBUCH: stan deski z podparcia (dziura, krater) i podmuchu.
    Ustawia te same pola co DeformCore.simulate (faza, lamana, zalamania, wezly, skret) - reszta
    (petle, bevel, gwozdzie, detale, podglad) dziala bez zmian. Zwraca True, gdy deska sie rusza."""
    from .wyrwanie import upper_hull, canon
    rnd = random.Random(p.seed * 1000003 + zlib.crc32(b.o.name.encode()))
    N, ds, L2 = b.N, b.ds, b.L / 2
    kA, kB = b.kA, b.kB
    bw = max(p.bevel_width, 1e-4)
    psi_max = math.radians(p.roll_max)
    wg = b.wg
    S = [[F.sup_rel(*(b.c + b.es * s + b.ew * w).xy) for w in wg] for s in b.sg]
    lowest = min(min(r) for r in S)
    ctr = b.c.xy
    blast_c = F.blast(*ctr)

    def free(k):
        x, y = (b.c + b.es * b.sk[k]).xy
        if F.sup_rel(x, y) < -0.01:                      # pod gwozdziem nie ma podlogi
            return True
        return F.mode == 'WYBUCH' and F.blast(x, y) > rnd.uniform(0.25, 0.6)
    fA, fB = free(kA), free(kB)
    b.released = {kA} if fA else set()
    b.released |= {kB} if fB else set()
    b.floor = b.z0 + min(0.0, lowest)
    if F.mode == 'WYBUCH' and (blast_c > 1 - p.throw_zone * rnd.uniform(0.8, 1.2)
                                or (fA and fB and F.sup_rel(*ctr) > -0.02)):
        b.phase = 4                                      # wyrzucona podmuchem
        b.kinks, b.poly, b.roll, b.side = [], None, [0.0] * (N + 1), 0
        b.nodes = None
        return True
    if not fA and not fB:
        return False                                     # trzyma sie (np. most nad dziura)

    def poly_nodes(poly):
        out = []
        for s in b.sk:
            for (s0, x0, z0), (s1, x1, z1) in zip(poly, poly[1:]):
                if s0 - 1e-9 <= s <= s1 + 1e-9:
                    f = (s - s0) / (s1 - s0) if s1 > s0 else 0.0
                    out.append((x0 + (x1 - x0) * f, z0 + (z1 - z0) * f)); break
        return out

    roll = [0.0] * (N + 1)
    kinks, poly = [], None
    if fA and fB:
        # spada: prosta deska oparta o dno / krawedz (otoczka wypukla podparcia pod deska), przechyl w poprzek
        ssum = sum(w * w for w in wg)
        sl = [sum(w * g for w, g in zip(wg, r)) / ssum for r in S if min(r) < -0.005]
        psi = max(-psi_max, min(psi_max, math.atan(sum(sl) / len(sl)))) if sl else 0.0
        t = math.tan(psi)
        gf = [(s, max(g - w * t for g, w in zip(r, wg)) + p.clearance) for s, r in zip(b.sg, S)]
        hl = upper_hull(gf)
        e = next(((u, v) for u, v in zip(hl, hl[1:]) if u[0] <= 0 <= v[0]), (hl[0], hl[-1]))
        m = (e[1][1] - e[0][1]) / (e[1][0] - e[0][0]) if e[1][0] > e[0][0] else 0.0
        z_c = e[0][1] + m * (0 - e[0][0])
        phi = math.atan(m)
        nodes = [((k - N / 2) * ds * math.cos(phi), z_c + (k - N / 2) * ds * math.sin(phi)) for k in range(N + 1)]
        roll = [psi] * (N + 1)
        b.phase, b.side = 3, 0
    else:
        side = 1 if fB else -1                           # side=1: wolny koniec po stronie +x, zawias przy kA
        kH = kA if side == 1 else kB
        sH = b.sk[kH]
        if F.mode == 'WYBUCH':
            # odgiecie w gore na gwozdziu dalej od srodka, kat od sily podmuchu przy wolnym koncu
            xe, ye = (b.c + b.es * (L2 if side == 1 else -L2)).xy
            th = math.radians(p.peel_max) * max(F.blast(xe, ye), 0.25 * blast_c) * rnd.uniform(0.6, 1.0)
            se, sgn = sH, 1.0
        else:
            # zwis: deska lezy na podlodze do krawedzi dziury, dalej opada az dotknie sciany / dna
            gmax = [max(r) for r in S]
            idx = [j for j, s in enumerate(b.sg) if side * (s - sH) > 0]
            j_edge = next((j for j in (idx if side == 1 else idx[::-1]) if gmax[j] < -0.005), None)
            if j_edge is None:
                return False
            se = b.sg[j_edge - 1] if side == 1 else b.sg[j_edge + 1]
            th = math.radians(p.hang_max) * rnd.uniform(0.85, 1.0)
            for j, s in enumerate(b.sg):
                dd = side * (s - se)
                if dd > 1e-4:
                    th = min(th, math.atan2(-(gmax[j] - p.clearance), dd))
            th = max(th, 0.0); sgn = -1.0
        a = sgn * th
        if side == 1:
            poly = [(-L2, -L2, 0.0), (se, se, 0.0), (L2, se + (L2 - se) * math.cos(a), (L2 - se) * math.sin(a))]
        else:
            poly = [(-L2, se - (se + L2) * math.cos(a), (se + L2) * math.sin(a)), (se, se, 0.0), (L2, L2, 0.0)]
        if th > 1e-4:
            kinks = [(se, a, bw)]
        nodes = poly_nodes(poly)
        b.phase, b.side = 2, side
    b.kinks, b.poly, b.roll = kinks, poly, roll
    b.nodes = canon(nodes, ds)
    b.H = [0.0] * len(b.sg)
    return True


def build_ground(context, p, F, coll):
    """dno dziury / krateru jako siatka (ten sam ksztalt, o ktory opieraja sie deski). Obiekt
    '<kolekcja> - dno' jest tworzony raz i podmieniany przy kolejnych przeliczeniach."""
    import bpy, bmesh
    name = coll.name + " - dno"
    n = 48
    bm = bmesh.new()
    M = p.emitter.matrix_world
    grid = {}
    for i in range(n + 1):
        for j in range(n + 1):
            u = -1.05 + 2.1 * i / n; v = -1.05 + 2.1 * j / n
            if math.hypot(u, v) > 1.05:
                continue
            q = M @ Vector((u * F.rx, v * F.ry, 0.0))
            grid[i, j] = bm.verts.new((q.x, q.y, F.floor + F.sup_rel(q.x, q.y) - 0.002))
    for i in range(n):
        for j in range(n):
            vs = [grid.get(k) for k in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))]
            if all(vs):
                bm.faces.new(vs)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    for f in bm.faces:
        if f.normal.z < 0:
            f.normal_flip()
        f.smooth = True
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.get(name)
    if ob is None:
        ob = bpy.data.objects.new(name, me)
        context.scene.collection.objects.link(ob)
        ob["deski_scen"] = 1
    else:
        old = ob.data; ob.data = me
        if old.users == 0:
            bpy.data.meshes.remove(old)
    ob.matrix_world = ob.matrix_world.Identity(4)
    return ob
