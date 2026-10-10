# Detale desek: drobne nierownosci nietknietych desek, drzazgi i odksztalcenie wyrwanych desek,
# gwozdzie wyrwane razem z deskami. Niski koszt geometrii: tylko potrzebne petle, bez n-gonow.
import bpy, bmesh, math, random, zlib
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from .przygotowanie import fit_top_uv, unfold_uv, top_outline
from .wyrwanie import Settings

NAIL_MESH = "AxisRupture_Nail"


# ---------------------------------------------------------------- uklad deski
def board_frame(o):
    """macierz: lokalne wspolrzedne obiektu -> uklad deski (x wzdluz, y w poprzek, z w gore, srodek w 0)
    oraz wymiary (L, W, T)"""
    vs = [v.co for v in o.data.vertices]
    mn = Vector([min(v[k] for v in vs) for k in range(3)])
    mx = Vector([max(v[k] for v in vs) for k in range(3)])
    size = mx - mn
    R = Matrix.Identity(4) if size.x >= size.y else Matrix.Rotation(-math.pi / 2, 4, 'Z')
    B = R @ Matrix.Translation(-(mn + mx) / 2)
    L, W = (size.x, size.y) if size.x >= size.y else (size.y, size.x)
    return B, L, W, size.z


def no_ngons(bm):
    """wszystkie n-gony -> trojkaty, potem z powrotem w czworokaty tam, gdzie plasko"""
    ng = [f for f in bm.faces if len(f.verts) > 4]
    if ng:
        r = bmesh.ops.triangulate(bm, faces=ng, quad_method='BEAUTY', ngon_method='BEAUTY')
        bmesh.ops.join_triangles(bm, faces=r["faces"], angle_face_threshold=math.radians(5),
                                 angle_shape_threshold=math.pi, cmp_uvs=True, cmp_materials=True)


def triangulate_ngons(bm, faces):
    ng = [f for f in faces if f.is_valid and len(f.verts) > 4]
    if ng:
        bmesh.ops.triangulate(bm, faces=ng, quad_method='BEAUTY', ngon_method='BEAUTY')


# ---------------------------------------------------------------- UV przy drzazgach i pekniecach
def top_uv(bm, L, p):
    """liniowe UV wierzchu deski (uklad deski), liczone z wierzchu z dala od koncow"""
    uvl = bm.loops.layers.uv.active
    if uvl is None:
        return None
    z_top = max(v.co.z for v in bm.verts)
    lim = L / 2 - min(p.splinter_depth, 0.25 * L) - 0.01
    samples = [((lp.vert.co.x, lp.vert.co.y), tuple(lp[uvl].uv)) for f in bm.faces
               if f.normal.z > 0.99 and all(v.co.z > z_top - 1e-5 and abs(v.co.x) < lim for v in f.verts)
               for lp in f.loops]
    uvf = fit_top_uv(samples)
    return (uvl, uvf, z_top) if uvf else None


def reuv(uv, faces):
    """UV scian po zmianie ksztaltu: wierzch - rzut (tekstura na miejscu), boki / czola - zagiecie wierzchu"""
    if uv:
        uvl, uvf, z_top = uv
        ident = lambda v, is_dir: v.copy()
        bm_faces = next((f for f in faces if f.is_valid), None)
        tops = []
        if bm_faces is not None:
            # aktualny obrys wierzchu (po przesunieciu zebow / peknieciu)
            seen, stack = set(), [f for f in faces if f.is_valid]
            while stack:                               # sciany tych samych bryl (oba kawalki pekniecia)
                f = stack.pop()
                if f in seen:
                    continue
                seen.add(f); stack.extend(g for e in f.edges for g in e.link_faces)
            tops = [f for f in seen if f.normal.z > 0.99 and all(v.co.z > z_top - 1e-4 for v in f.verts)]
        unfold_uv([f for f in faces if f.is_valid], uvl, ident, uvf, z_top, top_outline(tops, ident))


# ---------------------------------------------------------------- drzazgi na koncu deski
def splinter_end(bm, sgn, L, W, T, rng, p):
    """poszarpany koniec deski: petla w glebokosci D, krawedzie czola podzielone na 'zeby',
    zeby cofniete (czasem jeden wystaje) - bez n-gonow (sciany > 4 wierzcholki triangulowane)"""
    D = min(p.splinter_depth, 0.25 * L)
    xe = sgn * L / 2
    uv = top_uv(bm, L, p)                              # UV wierzchu sprzed wyrwania drzazg
    bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
                           plane_co=Vector((xe - sgn * D, 0, 0)), plane_no=Vector((1, 0, 0)))
    # wszystkie krawedzie w poprzek konca deski (wierzch, faza, czolo, spod) - rzedy od gory do dolu
    zone = 0.3 * D
    rows = [e for e in bm.edges
            if all(sgn * (v.co.x - xe) > -zone for v in e.verts)
            and abs(e.verts[0].co.x - e.verts[1].co.x) < 1e-5
            and abs(e.verts[0].co.y - e.verts[1].co.y) > 0.5 * W]
    if len(rows) < 2:
        return
    n = rng.randint(p.splinter_teeth_min, max(p.splinter_teeth_min, p.splinter_teeth_max))
    keys = {e: (round(e.verts[0].co.x, 5), round(e.verts[0].co.z, 5)) for e in rows}
    r = bmesh.ops.subdivide_edges(bm, edges=rows, cuts=n - 1, use_grid_fill=False)
    new = [g for g in r["geom_inner"] if isinstance(g, bmesh.types.BMVert)]
    groups = {}
    for v in new:                                      # nowe wierzcholki wg rzedu (x, z krawedzi)
        k = min(keys.values(), key=lambda kk: abs(kk[0] - v.co.x) + abs(kk[1] - v.co.z))
        groups.setdefault(k, []).append(v)
    ordered = [sorted(g, key=lambda v: v.co.y) for k, g in sorted(groups.items(), key=lambda kv: -kv[0][1])]
    if len({len(g) for g in ordered}) != 1:
        return
    for up, lo in zip(ordered, ordered[1:]):           # kolumny zebow: kazda sciana konca -> czworokaty
        for a, b in zip(up, lo):
            if set(a.link_faces) & set(b.link_faces):
                bmesh.ops.connect_verts(bm, verts=[a, b])
    cols = list(zip(*ordered))
    out_i = rng.randrange(len(cols)) if rng.random() < p.splinter_out else -1
    touched = set()
    for i, col in enumerate(cols):
        if i == out_i:
            dx = sgn * rng.uniform(0.15, 0.5) * D      # drzazga wystajaca poza koniec
        else:
            dx = -sgn * rng.uniform(0.15, 0.9) * D     # wyrwany kawalek
        k_bottom = rng.uniform(0.6, 1.0)               # spod rwie sie inaczej niz wierzch
        zmax = max(v.co.z for v in col); zmin = min(v.co.z for v in col)
        for v in col:
            f = (v.co.z - zmin) / max(zmax - zmin, 1e-9)
            v.co.x += dx * (k_bottom + (1 - k_bottom) * f)
            touched |= set(v.link_faces)
    reuv(uv, touched)                                  # tekstura zostaje na miejscu - kawalek jest wyrwany
    triangulate_ngons(bm, touched)


# ---------------------------------------------------------------- zwichrowanie / odksztalcenie
class Warp:
    """trwale odksztalcenie deski: luk wzdluz (strzalka sag) + skret (kat tw na koncach).
    Ta sama funkcja przesuwa wierzcholki i punkty gwozdzi."""
    def __init__(self, L, sag, tw):
        self.L, self.sag, self.tw = L, sag, tw

    def __call__(self, co):
        x, y, z = co
        t = max(-1.0, min(1.0, 2 * x / self.L))
        a = self.tw * t                                # skret narasta od srodka ku koncom
        return Vector((x, y * math.cos(a) - z * math.sin(a),
                       y * math.sin(a) + z * math.cos(a) + self.sag * (1 - t * t)))


def add_loops(bm, L, n, margin, avoid=()):
    """n petli w poprzek deski, rowno rozlozonych, z dala od koncow i od stref 'avoid' (x0, promien)"""
    lo, hi = -L / 2 + margin, L / 2 - margin
    if n <= 0 or hi <= lo:
        return
    for i in range(n):
        x = lo + (hi - lo) * (i + 1) / (n + 1)
        for x0, r in avoid:
            if abs(x - x0) < r:
                x = x0 + math.copysign(r, x - x0 if x != x0 else 1.0)
        if lo < x < hi:
            bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
                                   plane_co=Vector((x, 0, 0)), plane_no=Vector((1, 0, 0)))


def warp_board(bm, warp):
    for v in bm.verts:
        v.co = warp(v.co)
    triangulate_ngons(bm, bm.faces[:])


# ---------------------------------------------------------------- spalenie / zbutwienie
ATTR = {"SPALENIE": "deski_spalenie", "BUTWIENIE": "deski_butwienie"}


def _sm(t):
    t = max(0.0, min(1.0, t)); return t * t * (3 - 2 * t)


def decay_mask(p, pos):
    """sila spalenia / zbutwienia 0..1 w punkcie swiata: obszar z Empty (poszarpany brzeg z szumu)
    albo losowe plamy o zadanym pokryciu"""
    from mathutils import noise
    sc = max(p.decay_scale, 1e-3)
    n = noise.noise(pos / sc + Vector((p.seed * 1.37, p.seed * 0.71, 0.0)))
    if p.decay_mode == 'OBSZAR':
        e = p.decay_source
        if e is None:
            return 0.0
        q = e.matrix_world.inverted() @ pos
        R = e.empty_display_size if e.type == 'EMPTY' else max(max(e.dimensions.x, e.dimensions.y) / 2, 1e-3)
        d = math.hypot(q.x, q.y) / max(R, 1e-4) + 0.35 * n
        return _sm((1 - d) / 0.35)
    return _sm(((n + 1) / 2 - (1 - p.decay_coverage)) / 0.2)


def decay_material(mat, kind):
    """kopia materialu z przyciemnieniem (zweglenie / butwienie) i wieksza szorstkoscia wg atrybutu -
    oryginalny material zostaje nietkniety"""
    if mat is None or not mat.use_nodes:
        return mat
    name = f"{mat.name} ({'spalenie' if kind == 'SPALENIE' else 'butwienie'})"
    m2 = bpy.data.materials.get(name)
    if m2 is not None:
        return m2
    m2 = mat.copy(); m2.name = name; m2["deski_mat_scen"] = 1
    nt = m2.node_tree
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if bsdf is None:
        return m2
    at = nt.nodes.new('ShaderNodeAttribute'); at.attribute_type = 'GEOMETRY'; at.attribute_name = ATTR[kind]
    at.location = bsdf.location + Vector((-700, 300))
    col = (0.012, 0.010, 0.009, 1.0) if kind == 'SPALENIE' else (0.13, 0.13, 0.10, 1.0)
    for sock_name, dtype, val in (("Base Color", 'RGBA', col), ("Roughness", 'FLOAT', 0.95)):
        inp = bsdf.inputs.get(sock_name)
        if inp is None:
            continue
        mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = dtype
        mix.location = bsdf.location + Vector((-300, 200 if dtype == 'RGBA' else -50))
        A = next(s for s in mix.inputs if s.name == "A" and s.enabled)
        Bs = next(s for s in mix.inputs if s.name == "B" and s.enabled)
        out = next(s for s in mix.outputs if s.enabled)
        fac = next(s for s in mix.inputs if s.name == "Factor" and s.enabled)
        if inp.is_linked:
            nt.links.new(inp.links[0].from_socket, A)
        else:
            A.default_value = inp.default_value
        Bs.default_value = val
        nt.links.new(at.outputs["Fac"], fac)
        nt.links.new(out, inp)
    return m2


# ---------------------------------------------------------------- pekniecie deski
def crack_board(bm, L, W, T, rng, p):
    """pekniecie w poprzek po poszarpanej linii: deska dzieli sie na dwa zamkniete kawalki,
    kawalek od strony +x odgina sie na zawiasie przy gornej krawedzi (szpara otwiera sie od spodu).
    Zwraca funkcje przeksztalcajaca punkty (dla gwozdzi) albo None."""
    x0 = rng.uniform(-0.2, 0.2) * L
    amp = min(p.crack_jag, 0.15 * L)
    uv = top_uv(bm, L, p)                              # UV wierzchu sprzed pekniecia
    r = bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
                               plane_co=Vector((x0, 0, 0)), plane_no=Vector((1, 0, 0)))
    on = lambda v: abs(v.co.x - x0) < 1e-6
    rows = [e for e in r["geom_cut"] if isinstance(e, bmesh.types.BMEdge)
            and abs(e.verts[0].co.y - e.verts[1].co.y) > 0.5 * W]
    n = rng.randint(3, 6)
    if rows:
        bmesh.ops.subdivide_edges(bm, edges=rows, cuts=n - 1, use_grid_fill=False)
    ring = [e for e in bm.edges if on(e.verts[0]) and on(e.verts[1])]
    ring_v = {v for e in ring for v in e.verts}
    if len(ring) < 4:
        return None
    # poszarpana linia: przesuniecie wzdluz deski zalezne od polozenia w poprzek
    ys = sorted({round(v.co.y, 5) for v in ring_v})
    off = {y: rng.uniform(-1, 1) * amp for y in ys}
    for v in sorted(ring_v, key=lambda v: (round(v.co.y, 5), round(v.co.z, 5), round(v.co.x, 5))):  # stala kolejnosc losowan
        v.co.x += off[round(v.co.y, 5)] + rng.uniform(-0.1, 0.1) * amp
    bmesh.ops.split_edges(bm, edges=ring)
    bnd = [e for e in bm.edges if e.is_boundary]
    res = bmesh.ops.holes_fill(bm, edges=bnd, sides=0)
    # UV: sciany pekniecia i sciany przy poszarpanej linii - z wierzchu deski (bez sciagniecia)
    touched = set(res["faces"])
    for e in bm.edges:
        if e.is_valid and abs(e.verts[0].co.x - x0) <= 1.3 * amp and abs(e.verts[1].co.x - x0) <= 1.3 * amp:
            touched |= set(e.link_faces)
    reuv(uv, touched)
    triangulate_ngons(bm, res["faces"])
    # kawalek po stronie +x (spojna czesc zawierajaca najdalszy wierzcholek)
    start = max(bm.verts, key=lambda v: v.co.x)
    part, stack = set(), [start]
    while stack:
        v = stack.pop()
        if v in part:
            continue
        part.add(v); stack.extend(e.other_vert(v) for e in v.link_edges)
    if len(part) == len(bm.verts):
        return None
    zp = max(v.co.z for v in part)
    gap = rng.uniform(0.002, 0.006)
    ang = -math.radians(p.crack_angle) * rng.uniform(0.3, 1.0)       # odgiecie w gore
    piv = Vector((x0 - amp, 0, zp))
    R = Matrix.Translation(piv + Vector((gap, 0, 0))) @ Matrix.Rotation(ang, 4, 'Y') @ Matrix.Translation(-piv)
    for v in part:
        v.co = R @ v.co
    # zeby poszarpanej linii nie moga na siebie nachodzic - odsuwaj kawalek, az sie rozdziela
    pf = [f for f in bm.faces if f.verts[0] in part]
    of = [f for f in bm.faces if f.verts[0] not in part]

    def tree(fs):
        vs, idx, polys = [], {}, []
        for f in fs:
            for v in f.verts:
                if v not in idx:
                    idx[v] = len(vs); vs.append(v.co.copy())
            polys.append([idx[v] for v in f.verts])
        return BVHTree.FromPolygons(vs, polys)
    t_other = tree(of)
    shift = 0.0
    while tree(pf).overlap(t_other):
        if shift > 2 * amp + 0.02:
            return None                                # nie da sie rozdzielic - bez pekniecia
        for v in part:
            v.co.x += 0.003
        shift += 0.003
    if any(not e.is_manifold for e in bm.edges):
        return None
    R = Matrix.Translation((shift, 0, 0)) @ R
    split_x = x0
    return (lambda co: (R @ co) if co.x > split_x else co), (x0, amp + gap + shift + 0.02)


def mesh_ok(me):
    """zamknieta bryla (kazda krawedz miedzy 2 scianami) i bez przeciec siatki samej ze soba"""
    bm = bmesh.new(); bm.from_mesh(me)
    ok = all(e.is_manifold for e in bm.edges)
    if ok:
        bmesh.ops.triangulate(bm, faces=bm.faces[:]); bm.faces.ensure_lookup_table()
        t = BVHTree.FromBMesh(bm)
        ok = not any(i < j and not (set(bm.faces[i].verts) & set(bm.faces[j].verts)) for i, j in t.overlap(t))
    bm.free()
    return ok


# ---------------------------------------------------------------- gwozdz (wspolna siatka)
def nail_mesh(p):
    me = bpy.data.meshes.get(NAIL_MESH)
    if me:
        return me
    bm = bmesh.new()
    seg, rh, hh, rs, ln = 6, p.nail_head / 2, 0.0012, p.nail_diameter / 2, p.nail_length

    def ring(r, z):
        return [bm.verts.new((r * math.cos(2 * math.pi * i / seg), r * math.sin(2 * math.pi * i / seg), z))
                for i in range(seg)]
    r0, r1, r2, r3 = ring(rh, 0.0), ring(rh, -hh), ring(rs, -hh), ring(rs, -ln + 0.004)
    tip = bm.verts.new((0, 0, -ln))
    top = bm.faces.new(r0)                             # leb (gora gwozdzia w 0, trzon w -Z)
    for a, b in ((r0, r1), (r2, r3)):
        for i in range(seg):
            bm.faces.new((a[i], a[(i + 1) % seg], b[(i + 1) % seg], b[i]))
    for i in range(seg):                               # spod lba -> trzon (pierscien z czworokatow)
        bm.faces.new((r1[i], r1[(i + 1) % seg], r2[(i + 1) % seg], r2[i]))
    for i in range(seg):                               # czubek
        bm.faces.new((r3[i], r3[(i + 1) % seg], tip))
    bmesh.ops.triangulate(bm, faces=[top], quad_method='BEAUTY', ngon_method='BEAUTY')
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    me = bpy.data.meshes.new(NAIL_MESH)
    bm.to_mesh(me); bm.free()
    mat = bpy.data.materials.get("AxisRupture_NailMetal")
    if mat is None:                                    # prosty, ciemny, lekko skorodowany metal
        mat = bpy.data.materials.new("AxisRupture_NailMetal")
        mat.use_nodes = True
        bsdf = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.12, 0.1, 0.09, 1.0)
            bsdf.inputs["Metallic"].default_value = 0.9
            bsdf.inputs["Roughness"].default_value = 0.55
        mat.diffuse_color = (0.12, 0.1, 0.09, 1.0)
    me.materials.append(mat)
    return me


# ---------------------------------------------------------------- operator
class DetailCore:
    def run(self, context, p, coll=None):
        """caly etap 3; zwraca (poziom, komunikat)"""
        coll = coll or p.boards or context.scene.deski_teren.boards
        pw = context.scene.deski_teren
        terrain = pw.terrain if pw.scenario == 'TEREN' else None
        self.TB = None
        if terrain:
            dg = context.evaluated_depsgraph_get()
            te = terrain.evaluated_get(dg); me = te.to_mesh()
            vs = [te.matrix_world @ v.co for v in me.vertices]
            self.TB = BVHTree.FromPolygons(vs, [tuple(q.vertices) for q in me.polygons])
            self.ttop = max(v.z for v in vs)
            te.to_mesh_clear()
        boards = [o for o in coll.all_objects if o.type == 'MESH' and o.data.vertices and not o.data.shape_keys
                  and not o.name.startswith(NAIL_MESH) and o.dimensions.z > 1e-4]
        from .wyrwanie import remember_originals
        remember_originals(boards)
        st = {"nierownosci": 0, "drzazgi": 0, "odksztalcone": 0, "pekniete": 0, "gwozdzie": 0, "pominiete": 0,
              "rozklad": 0}
        backup = {}                                    # deska -> (siatka, macierz) sprzed detali
        for o in boards:
            if o.get("deski_detale"):
                st["pominiete"] += 1
                continue
            rng = random.Random(p.seed * 7919 + zlib.crc32(o.name.encode()))
            faza = o.get("deski_faza")
            if faza is None and (p.use_jitter or p.use_twist_nailed):
                backup[o] = (o.data.copy(), o.matrix_world.copy())
                self.jitter(o, rng, p); st["nierownosci"] += 1
            elif faza == "wyrwana" and (p.use_splinters or p.use_warp or p.use_crack):
                backup[o] = (o.data.copy(), o.matrix_world.copy())
                s, w, c = self.damage(o, rng, p)
                if c and not mesh_ok(o.data):
                    # pekniecie zepsulo siatke - powtorz bez pekniecia (z kopii sprzed detali)
                    bad = o.data
                    o.data = backup[o][0].copy(); o.matrix_world = backup[o][1]
                    if bad.users == 0:
                        bpy.data.meshes.remove(bad)
                    rng = random.Random(p.seed * 7919 + zlib.crc32(o.name.encode()))
                    s, w, c = self.damage(o, rng, p, allow_crack=False)
                st["drzazgi"] += s; st["odksztalcone"] += w; st["pekniete"] += c
            o["deski_detale"] = 1
        # spalenie / zbutwienie (wszystkie deski w obszarze, takze nietkniete)
        if p.use_decay:
            for o in boards:
                if o.get("deski_faza") == "wyrzucona" or o.hide_render or o.get("deski_rozklad"):
                    continue
                rng = random.Random(p.seed * 15485863 + zlib.crc32(o.name.encode()))
                had = o in backup
                if not had:
                    backup[o] = (o.data.copy(), o.matrix_world.copy())
                if self.decay(o, rng, p):
                    st["rozklad"] += 1; o["deski_rozklad"] = 1
        # detale nie moga wcisnac deski w sasiednia - takie zmiany sa cofane
        context.view_layer.update()
        reverted = 0
        for o in self.penetrating(boards, set(backup)):
            me, mw = backup[o]
            old = o.data
            o.data = me; o.matrix_world = mw
            if old is not me and old.users == 0:
                bpy.data.meshes.remove(old)
            reverted += 1
        for o, (me, mw) in backup.items():             # usun niepotrzebne kopie
            if me is not o.data and me.users == 0:
                bpy.data.meshes.remove(me)
        # gwozdzie na koncu - przypiete do desek w ostatecznym polozeniu
        nails_coll = None
        if p.use_nails:
            for o in boards:
                if o.get("deski_gwozdzie") and not o.get("deski_gwozdzie_dodane"):
                    if nails_coll is None:
                        nails_coll = bpy.data.collections.get(coll.name + " - nails")
                        if nails_coll is None:
                            nails_coll = bpy.data.collections.new(coll.name + " - nails")
                            context.scene.collection.children.link(nails_coll)
                    rng = random.Random(p.seed * 104729 + zlib.crc32(o.name.encode()))
                    st["gwozdzie"] += self.nails(o, rng, p, nails_coll)
                    o["deski_gwozdzie_dodane"] = 1
        msg = (f"Details: unevenness {st['nierownosci']}, splinters (ends) {st['drzazgi']}, "
               f"odksztalcone {st['odksztalcone']}, cracked {st['pekniete']}, nails {st['gwozdzie']}")
        if st["rozklad"]:
            msg += f", {'spalone' if p.decay_type == 'SPALENIE' else 'zbutwiale'} {st['rozklad']}"
        if reverted:
            msg += f", reverted (collision with a neighbouring plank) {reverted}"
        if st["pominiete"]:
            msg += f", skipped (already have details) {st['pominiete']}"
        return 'INFO', msg

    # --- kontrola: zmienione deski, ktore przenikaja sasiednie o wiecej niz 1 mm
    def penetrating(self, boards, changed):
        dg = bpy.context.evaluated_depsgraph_get()
        geo = {}
        for o in boards:
            oe = o.evaluated_get(dg); me = oe.to_mesh()
            vs = [oe.matrix_world @ v.co for v in me.vertices]
            geo[o] = (vs, BVHTree.FromPolygons(vs, [tuple(q.vertices) for q in me.polygons]))
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
                if r[0] is not None and r[3] > dd and inside(v, bvh):
                    dd = r[3]
            return dd
        bad = set()
        for a in changed:
            for b in boards:
                if b is a or (b in changed and b.name < a.name):
                    continue
                if not geo[a][1].overlap(geo[b][1]):
                    continue
                if max(depth(geo[a][0], geo[b][1]), depth(geo[b][0], geo[a][1])) > 0.001:
                    bad.add(a)
                    if b in changed:
                        bad.add(b)
        return bad

    # --- teren / podloga
    def th(self, x, y):
        if not self.TB:
            return -1e9
        hit = self.TB.ray_cast(Vector((x, y, self.ttop + 1.0)), Vector((0, 0, -1)))
        return hit[0].z if hit[0] else -1e9

    def lift_clear(self, o, floor):
        """podnies obiekt, zeby zaden wierzcholek nie byl pod terenem / podloga"""
        mw = o.matrix_world
        need = 0.0
        for v in o.data.vertices:
            q = mw @ v.co
            need = max(need, self.th(q.x, q.y) + 0.0015 - q.z, (floor if floor is not None else -1e9) - q.z)
        if need > 0:
            o.matrix_world = Matrix.Translation((0, 0, need)) @ mw

    # --- 5: nierownosci nietknietych desek
    def jitter(self, o, rng, p):
        B, L, W, T = board_frame(o)
        mw = o.matrix_world
        vs = [mw @ v.co for v in o.data.vertices]
        z_min = min(v.z for v in vs)
        Bw = mw @ B.inverted()                         # uklad deski -> swiat
        pivot = Bw @ Vector((0, 0, -T / 2))
        on = 1.0 if p.use_jitter else 0.0
        yaw = on * rng.uniform(-1, 1) * p.jitter_shift / max(L / 2, 1e-6)   # koniec przesuwa sie max o jitter_shift
        roll = on * math.radians(p.jitter_tilt) * rng.uniform(-1, 1)
        pitch = on * math.radians(p.jitter_tilt) * rng.uniform(-1, 1) * min(1.0, W / L)
        Rb = Bw.to_3x3().normalized()
        R = (Rb @ Matrix.Rotation(yaw, 3, 'Z') @ Matrix.Rotation(roll, 3, 'X') @ Matrix.Rotation(pitch, 3, 'Y')
             @ Rb.inverted()).to_4x4()
        M = Matrix.Translation(pivot) @ R @ Matrix.Translation(-pivot)
        short = L < bpy.context.scene.deski_teren.bend_min_length
        if p.use_twist_nailed and p.twist_nailed > 0 and not short:
            # lekkie zwichrowanie przybitej deski (unosza sie narozniki) - 2 petle
            if o.data.users > 1:
                o.data = o.data.copy()
            bm = bmesh.new(); bm.from_mesh(o.data); bm.transform(B)
            add_loops(bm, L, 2, 0.05 * L)
            warp_board(bm, Warp(L, 0.0, math.radians(p.twist_nailed) * rng.uniform(-1, 1)))
            bm.transform(B.inverted())
            bm.to_mesh(o.data); bm.free(); o.data.update()
        new = M @ mw
        z2 = min((new @ v.co).z for v in o.data.vertices)
        o.matrix_world = Matrix.Translation((0, 0, z_min - z2 + on * rng.uniform(0, p.jitter_lift))) @ new

    # --- 6 + odksztalcenie + pekniecie: wyrwane deski
    def damage(self, o, rng, p, allow_crack=True):
        B, L, W, T = board_frame(o)
        if o.data.users > 1:
            o.data = o.data.copy()
        bm = bmesh.new(); bm.from_mesh(o.data)
        bm.transform(B)
        # gwozdzie przesuwaja sie razem z deska (te same przeksztalcenia co wierzcholki)
        nd = list(o.get("deski_gwozdzie", []))
        npts = [B @ Vector(nd[i:i + 3]) for i in range(0, len(nd), 7)] + \
               [B @ Vector(nd[i + 3:i + 6]) for i in range(0, len(nd), 7)]
        n_spl = cracked = warped = 0
        if p.use_splinters:
            for sgn in (-1, 1):
                if rng.random() < p.splinter_chance:
                    splinter_end(bm, sgn, L, W, T, rng, p); n_spl += 1
        end_margin = (min(p.splinter_depth, 0.25 * L) + 0.01) if n_spl else 0.02
        avoid = []
        if allow_crack and p.use_crack and L > p.crack_min_length and rng.random() < p.crack_chance:
            trial = bm.copy()                          # pekniecie na kopii - gdy sie nie uda, deska zostaje cala
            res = crack_board(trial, L, W, T, rng, p)
            if res:
                bm.free(); bm = trial
            else:
                trial.free()
            if res:
                f, zone = res
                npts = [f(q) for q in npts]; cracked = 1
                avoid = [zone]                          # petle odksztalcenia omijaja linie pekniecia
        if p.use_warp and L >= bpy.context.scene.deski_teren.bend_min_length:
            add_loops(bm, L, p.warp_loops, end_margin, avoid=avoid)
            warp = Warp(L, rng.uniform(0.3, 1.0) * p.warp_bow * L * rng.choice((-1, 1)),
                        math.radians(p.warp_twist) * rng.uniform(-1, 1))
            warp_board(bm, warp)
            npts = [warp(q) for q in npts]; warped = 1
        no_ngons(bm)
        bm.transform(B.inverted())
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        bm.to_mesh(o.data); bm.free(); o.data.update()
        if nd:
            Bi = B.inverted(); k = len(npts) // 2
            for j, i in enumerate(range(0, len(nd), 7)):
                nd[i:i + 3] = list(Bi @ npts[j]); nd[i + 3:i + 6] = list(Bi @ npts[k + j])
            o["deski_gwozdzie"] = nd
        self.lift_clear(o, o.get("deski_podloga"))
        return n_spl, warped, cracked

    # --- spalenie / zbutwienie
    def decay(self, o, rng, p):
        """krawedzie zjedzone do srodka (nieregularnie), plaska deska ciensza, konce poszarpane,
        atrybut 0..1 do materialu. Petle dodawane tylko tam, gdzie efekt siega."""
        from mathutils import noise
        B, L, W, T = board_frame(o)
        mw = o.matrix_world
        toW = mw @ B.inverted()
        ns = max(3, int(L / 0.02))
        xs = [-L / 2 + L * i / (ns - 1) for i in range(ns)]
        ms = [decay_mask(p, toW @ Vector((x, 0, 0))) for x in xs]
        if max(ms) < 0.05:
            return False
        if o.data.users > 1:
            o.data = o.data.copy()
        bm = bmesh.new(); bm.from_mesh(o.data); bm.transform(B)
        zmin = min(v.co.z for v in bm.verts); zmax = max(v.co.z for v in bm.verts)
        flat = (zmax - zmin) < T * 1.05 + 1e-5
        uv = top_uv(bm, L, p) if flat else None
        # konce mocno objete efektem - poszarpane (jak drzazgi)
        for sgn, m_end in ((-1, ms[0]), (1, ms[-1])):
            if m_end > 0.35 and flat:
                q = Settings(p, splinter_depth=min(0.25 * L, p.splinter_depth * (0.6 + m_end)), splinter_out=0.0)
                splinter_end(bm, sgn, L, W, T, rng, q)
        # petle w strefie efektu (co decay_seg), z dala od istniejacych
        have = sorted({round(v.co.x, 4) for v in bm.verts})
        seg = max(p.decay_seg, 0.005)
        x = -L / 2 + seg
        while x < L / 2 - seg * 0.5:
            i = min(ns - 1, max(0, int(round((x + L / 2) / L * (ns - 1)))))
            if ms[i] > 0.08 and all(abs(x - h) > 0.4 * seg for h in have):
                bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
                                       plane_co=Vector((x, 0, 0)), plane_no=Vector((1, 0, 0)))
                have.append(x)
            x += seg
        triangulate_ngons(bm, bm.faces[:])
        # krawedzie zjedzone do srodka: przesuniecie malejace ku srodkowi (bez odwracania kolejnosci)
        dmax = min(p.decay_erode, 0.3 * W)
        band = 2.0 * max(dmax, 1e-4)
        vals = {}
        for v in bm.verts:
            wp = toW @ v.co
            m = decay_mask(p, wp)
            nz = noise.noise(wp * 20.0)
            vals[v] = (m, nz)
            if m <= 0:
                continue
            ay = abs(v.co.y)
            f = _sm((ay - (W / 2 - band)) / band)
            v.co.y -= math.copysign(dmax * m * (0.55 + 0.45 * nz) * f, v.co.y)
        if uv:
            reuv(uv, bm.faces[:])
        # plaska deska: ciensza (gorna czesc sciskana ku spodowi - kolejnosc warstw zachowana)
        if flat and p.decay_thin > 0:
            for v, (m, nz) in vals.items():
                if m > 0 and v.is_valid:
                    k = 1 - p.decay_thin * m * (0.6 + 0.4 * nz)
                    v.co.z = zmin + (v.co.z - zmin) * k
        no_ngons(bm)
        bm.transform(B.inverted())
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        bm.to_mesh(o.data); bm.free(); o.data.update()
        # atrybut: mocniej przy krawedziach i koncach
        me = o.data
        name = ATTR[p.decay_type]
        att = me.attributes.get(name) or me.attributes.new(name, 'FLOAT', 'POINT')
        out = []
        for v in me.vertices:
            q = B @ v.co
            wp = mw @ v.co
            m = decay_mask(p, wp)
            edge = 1 - min(1.0, min(W / 2 - abs(q.y), L / 2 - abs(q.x)) / (0.25 * W))
            out.append(max(0.0, min(1.0, m * (0.75 + 0.35 * edge) + 0.15 * noise.noise(wp * 8.0) * m)))
        att.data.foreach_set("value", out)
        if p.decay_material:
            for i, mat in enumerate(me.materials):
                me.materials[i] = decay_material(mat, p.decay_type)
        return True

    # --- 7: gwozdzie wyrwane razem z deska (wystaja spod deski)
    def nails(self, o, rng, p, coll):
        me = nail_mesh(p)
        d = list(o["deski_gwozdzie"])
        mw = o.matrix_world
        floor = o.get("deski_podloga")
        made = 0
        for i in range(0, len(d), 7):
            top, bot, rel = Vector(d[i:i + 3]), Vector(d[i + 3:i + 6]), d[i + 6]
            if rel < 0.5 or rng.random() > p.nail_chance:
                continue
            down = (bot - top).normalized()
            # odchylenie gwozdzia (wyrwany zwykle stoi krzywo)
            ax = down.orthogonal().normalized()
            ax = Matrix.Rotation(rng.uniform(0, 2 * math.pi), 3, down) @ ax
            down2 = Matrix.Rotation(math.radians(p.nail_tilt) * rng.uniform(0.2, 1.0), 3, ax) @ down
            rot = (-down2).to_track_quat('Z', 'Y').to_matrix().to_4x4()
            M = mw @ Matrix.Translation(top) @ rot @ Matrix.Rotation(rng.uniform(0, 2 * math.pi), 4, 'Z')
            tip = M @ Vector((0, 0, -p.nail_length))
            if tip.z < max(self.th(tip.x, tip.y), floor if floor is not None else -1e9):
                continue                               # czubek bylby w terenie / podlodze - pomijamy
            nob = bpy.data.objects.new(o.name + "_Gwozdz", me)
            coll.objects.link(nob)
            nob["deski_gwozdz"] = 1
            nob.parent = o
            nob.matrix_world = M
            made += 1
        return made


class DESKI_OT_details(DetailCore, bpy.types.Operator):
    bl_idname = "object.axis_rupture_detale"
    bl_label = "Add Details"
    bl_description = ("Unevenness of intact planks, splinters and deformation of torn planks, "
                      "nails torn out with the planks")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        pg = getattr(context.scene, "deski_podglad", None)
        return ((context.scene.deski_detale.boards or context.scene.deski_teren.boards) is not None
                and not (pg and pg.active))

    def execute(self, context):
        lvl, msg = self.run(context, context.scene.deski_detale)
        self.report({lvl}, msg)
        return {'FINISHED'}


# ---------------------------------------------------------------- ustawienia + panel
class DeskiDetailSettings(bpy.types.PropertyGroup):
    boards: bpy.props.PointerProperty(name="Planks", type=bpy.types.Collection,
                                      description="Empty = collection from the Tear Planks panel")
    seed: bpy.props.IntProperty(name="Seed", default=3)
    use_jitter: bpy.props.BoolProperty(name="Unevenness of intact planks", default=True)
    jitter_lift: bpy.props.FloatProperty(name="Max. Lift", default=0.002, min=0.0, unit='LENGTH')
    jitter_tilt: bpy.props.FloatProperty(name="Max. Tilt (deg.)", default=0.3, min=0.0, max=5.0)
    jitter_shift: bpy.props.FloatProperty(name="Max. End Offset", default=0.0004, min=0.0, unit='LENGTH',
                                          description="Horizontal rotation: how far the plank end moves. "
                                                      "Keep it below half of the gap between planks")
    use_twist_nailed: bpy.props.BoolProperty(name="Warp of intact planks", default=True)
    twist_nailed: bpy.props.FloatProperty(name="Max. Twist (deg.)", default=1.5, min=0.0, max=10.0,
                                          description="Twist of a nailed plank (corners lift), 2 loops")
    use_splinters: bpy.props.BoolProperty(name="Splinters at the ends of torn planks", default=True)
    splinter_chance: bpy.props.FloatProperty(name="Chance per End", default=0.6, min=0.0, max=1.0, subtype='FACTOR')
    splinter_depth: bpy.props.FloatProperty(name="Depth", default=0.05, min=0.005, unit='LENGTH')
    splinter_teeth_min: bpy.props.IntProperty(name="Teeth Min", default=3, min=2, max=8)
    splinter_teeth_max: bpy.props.IntProperty(name="Teeth Max", default=5, min=2, max=8)
    splinter_out: bpy.props.FloatProperty(name="Chance of a Protruding Splinter", default=0.35, min=0.0, max=1.0,
                                          subtype='FACTOR')
    use_warp: bpy.props.BoolProperty(name="Deformation of torn planks", default=True)
    warp_bow: bpy.props.FloatProperty(name="Bend (lengths)", default=0.02, min=0.0, max=0.1, subtype='FACTOR',
                                      description="Arc height as a fraction of the plank length")
    warp_twist: bpy.props.FloatProperty(name="Twist (deg.)", default=8.0, min=0.0, max=30.0)
    warp_loops: bpy.props.IntProperty(name="Loops", default=4, min=0, max=12,
                                      description="How many loops along the plank for deformation (fewer = fewer polygons)")
    use_crack: bpy.props.BoolProperty(name="Cracks in torn planks", default=True)
    crack_chance: bpy.props.FloatProperty(name="Chance", default=0.35, min=0.0, max=1.0, subtype='FACTOR')
    crack_min_length: bpy.props.FloatProperty(name="Min. Plank Length", default=0.8, min=0.0, unit='LENGTH')
    crack_angle: bpy.props.FloatProperty(name="Max. Bend (deg.)", default=12.0, min=0.0, max=60.0)
    crack_jag: bpy.props.FloatProperty(name="Poszarpanie", default=0.04, min=0.0, unit='LENGTH',
                                       description="How far along the plank the jagged crack line runs")
    use_nails: bpy.props.BoolProperty(name="Nails torn out with planks", default=True)
    nail_chance: bpy.props.FloatProperty(name="Chance", default=0.8, min=0.0, max=1.0, subtype='FACTOR')
    nail_length: bpy.props.FloatProperty(name="Length", default=0.05, min=0.01, unit='LENGTH')
    nail_diameter: bpy.props.FloatProperty(name="Diameter", default=0.003, min=0.0005, unit='LENGTH')
    nail_head: bpy.props.FloatProperty(name="Head Diameter", default=0.008, min=0.001, unit='LENGTH')
    nail_tilt: bpy.props.FloatProperty(name="Max. Deviation (deg.)", default=20.0, min=0.0, max=80.0)
    use_decay: bpy.props.BoolProperty(name="Burn / Rot", default=False)
    decay_type: bpy.props.EnumProperty(name="Type", default='SPALENIE', items=(
        ('SPALENIE', "Burn", "Charred, eaten edges and ends, dark material"),
        ('BUTWIENIE', "Rot", "Rotten edges, grey-green coating")))
    decay_mode: bpy.props.EnumProperty(name="Area", default='OBSZAR', items=(
        ('OBSZAR', "From Object", "Center and radius from an Empty (edge frayed with noise)"),
        ('LOSOWO', "Random Patches", "Noise patches across the whole floor")))
    decay_source: bpy.props.PointerProperty(name="Source", type=bpy.types.Object,
                                            description="Empty: center and radius (Empty size) of the area")
    decay_coverage: bpy.props.FloatProperty(name="Coverage", default=0.3, min=0.0, max=1.0, subtype='FACTOR')
    decay_scale: bpy.props.FloatProperty(name="Patch Scale", default=0.5, min=0.01, unit='LENGTH')
    decay_erode: bpy.props.FloatProperty(name="Edge Erosion", default=0.012, min=0.0, unit='LENGTH')
    decay_thin: bpy.props.FloatProperty(name="Thinning", default=0.35, min=0.0, max=0.9, subtype='FACTOR')
    decay_seg: bpy.props.FloatProperty(name="Loop Density", default=0.04, min=0.005, unit='LENGTH',
                                       description="Spacing of extra loops in the effect zone (more = smoother "
                                                   "edge, more polygons)")
    decay_material: bpy.props.BoolProperty(name="Effect Material (copy)", default=True,
                                           description="Planks in the area get a copy of the material darkened "
                                                       "by an attribute; the original material stays unchanged")


from .wyrwanie import add_live_updates
add_live_updates(DeskiDetailSettings, skip=("boards",))


classes = (DeskiDetailSettings, DESKI_OT_details)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.deski_detale = bpy.props.PointerProperty(type=DeskiDetailSettings)


def unregister():
    del bpy.types.Scene.deski_detale
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
