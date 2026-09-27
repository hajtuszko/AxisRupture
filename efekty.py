# Efekty statyczne (pod gamedev): pekajacy lod, odklejanie wykladziny.
# Wynik: osobne obiekty w kolekcji, UV z plaszczyzny zrodlowej, bez n-gonow, bez przenikan.
import bpy, bmesh, math, random
from mathutils import Vector, Matrix
from mathutils.kdtree import KDTree


# ================================================================ wspolne
def plane_affine(src):
    """plaszczyzna -> (sciany [(uv, f3, material)], f3 dowolnej sciany, macierz UV->lokalnie (a, b, P0, uv0))"""
    from .przygotowanie import face_uv_affine, face_material
    me = src.data
    uv = me.uv_layers.active.data
    faces = []
    for poly in me.polygons:
        aff = face_uv_affine(me, poly, uv)
        if aff:
            faces.append(([tuple(uv[l].uv) for l in poly.loop_indices], aff[0], face_material(src, poly)))
    return faces


def uv_of(f3, local):
    """odwrotnosc mapowania UV -> punkt (plaszczyzna): najmniejsze kwadraty"""
    P0 = f3(0.0, 0.0); a = f3(1.0, 0.0) - P0; b = f3(0.0, 1.0) - P0
    d = local - P0
    aa, ab, bb = a.dot(a), a.dot(b), b.dot(b)
    det = aa * bb - ab * ab
    return ((d.dot(a) * bb - d.dot(b) * ab) / det, (d.dot(b) * aa - d.dot(a) * ab) / det)


def clip_half(poly, n, c):
    """wielokat (lista (x, y)) przyciety do polplaszczyzny n . p <= c"""
    out = []
    for i, cur in enumerate(poly):
        prev = poly[i - 1]
        dc = n[0] * cur[0] + n[1] * cur[1] - c
        dp = n[0] * prev[0] + n[1] * prev[1] - c
        if dc <= 0:
            if dp > 0:
                t = dp / (dp - dc); out.append((prev[0] + (cur[0] - prev[0]) * t, prev[1] + (cur[1] - prev[1]) * t))
            out.append(cur)
        elif dp <= 0:
            t = dp / (dp - dc); out.append((prev[0] + (cur[0] - prev[0]) * t, prev[1] + (cur[1] - prev[1]) * t))
    return out


def voronoi_cells(seeds, rect, gap, k=24):
    """komorki Voronoi (wypukle) przyciete do prostokata, kazda cofnieta o gap/2 od dwusiecznej"""
    kd = KDTree(len(seeds))
    for i, (x, y) in enumerate(seeds):
        kd.insert((x, y, 0.0), i)
    kd.balance()
    x0, y0, x1, y1 = rect
    cells = []

    def cut(poly, i, x, y, js):
        for j in js:
            if j == i:
                continue
            qx, qy = seeds[j]
            nx, ny = qx - x, qy - y
            L = math.hypot(nx, ny)
            if L < 1e-9:
                continue
            nx, ny = nx / L, ny / L
            mx, my = (x + qx) / 2, (y + qy) / 2
            poly = clip_half(poly, (nx, ny), nx * mx + ny * my - gap / 2)
            if len(poly) < 3:
                return None
        return poly
    for i, (x, y) in enumerate(seeds):
        poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        poly = cut(poly, i, x, y, [j for _, j, _ in kd.find_n((x, y, 0.0), min(k, len(seeds)))])
        if poly:
            # dokladnie: sasiad wplywa na komorke tylko blizej niz 2x jej promien - dociecie wszystkimi takimi
            rmax = max(math.hypot(px - x, py - y) for px, py in poly)
            poly = cut(poly, i, x, y, [j for _, j, _ in kd.find_range((x, y, 0.0), 2 * rmax + gap)])
        cells.append(poly if poly and len(poly) >= 3 else None)
    return cells


class _P:
    """ustawienia elementu dla make_piece (grubosc, faza, min. kawalek, nazwa)"""
    def __init__(self, **kw):
        self.__dict__.update(kw)


# ================================================================ pekajacy lod
class DESKI_OT_ice(bpy.types.Operator):
    bl_idname = "object.deski_lod"
    bl_label = "Rozbij lód"
    bl_description = ("Dzieli plyte lodu na kry (Voronoi): wokol punktu uderzenia promieniste odlamki, "
                      "dalej wieksze kry; kry przechylone i zapadniete wg odleglosci od uderzenia")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.deski_efekty.ice_source is not None

    def execute(self, context):
        from .przygotowanie import make_piece, clip_board
        from .wyrwanie import DeformCore
        e = context.scene.deski_efekty
        src = e.ice_source
        faces = plane_affine(src)
        if not faces:
            self.report({'ERROR'}, "Plaszczyzna lodu musi miec UV"); return {'CANCELLED'}
        rng = random.Random(e.ice_seed)
        f3 = faces[0][1]
        sc_w = max(src.matrix_world.to_scale())
        mu = (f3(1.0, 0.0) - f3(0.0, 0.0)).length * sc_w      # metry na jednostke U
        mv = (f3(0.0, 1.0) - f3(0.0, 0.0)).length * sc_w
        us = [q[0] for fu, _, _ in faces for q in fu]; vs = [q[1] for fu, _, _ in faces for q in fu]
        rect = (min(us) * mu, min(vs) * mv, max(us) * mu, max(vs) * mv)       # w metrach
        W, H = rect[2] - rect[0], rect[3] - rect[1]
        # punkt uderzenia -> UV (metry)
        imp = None
        if e.ice_impact is not None:
            loc = src.matrix_world.inverted() @ e.ice_impact.matrix_world.translation
            u, v = uv_of(f3, loc)
            imp = (u * mu, v * mv)
        R = e.ice_radius
        seeds = []
        if imp:
            # promieniste odlamki: pierscienie coraz rzadsze, ziarna z odchyleniem
            r = max(0.04, R * 0.06)
            while r < R:
                m = max(5, int(2 * math.pi * r / max(r * 0.55, 0.05)))
                a0 = rng.uniform(0, 2 * math.pi)
                for k in range(m):
                    a = a0 + 2 * math.pi * (k + rng.uniform(-0.3, 0.3)) / m
                    rr = r * rng.uniform(0.85, 1.15)
                    seeds.append((imp[0] + rr * math.cos(a), imp[1] + rr * math.sin(a)))
                r *= 1.45
        # kry: siatka z rozrzutem, rozmiar ice_floe
        fl = max(e.ice_floe, 0.05)
        nx, ny = max(1, int(W / fl)), max(1, int(H / fl))
        for i in range(nx + 1):
            for j in range(ny + 1):
                x = rect[0] + (i + rng.uniform(0.1, 0.9)) * W / nx
                y = rect[1] + (j + rng.uniform(0.1, 0.9)) * H / ny
                if imp and math.hypot(x - imp[0], y - imp[1]) < R * 0.9:
                    continue
                if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                    seeds.append((x, y))
        cells = voronoi_cells(seeds, rect, e.ice_gap)
        out = bpy.data.collections.new(e.ice_output or (src.name + " - kry"))
        context.scene.collection.children.link(out)
        P = _P(thickness=e.ice_thickness, chamfer=e.ice_chamfer, min_piece=0.01, surface="LOD")
        pieces = []
        for idx, cell in enumerate(cells):
            if cell is None:
                continue
            cuv = [(x / mu, y / mv) for x, y in cell]
            polys = []
            for fuv, ff3, mat in faces:
                cl = clip_board(cuv, fuv)
                if len(cl) >= 3:
                    polys.append(([(ff3(u, v), (u, v)) for u, v in cl], mat))
            if not polys:
                continue
            res = make_piece(src, out, polys, P, len(pieces) + 1)
            if res is None:
                continue
            ob, _ = res
            cx = sum(x for x, _ in cell) / len(cell); cy = sum(y for _, y in cell) / len(cell)
            pieces.append((ob, (cx, cy)))
        # ulozenie: przechyl (wewnetrzna krawedz w dol) i zapadniecie wg odleglosci od uderzenia
        n = (src.matrix_world.to_3x3() @ (f3(1, 0) - f3(0, 0)).cross(f3(0, 1) - f3(0, 0))).normalized()
        from .przygotowanie import src_normal
        if n.dot(src.matrix_world.to_3x3() @ src_normal(src)) < 0:
            n = -n
        removed = 0
        moved = []
        for ob, (cx, cy) in pieces:
            c = ob.matrix_world.translation.copy()
            if imp:
                d = math.hypot(cx - imp[0], cy - imp[1]) / max(R, 1e-6)
                if e.ice_hole > 0 and d < e.ice_hole:
                    bpy.data.objects.remove(ob); removed += 1
                    continue
                s = max(0.0, 1 - d) ** 1.5
                # os przechylu: prostopadla do kierunku od uderzenia, w plaszczyznie
                wdir = (src.matrix_world @ f3((cx - imp[0]) / mu, (cy - imp[1]) / mv)
                        - src.matrix_world @ f3(0, 0)).normalized() if d > 1e-6 else n.orthogonal()
                axis = n.cross(wdir).normalized()
                ang = -math.radians(e.ice_tilt) * s * rng.uniform(0.4, 1.0)
                sink = e.ice_sink * s * rng.uniform(0.6, 1.0)
            else:
                s = 0.0; axis = n.orthogonal().normalized(); ang = 0.0; sink = 0.0
            # lekkie kolysanie wszystkich kry (plywaja)
            bob = math.radians(e.ice_bob) * rng.uniform(-1, 1)
            ax2 = Matrix.Rotation(rng.uniform(0, 2 * math.pi), 3, n) @ n.orthogonal().normalized()
            Rm = (Matrix.Rotation(ang, 4, axis) @ Matrix.Rotation(bob, 4, ax2))
            ob.matrix_world = Matrix.Translation(c - n * sink) @ Rm @ Matrix.Translation(-c) @ ob.matrix_world
            if ang or bob or sink:
                moved.append(ob)
        # kolizje: kra bardziej przesunieta dociskana w dol (w wode)
        objs = [ob for ob in out.objects]
        core = DeformCore()
        pushed = 0
        for _ in range(12):
            context.view_layer.update()
            pairs = core.collisions(context, objs, {o.name for o in moved})
            if not pairs:
                break
            done = set()
            for a_, b_, dep in sorted(pairs, key=lambda t: -t[2]):
                oa, ob_ = bpy.data.objects[a_], bpy.data.objects[b_]
                mv_ = oa if (oa.matrix_world.translation - src.matrix_world.translation).dot(n) < \
                    (ob_.matrix_world.translation - src.matrix_world.translation).dot(n) else ob_
                if mv_.name in done:
                    continue
                mv_.matrix_world = Matrix.Translation(-n * (dep + 0.001)) @ mv_.matrix_world
                done.add(mv_.name); pushed += 1
        if e.ice_water:
            self.water(context, src, n, e)
        src.hide_set(True); src.hide_render = True
        self.report({'INFO'}, f"Kry: {len(objs)} (przerebel: {removed} usunietych, dociśniete w kolizjach: {pushed})")
        return {'FINISHED'}

    def water(self, context, src, n, e):
        name = src.name + " - woda"
        old = bpy.data.objects.get(name)
        if old:
            bpy.data.objects.remove(old)
        me = src.data.copy(); me.name = name
        ob = bpy.data.objects.new(name, me)
        context.scene.collection.objects.link(ob)
        ob.matrix_world = Matrix.Translation(-n * (e.ice_thickness * 0.85)) @ src.matrix_world
        ob.data.materials.clear()
        return ob


# ================================================================ ustawienia
class DeskiEffectsSettings(bpy.types.PropertyGroup):
    ice_source: bpy.props.PointerProperty(name="Płyta lodu", type=bpy.types.Object,
                                          poll=lambda s, o: o.type == 'MESH' and len(o.data.uv_layers) > 0)
    ice_impact: bpy.props.PointerProperty(name="Uderzenie", type=bpy.types.Object,
                                          description="Empty w miejscu uderzenia (puste = rownomierne kry)")
    ice_output: bpy.props.StringProperty(name="Kolekcja wynikowa", default="")
    ice_radius: bpy.props.FloatProperty(name="Promień uderzenia", default=1.2, min=0.05, unit='LENGTH')
    ice_floe: bpy.props.FloatProperty(name="Rozmiar kry", default=0.6, min=0.05, unit='LENGTH',
                                      description="Typowa wielkosc kry poza strefa uderzenia (mniej = wiecej elementow)")
    ice_thickness: bpy.props.FloatProperty(name="Grubość", default=0.05, min=0.001, unit='LENGTH')
    ice_gap: bpy.props.FloatProperty(name="Szerokość pęknięć", default=0.006, min=0.0, unit='LENGTH')
    ice_chamfer: bpy.props.FloatProperty(name="Faza", default=0.004, min=0.0, unit='LENGTH')
    ice_tilt: bpy.props.FloatProperty(name="Maks. przechył (°)", default=14.0, min=0.0, max=60.0)
    ice_sink: bpy.props.FloatProperty(name="Zapadnięcie", default=0.04, min=0.0, unit='LENGTH')
    ice_hole: bpy.props.FloatProperty(name="Przerębel", default=0.12, min=0.0, max=0.9, subtype='FACTOR',
                                      description="Kry blizej srodka niz tyle (czesc promienia) sa usuwane - otwarta woda")
    ice_bob: bpy.props.FloatProperty(name="Kołysanie (°)", default=1.0, min=0.0, max=10.0,
                                     description="Drobny losowy przechyl wszystkich kry (plywaja)")
    ice_water: bpy.props.BoolProperty(name="Dodaj wodę", default=True,
                                      description="Plaszczyzna wody pod lodem (bez materialu - przypisz swoj)")
    ice_seed: bpy.props.IntProperty(name="Seed", default=7)


classes = (DeskiEffectsSettings, DESKI_OT_ice)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.deski_efekty = bpy.props.PointerProperty(type=DeskiEffectsSettings)


def unregister():
    del bpy.types.Scene.deski_efekty
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
