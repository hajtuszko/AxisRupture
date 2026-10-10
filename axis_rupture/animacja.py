# Animacja wyrywania (etap 1 w czasie).
# Dla probek klatek liczony jest stan fizyczny desek (ten sam solver co wynik statyczny): teren w danej
# klatce (animowany teren) albo teren wynurzajacy sie spod podlogi / narastajacy scenariusz (brak animacji).
# Puszczony gwozdz juz sie nie wbija (faza nie maleje). Deski:
#   - sztywne przez caly czas  -> klucze obiektu (polozenie, obrot),
#   - wyginane w jakiejkolwiek klatce -> jedna topologia (petle w miejscach zgiec, przy gwozdziach, kilka
#     rownomiernych), kazda probka = odwzorowanie deski plaskiej na krzywa z lukami w zalamaniach;
#     absolutne shape keys + jedna krzywa Evaluation Time,
#   - wyrzucone podmuchem -> lot balistyczny z obrotem, potem ukryte.
import bpy, bmesh, math, json, random, zlib
from mathutils import Vector, Matrix, Quaternion

ORIG = "deski_anim_orig"


# ---------------------------------------------------------------- stan deski -> krzywa
def _state(b, moved):
    if not moved or b.phase == 0:
        return None
    return {"phase": b.phase, "side": getattr(b, "side", 0), "poly": b.poly, "kinks": list(b.kinks or []),
            "nodes": b.nodes, "roll": b.roll[0] if b.roll else 0.0, "shift": b.shift.copy()}


def _offsets(b, kinks):
    """szerokosc luku kazdego zalamania - jak bevel w wyniku statycznym (nie nachodzi na sasiednie / koniec)"""
    stops = sorted([-b.L / 2, b.L / 2] + [k[0] for k in kinks])
    out = []
    for s_k, ang, wd in kinks:
        gap = min(abs(s_k - t) for t in stops if abs(s_k - t) > 1e-6)
        out.append((s_k, ang, min(wd, 0.45 * gap)))
    return out


class Curve:
    """linia spodu deski w stanie: s (spoczynkowe) -> punkt (x, z) i normalna; luk (Bezier) w zalamaniach"""

    def __init__(self, b, st):
        L2 = b.L / 2
        if st is None:
            self.poly = [(-L2, -L2, 0.0), (L2, L2, 0.0)]; self.k = []; self.psi = 0.0; self.shift = Vector()
            return
        if st["poly"]:
            self.poly = st["poly"]
        else:
            (x0, z0), (x1, z1) = st["nodes"][0], st["nodes"][-1]
            self.poly = [(-L2, x0, z0), (L2, x1, z1)]
        self.k = _offsets(b, st["kinks"]) if st["kinks"] else []
        self.psi = st["roll"]; self.shift = st["shift"]

    def lin(self, s):
        poly = self.poly
        i = next((i for i in range(len(poly) - 1) if s <= poly[i + 1][0] + 1e-9), len(poly) - 2)
        (s0, x0, z0), (s1, x1, z1) = poly[i], poly[i + 1]
        d = Vector((x1 - x0, z1 - z0)); ln = d.length
        d = d / ln if ln > 1e-12 else Vector((1, 0))
        return Vector((x0, z0)) + d * (s - s0), d

    def at(self, s):
        for s_k, ang, off in self.k:
            if off > 1e-6 and abs(s - s_k) < off:
                P1, _ = self.lin(s_k - off); P2, _ = self.lin(s_k + off)
                j = next(i for i, q in enumerate(self.poly) if abs(q[0] - s_k) < 1e-6)
                K = Vector((self.poly[j][1], self.poly[j][2]))
                t = (s - (s_k - off)) / (2 * off)
                P = P1 * (1 - t) ** 2 + K * (2 * t * (1 - t)) + P2 * t * t
                T = (K - P1) * (2 * (1 - t)) + (P2 - K) * (2 * t)
                T = T.normalized() if T.length > 1e-12 else Vector((1, 0))
                return P, Vector((-T.y, T.x))
        P, d = self.lin(s)
        return P, Vector((-d.y, d.x))

    def world(self, b, s, w, h):
        P, n = self.at(s)
        u = w * math.sin(self.psi) + h * math.cos(self.psi)
        y = w * math.cos(self.psi) - h * math.sin(self.psi)
        q = P + n * u
        return b.F @ Vector((q.x, y, q.y - b.T / 2)) + self.shift


def _rigid_matrix(b, st, mw):
    """macierz obiektu dla sztywnej deski w stanie st (jak apply() dla deski prostej)"""
    if st is None:
        return mw.copy()
    c = Curve(b, st)
    (x0, z0), (x1, z1) = (c.poly[0][1], c.poly[0][2]), (c.poly[-1][1], c.poly[-1][2])
    phi = math.atan2(z1 - z0, x1 - x0)
    s0 = c.poly[0][0]
    G = Matrix.Translation(Vector((x0, 0, z0 - b.T / 2))) @ Matrix.Rotation(-phi, 4, 'Y') \
        @ Matrix.Rotation(c.psi, 4, 'X') @ Matrix.Translation(Vector((-s0, 0, b.T / 2)))
    return Matrix.Translation(c.shift) @ b.F @ G @ b.Finv @ mw


# ---------------------------------------------------------------- operator
class DESKI_OT_animate(bpy.types.Operator):
    bl_idname = "object.deski_animuj"
    bl_label = "Animuj wyrywanie"
    bl_description = ("Animacja etapu 1 w zakresie klatek sceny: teren (animowany albo wynurzajacy sie) / "
                      "narastajacy scenariusz wygina i wyrywa deski klatka po klatce")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        p = context.scene.deski_teren
        src = p.terrain if p.scenario == 'TEREN' else p.emitter
        pg = getattr(context.scene, "deski_podglad", None)
        return src is not None and p.boards is not None and not (pg and pg.active)

    def execute(self, context):
        from . import wyrwanie
        sc = context.scene; p = sc.deski_teren
        scen = p.scenario
        fs, fe = sc.frame_start, sc.frame_end
        step = max(1, p.anim_step)
        frames = list(range(fs, fe + 1, step))
        if frames[-1] != fe:
            frames.append(fe)
        if len(frames) < 2:
            self.report({'ERROR'}, "Za krotki zakres klatek"); return {'CANCELLED'}
        boards_objs = [o for o in p.boards.all_objects if o.type == 'MESH' and not o.get("deski_gwozdz")
                       and not o.get(ORIG) and o.data.vertices]
        wyrwanie.reset_originals(context, boards_objs)       # animacja od plaskich desek
        if p.remove_rigidbody:
            wyrwanie.remove_rigid_bodies(context, boards_objs)
        # teren animowany? (inaczej wynurza sie spod podlogi w zakresie klatek)
        t_anim = False
        if scen == 'TEREN':
            sc.frame_set(fs); m0 = p.terrain.matrix_world.copy()
            sc.frame_set(fe); m1 = p.terrain.matrix_world.copy()
            t_anim = any(abs(m0[i][j] - m1[i][j]) > 1e-6 for i in range(4) for j in range(4))
        core = wyrwanie.DeformCore()
        core.min_phase = {}
        states = {o.name: [] for o in boards_objs}
        ref = {}
        wm = context.window_manager
        wm.progress_begin(0, len(frames))
        D = None
        for fi, f in enumerate(frames):
            wm.progress_update(fi)
            t = (f - fs) / max(fe - fs, 1)
            if scen == 'WYBUCH':
                t = min(1.0, t / 0.25)                   # wybuch: pelna sila w pierwszej cwiartce, potem lot gruzu
            if scen == 'TEREN':
                sc.frame_set(f if t_anim else fe)
                if not t_anim:
                    if D is None:
                        core.terrain_bvh(context, p.terrain)
                        z0s = [min((o.matrix_world @ v.co).z for v in o.data.vertices) for o in boards_objs]
                        D = max(0.0, core.ttop - min(z0s)) + 0.02
                    core.zoff = -(1 - t) * D
                P = p
            else:
                over = {}
                if scen == 'WILGOC':
                    over["bulge_height"] = p.bulge_height * t
                elif scen == 'ZAPADNIECIE':
                    over["hole_depth"] = p.hole_depth * t
                else:
                    over["blast_strength"] = p.blast_strength * t; over["hole_depth"] = p.hole_depth * t
                P = wyrwanie.Settings(p, **over)
            res = core.solve(context, P, boards_in=boards_objs)
            if res[0] is None:
                for o in boards_objs:
                    states[o.name].append(None)
                continue
            boards, moved, thrown, _ = res
            mv = {b.o.name for b in moved} | {b.o.name for b in thrown}
            got = set()
            for b in boards:
                ref.setdefault(b.o.name, b)
                st = _state(b, b.o.name in mv)
                states[b.o.name].append(st); got.add(b.o.name)
                if st and st["phase"] < 4:
                    prev = core.min_phase.get(b.o.name)
                    if not prev or st["phase"] > prev[0]:
                        core.min_phase[b.o.name] = (st["phase"], st["side"])
            for o in boards_objs:
                if o.name not in got:
                    states[o.name].append(None)
        wm.progress_end()
        sc.frame_set(fs)
        core.zoff = 0.0
        if scen == 'TEREN' and not t_anim and D:
            # teren nieanimowany: wynurza sie spod podlogi tak samo, jak w obliczeniach
            T = p.terrain
            if not T.get(ORIG):
                T[ORIG] = json.dumps({"mw": [x for r in T.matrix_world for x in r], "frames": [fs, fe]})
            z = T.location.z
            T.location.z = z - D; T.keyframe_insert("location", index=2, frame=fs)
            T.location.z = z; T.keyframe_insert("location", index=2, frame=fe)
            for fc in T.animation_data.action.layers[0].strips[0].channelbag(T.animation_data.action_slot).fcurves:
                for k in fc.keyframe_points:
                    k.interpolation = 'LINEAR'

        if scen in ('ZAPADNIECIE', 'WYBUCH') and p.make_ground and core.F is not None:
            from . import scenariusze
            full = core.solve(context, p, boards_in=boards_objs)   # pole w pelnej glebokosci
            if full[0] is not None:
                g = scenariusze.build_ground(context, p, core.F, p.boards)
                f_full = fs + (0.25 if scen == 'WYBUCH' else 1.0) * (fe - fs)
                g.scale.z = 0.001; g.keyframe_insert("scale", index=2, frame=fs)
                g.scale.z = 1.0; g.keyframe_insert("scale", index=2, frame=round(f_full))
        # zastosowanie
        prefs = context.preferences.edit
        old_interp = prefs.keyframe_new_interpolation_type
        prefs.keyframe_new_interpolation_type = 'LINEAR'
        n_rigid = n_bent = n_thrown = 0
        try:
            for o in boards_objs:
                seq = states[o.name]
                if all(st is None for st in seq) or o.name not in ref:
                    continue
                b = ref[o.name]
                mw = o.matrix_world.copy()
                ad = o.animation_data
                act = ad.action.name if ad and ad.action else None
                o[ORIG] = json.dumps({"mesh": o.data.name, "mw": [x for r in mw for x in r],
                                      "rot": o.rotation_mode, "action": act,
                                      "slot": ad.action_slot.identifier if act and ad.action_slot else None})
                if act:
                    bpy.data.actions[act].use_fake_user = True
                    ad.action = None                      # klucze animacji we wlasnej akcji - oryginal nietkniety
                o.data.use_fake_user = True
                # dopoki deska nie zostala wyrzucona
                i_throw = next((i for i, st in enumerate(seq) if st and st["phase"] == 4), None)
                live = seq[:i_throw] if i_throw is not None else seq
                if not live:
                    live = [None]
                bent = any(st and st["kinks"] for st in live)
                if bent:
                    self.bend_keys(o, b, live, frames[:len(live)], p, mw)
                    n_bent += 1
                else:
                    o.rotation_mode = 'QUATERNION'
                    for st, f in zip(live, frames):
                        o.matrix_world = _rigid_matrix(b, st, mw)
                        o.keyframe_insert("location", frame=f); o.keyframe_insert("rotation_quaternion", frame=f)
                    n_rigid += 1
                if i_throw is not None:
                    self.fly(o, b, mw, frames[i_throw - 1] if i_throw > 0 else frames[0], p, bent)
                    n_thrown += 1
                last = next((st for st in reversed(live) if st), None)
                o["deski_faza"] = ("przybita", "jeden gwozdz", "wyrwana", "wyrzucona")[
                    (4 if i_throw is not None else (last["phase"] if last else 1)) - 1]
        finally:
            prefs.keyframe_new_interpolation_type = old_interp
        self.report({'INFO'}, f"Animacja {fs}-{fe} (co {step} kl., probek {len(frames)}): sztywne {n_rigid}, "
                              f"wyginane (shape keys) {n_bent}, wyrzucone {n_thrown}"
                              + (", teren animowany" if t_anim else ""))
        return {'FINISHED'}

    # --- deska wyginana: wspolna topologia + absolutne shape keys
    def bend_keys(self, o, b, seq, frames, p, mw):
        L2 = b.L / 2
        cuts = []

        def add(s, tol):
            if -L2 + 0.002 < s < L2 - 0.002 and all(abs(s - c) > tol for c in cuts):
                cuts.append(s)
        seg = max(1, p.bevel_segments)
        final = next((st for st in reversed(seq) if st), None)
        for st in ([final] if final else []) + [st for st in seq if st]:
            for s_k, ang, off in _offsets(b, st["kinks"]):
                if st is final:
                    for i in range(seg + 1):
                        add(s_k - off + 2 * off * i / seg, 0.003)
                else:
                    add(s_k, 0.02)
        for k in (b.kA, b.kB):
            add(b.sk[k], 0.02)
        for i in range(1, p.anim_loops + 1):
            add(-L2 + b.L * i / (p.anim_loops + 1), 0.03)
        me = o.data.copy()
        bm = bmesh.new(); bm.from_mesh(me)
        bm.transform(b.Finv @ mw)                       # uklad deski
        for s in sorted(cuts):
            bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
                                   plane_co=Vector((s, 0, 0)), plane_no=Vector((1, 0, 0)))
        ngons = [f for f in bm.faces if len(f.verts) > 4]      # bez n-gonow: trojkaty -> czworokaty
        if ngons:
            r = bmesh.ops.triangulate(bm, faces=ngons, ngon_method='BEAUTY')
            bmesh.ops.join_triangles(bm, faces=r["faces"], angle_face_threshold=math.radians(10),
                                     angle_shape_threshold=math.pi, cmp_uvs=True, cmp_materials=True)
        # ostre krawedzie deski (w spoczynku > 30 st.), reszta gladka - luki ladnie sie cieniuja
        for e in bm.edges:
            e.smooth = not (len(e.link_faces) == 2 and e.calc_face_angle(0.0) > math.radians(30))
        for f in bm.faces:
            f.smooth = True
        rest = [(v.co.x, v.co.y, v.co.z + b.T / 2) for v in bm.verts]
        inv = mw.inverted()
        bm.transform((b.Finv @ mw).inverted())
        bm.to_mesh(me); bm.free()
        old = o.data; o.data = me
        if old.users == 0:
            bpy.data.meshes.remove(old)
        o.shape_key_add(name="Basis", from_mix=False)
        for st, f in zip(seq, frames):
            c = Curve(b, st)
            kb = o.shape_key_add(name=f"f{f:04d}", from_mix=False)
            co = []
            for s, w, h in rest:
                co.extend(inv @ c.world(b, s, w, h))
            kb.data.foreach_set("co", co)
        key = o.data.shape_keys
        key.use_relative = False
        # blok 0 = Basis (spoczynek), bloki 1..n = probki; eval_time liniowo przez probki
        for i, f in enumerate(frames):
            key.eval_time = (i + 1) * 10
            key.keyframe_insert("eval_time", frame=f)

    # --- lot wyrzuconej deski
    def fly(self, o, b, mw, f0, p, bent):
        sc = bpy.context.scene
        rnd = random.Random(p.seed * 31337 + zlib.crc32(o.name.encode()))
        fps = sc.render.fps / sc.render.fps_base
        e = p.emitter
        c = b.c.copy()
        out = (c - e.matrix_world.translation); out.z = 0
        out = out.normalized() if out.length > 1e-6 else Vector((1, 0, 0))
        v = out * rnd.uniform(1.5, 4.0) * p.blast_strength + Vector((0, 0, rnd.uniform(3.0, 6.0) * p.blast_strength))
        axis = Vector((rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(-1, 1))).normalized()
        spin = rnd.uniform(4, 12)                        # rad/s
        o.rotation_mode = 'QUATERNION'
        n = int(fps * 1.2)
        base = o.matrix_world.copy() if not bent else mw.copy()
        loc0, rot0, sca0 = base.decompose()
        for i in range(0, n + 1, 2):
            t = i / fps
            o.location = loc0 + v * t + Vector((0, 0, -4.905 * t * t))
            o.rotation_quaternion = Quaternion(axis, spin * t) @ rot0
            o.keyframe_insert("location", frame=f0 + i); o.keyframe_insert("rotation_quaternion", frame=f0 + i)
        for attr in ("hide_viewport", "hide_render"):
            setattr(o, attr, False); o.keyframe_insert(attr, frame=f0 + n)
            setattr(o, attr, True); o.keyframe_insert(attr, frame=f0 + n + 1)
        setattr(o, "hide_viewport", False); o.hide_render = False


class DESKI_OT_animate_clear(bpy.types.Operator):
    bl_idname = "object.deski_animuj_usun"
    bl_label = "Usuń animację"
    bl_description = "Przywraca deski sprzed animacji (siatki, polozenie) i usuwa klucze"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.deski_teren.boards is not None

    def execute(self, context):
        n = 0
        T = context.scene.deski_teren.terrain
        if T and T.get(ORIG):                          # teren: usun klucze wynurzania
            d = json.loads(T[ORIG])
            for f in d["frames"]:
                try:
                    T.keyframe_delete("location", index=2, frame=f)
                except RuntimeError:
                    pass
            T.matrix_world = Matrix([d["mw"][i * 4:i * 4 + 4] for i in range(4)])
            del T[ORIG]
        for ob in [ob for ob in bpy.data.objects if ob.get("deski_scen")]:
            if ob.animation_data:
                ob.animation_data_clear(); ob.scale.z = 1.0
        for o in list(context.scene.deski_teren.boards.all_objects):
            d = o.get(ORIG)
            if not d:
                continue
            d = json.loads(d)
            own = o.animation_data.action if o.animation_data else None
            o.animation_data_clear()
            if own is not None and own.users == 0:
                bpy.data.actions.remove(own)
            if d.get("action") and bpy.data.actions.get(d["action"]):
                act = bpy.data.actions[d["action"]]
                ad = o.animation_data_create(); ad.action = act
                if d.get("slot"):
                    sl = next((x for x in act.slots if x.identifier == d["slot"]), None)
                    if sl is not None:
                        ad.action_slot = sl
            me = bpy.data.meshes.get(d["mesh"])
            if me is not None:
                old = o.data; o.data = me; me.use_fake_user = False
                if old is not me and old.users == 0:
                    bpy.data.meshes.remove(old)
            o.rotation_mode = d["rot"]
            o.matrix_world = Matrix([d["mw"][i * 4:i * 4 + 4] for i in range(4)])
            o.hide_viewport = False; o.hide_render = False
            for k in [k for k in o.keys() if k.startswith("deski_") and k not in ("deski_gwozdzie_tex",)]:
                del o[k]
            n += 1
        self.report({'INFO'}, f"Przywrocono {n} desek")
        return {'FINISHED'}


classes = (DESKI_OT_animate, DESKI_OT_animate_clear)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
