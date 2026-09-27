# Podglad na zywo: zmiana ustawien Wyrwania / Detali (albo ruch terenu) od razu przelicza deski
# w jakosci szkicowej. Dopiero Zatwierdz daje wynik finalny (jak dotad przyciski), Anuluj przywraca deski.
#
# Jak to dziala: na starcie kazda deska zapamietuje swoja oryginalna siatke (zostaje nietknieta - fake user,
# potoki i tak kopiuja siatke wspoldzielona), macierz i wlasciwosci deski_*. Kazde przeliczenie zaczyna od
# przywrocenia oryginalu, wiec wynik nie zalezy od historii zmian.
import bpy, json, time
from bpy.app.handlers import persistent
from mathutils import Matrix

SNAP = "deski_snap"
_state = {"dirty": 0.0, "running": False, "last": 0.0}
# szkic: wynik rysowany jako polprzezroczyste deski (nakladka GPU), prawdziwe siatki nietkniete
_GH = {"core": None, "key": None, "sig": None, "tris": None, "lines": None, "hidden": set(), "meshes_dirty": False,
       "handler": None}
PHASE_COL = {1: (0.35, 0.85, 0.35, 0.28), 2: (1.0, 0.75, 0.15, 0.32), 3: (1.0, 0.30, 0.25, 0.32)}


def _boards(coll):
    return [o for o in coll.all_objects if o.type == 'MESH' and not o.get("deski_gwozdz")]


def _mw_list(m):
    return [m[i][j] for i in range(4) for j in range(4)]


def _mw_mat(l):
    return Matrix([l[i * 4:i * 4 + 4] for i in range(4)])


def _plain(v):
    return v.to_list() if hasattr(v, "to_list") else (v.to_dict() if hasattr(v, "to_dict") else v)


# ---------------------------------------------------------------- zapamietanie / przywracanie
def snapshot(coll):
    n = 0
    bset = set(_boards(coll))
    for ob in bpy.data.objects:                             # gwozdzie i dno sprzed podgladu zostaja
        if (ob.get("deski_gwozdz") and ob.parent in bset) or ob.get("deski_scen"):
            ob["deski_gwozdz_staly"] = 1
    for o in _boards(coll):
        if o.get(SNAP):
            continue
        o.data.use_fake_user = True                   # oryginal zostaje nietkniety
        props = {k: _plain(o[k]) for k in o.keys() if k.startswith("deski_") and k != SNAP}
        o[SNAP] = json.dumps({"mesh": o.data.name, "mw": _mw_list(o.matrix_world), "props": props,
                              "hide": [o.hide_get(), o.hide_render]})
        n += 1
    return n


def restore(coll):
    """deski jak przed podgladem; usuwa gwozdzie dodane przez Detale"""
    boards = _boards(coll)
    bset = set(boards)
    for ob in [ob for ob in bpy.data.objects if not ob.get("deski_gwozdz_staly")
               and ((ob.get("deski_gwozdz") and ob.parent in bset) or ob.get("deski_scen"))]:
        bpy.data.objects.remove(ob)
    for o in boards:
        d = o.get(SNAP)
        if not d:
            continue
        d = json.loads(d)
        me = bpy.data.meshes.get(d["mesh"])
        if me is None:
            continue
        old = o.data
        if old is not me:
            o.data = me
            if old.users == 0:
                bpy.data.meshes.remove(old)
        o.matrix_world = _mw_mat(d["mw"])
        if "hide" in d:
            o.hide_set(d["hide"][0]); o.hide_render = d["hide"][1]
        for k in [k for k in o.keys() if k.startswith("deski_") and k != SNAP]:
            del o[k]
        for k, v in d["props"].items():
            o[k] = v


def forget(coll):
    """koniec sesji: zdejmij znaczniki; oryginaly, ktorych nikt juz nie uzywa, sa usuwane"""
    for ob in bpy.data.objects:
        if ob.get("deski_gwozdz_staly"):
            del ob["deski_gwozdz_staly"]
    for o in _boards(coll):
        d = o.get(SNAP)
        if not d:
            continue
        del o[SNAP]
        me = bpy.data.meshes.get(json.loads(d)["mesh"])
        if me is not None:
            me.use_fake_user = False
            if me.users == 0:
                bpy.data.meshes.remove(me)


# ---------------------------------------------------------------- szkic
def _field_sig(pw):
    """podpis wszystkiego, od czego zalezy teren pod deskami - zmiana = przelicz wysokosci"""
    def mw(o):
        return tuple(round(x, 6) for r in o.matrix_world for x in r) if o else None
    sig = (pw.scenario, mw(pw.terrain), mw(pw.emitter), pw.bulge_height, pw.hole_depth, pw.crater_size,
           pw.blast_strength, bpy.context.scene.frame_current)
    if pw.emitter is not None and pw.emitter.type == 'EMPTY':
        sig += (pw.emitter.empty_display_size,)
    return sig


def _set_hidden(names):
    for n in _GH["hidden"] - names:
        o = bpy.data.objects.get(n)
        if o and n in bpy.context.view_layer.objects:
            o.hide_set(False)
    for n in names - _GH["hidden"]:
        o = bpy.data.objects.get(n)
        if o and n in bpy.context.view_layer.objects:
            o.hide_set(True)
    _GH["hidden"] = set(names)


def ghost_clear():
    _set_hidden(set())
    _GH.update(tris=None, lines=None, core=None, key=None, sig=None)


def ghost_compute(context, coll):
    """tylko stan fizyczny desek (bez siatek, bez detali) -> nakladka; ruszone oryginaly ukryte"""
    from . import wyrwanie
    pw = context.scene.deski_teren
    if _GH["meshes_dirty"]:                                  # po trybie Pelnym - najpierw oryginaly
        restore(coll); _GH["meshes_dirty"] = False; _GH["hidden"] = set()
    key = (coll.name, pw.segment_length, pw.nail_position)
    if _GH["core"] is None or _GH["key"] != key:
        core = wyrwanie.DeformCore()
        core.board_cache, core.h_cache, core.keep_tb = {}, {}, True
        _GH.update(core=core, key=key, sig=None)
    core = _GH["core"]
    sig = _field_sig(pw)
    if sig != _GH["sig"]:
        core.h_cache.clear(); core.TB = None; _GH["sig"] = sig
    res = core.solve(context, wyrwanie.Settings(pw, boards=coll, steps=min(pw.steps, 60), verify=False))
    if res[0] is None:
        _set_hidden(set()); _GH.update(tris=None, lines=None)
        return res[2]
    boards, moved, thrown, skipped = res
    tris, tcol, lines = [], [], []
    for b in moved:
        col = PHASE_COL.get(b.phase, PHASE_COL[3])
        ks = range(b.N + 1) if b.kinks else (0, b.N)
        W2 = b.W / 2
        secs = [(b.node_point(k, -W2, 0.0), b.node_point(k, W2, 0.0), b.node_point(k, W2, b.T), b.node_point(k, -W2, b.T))
                for k in ks]
        for s0, s1 in zip(secs, secs[1:]):
            for i in (1, 2, 3):                                # boki i wierzch (spod niewidoczny - bez warstw)
                a0, a1, b0, b1 = s0[i], s0[(i + 1) % 4], s1[i], s1[(i + 1) % 4]
                tris += [a0, a1, b1, a0, b1, b0]
        for sc_ in (secs[0], secs[-1]):
            for i in range(4):
                lines += [sc_[i], sc_[(i + 1) % 4]]
        for i in (2, 3):                                     # dlugie krawedzie wierzchu
            for s0, s1 in zip(secs, secs[1:]):
                lines += [s0[i], s1[i]]
        tcol += [col] * (len(tris) - len(tcol))
    _GH["tris"] = ([tuple(v) for v in tris], tcol)
    _GH["lines"] = [tuple(v) for v in lines]
    _set_hidden({b.o.name for b in moved} | {b.o.name for b in thrown})
    cnt = {1: 0, 2: 0, 3: 0}
    for b in moved:
        cnt[min(b.phase, 3)] += 1
    msg = (f"przybite {cnt[1]} · luźne {cnt[2]} · wyrwane {cnt[3]} | "
           + (f"wyrzucone {len(thrown)}, " if thrown else "") + f"nietknięte {len(boards) - len(moved) - len(thrown)}")
    return msg


def _draw_ghost():
    ctx = bpy.context
    sc = ctx.scene
    if sc is None or not getattr(sc, "deski_podglad", None) or not sc.deski_podglad.active or not _GH["tris"]:
        return
    import gpu
    from gpu_extras.batch import batch_for_shader
    pos, cols = _GH["tris"]
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(False)
    sh = gpu.shader.from_builtin('SMOOTH_COLOR')
    batch_for_shader(sh, 'TRIS', {"pos": pos, "color": cols}).draw(sh)
    if _GH["lines"]:
        shl = gpu.shader.from_builtin('UNIFORM_COLOR')
        shl.uniform_float("color", (0.05, 0.05, 0.05, 0.8))
        batch_for_shader(shl, 'LINES', {"pos": _GH["lines"]}).draw(shl)
    gpu.state.depth_mask_set(True)
    gpu.state.blend_set('NONE')


# ---------------------------------------------------------------- przeliczenie
def compute(context, final=False):
    from . import wyrwanie, detale
    sc = context.scene
    pg, pw, pd = sc.deski_podglad, sc.deski_teren, sc.deski_detale
    coll = pg.coll or pw.boards                          # kolekcja zapamietana na starcie sesji
    if coll is None or (pw.terrain if pw.scenario == 'TEREN' else pw.emitter) is None:
        pg.message = "Wskaz teren (albo obszar scenariusza) i kolekcje desek"; return
    _state["running"] = True
    t0 = time.time()
    try:
        if not final and pg.mode == 'SZKIC':
            msg = ghost_compute(context, coll)
            pg.message = f"Szkic: {time.time() - t0:.2f} s | " + msg
            return
        ghost_clear()
        _GH["meshes_dirty"] = not final
        restore(coll)
        over = {"boards": coll} if final else {"boards": coll, "bevel_segments": min(2, pw.bevel_segments),
                                                  "verify": False, "steps": min(pw.steps, 60)}
        core = wyrwanie.DeformCore()
        lvl, msg = core.run(context, wyrwanie.Settings(pw, **over), remove_rigidbody=False)
        if pg.with_details and core.moved is not None:
            dl, dmsg = detale.DetailCore().run(context, pd, coll)
            msg += " | " + dmsg
        pg.message = ("" if final else f"[podglad {time.time() - t0:.1f} s] ") + msg
    finally:
        _state["running"] = False
        _state["last"] = time.time()


def _tick():
    ctx = bpy.context
    sc = ctx.scene
    if sc is None or not sc.deski_podglad.active or not _state["dirty"]:
        _state["dirty"] = 0.0
        return None
    if time.time() - _state["dirty"] < 0.25:                # czekaj, az suwak sie zatrzyma
        return 0.1
    _state["dirty"] = 0.0
    try:
        compute(ctx)
    except Exception as e:                                   # podglad nie moze wywrocic Blendera
        sc.deski_podglad.message = f"Blad podgladu: {e}"
        import traceback; traceback.print_exc()
    for a in ctx.screen.areas if ctx.screen else ():
        if a.type == 'VIEW_3D':
            a.tag_redraw()
    return None


def changed(context=None):
    """wolane z update= ustawien i z handlerow: zaplanuj przeliczenie (z opoznieniem)"""
    sc = (context or bpy.context).scene
    if sc is None or not getattr(sc, "deski_podglad", None) or not sc.deski_podglad.active or _state["running"]:
        return
    if not sc.deski_podglad.auto:
        return
    _state["dirty"] = time.time()
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=0.3)


def live_update(self, context):
    changed(context)


def _watched(sc):
    pw = sc.deski_teren
    names = set()
    if pw.terrain:
        names.add(pw.terrain.name)
    for k in ("emitter",):                                   # obiekty sterujace scenariuszy
        ob = getattr(pw, k, None)
        if ob:
            names.add(ob.name)
    ob = sc.deski_detale.decay_source
    if ob:
        names.add(ob.name)
    return names


@persistent
def _on_depsgraph(scene, depsgraph):
    pg = getattr(scene, "deski_podglad", None)
    if not pg or not pg.active or _state["running"]:
        return
    if time.time() - _state["last"] < 0.05:                  # zmiany wywolane przez samo przeliczenie
        return
    names = _watched(scene)
    for u in depsgraph.updates:
        if isinstance(u.id, bpy.types.Object) and u.id.name in names and (u.is_updated_transform or u.is_updated_geometry):
            changed(); return


@persistent
def _on_frame(scene, depsgraph=None):
    pg = getattr(scene, "deski_podglad", None)
    if pg and pg.active and scene.deski_teren.terrain and scene.deski_teren.terrain.animation_data:
        changed()


# ---------------------------------------------------------------- operatory
class DESKI_OT_live_start(bpy.types.Operator):
    bl_idname = "object.deski_podglad_start"
    bl_label = "Podgląd na żywo"
    bl_description = ("Wlacza podglad: kazda zmiana ustawien Wyrwania i Detali (albo przesuniecie terenu) "
                      "od razu przelicza deski w jakosci szkicowej. Zatwierdz = wynik finalny, Anuluj = powrot")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        pw = context.scene.deski_teren
        src = pw.terrain if pw.scenario == 'TEREN' else pw.emitter
        return src is not None and pw.boards is not None and not context.scene.deski_podglad.active

    def execute(self, context):
        from . import wyrwanie
        sc = context.scene; pw = sc.deski_teren; pg = sc.deski_podglad
        coll = pw.boards
        if pw.remove_rigidbody:                               # raz, na starcie (tego Anuluj nie cofa)
            wyrwanie.remove_rigid_bodies(context, _boards(coll))
        n = snapshot(coll)
        pg.coll = coll
        pg.active = True
        compute(context)
        self.report({'INFO'}, f"Podglad wlaczony ({n} desek zapamietanych). " + pg.message)
        return {'FINISHED'}


class DESKI_OT_live_accept(bpy.types.Operator):
    bl_idname = "object.deski_podglad_zatwierdz"
    bl_label = "Zatwierdź"
    bl_description = "Przelicza deski w pelnej jakosci (jak przyciski Wyrwij / Dodaj detale) i konczy podglad"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.deski_podglad.active

    def execute(self, context):
        sc = context.scene; pg = sc.deski_podglad
        ghost_clear()
        compute(context, final=True)
        forget(pg.coll or sc.deski_teren.boards)
        pg.active = False; pg.coll = None
        self.report({'INFO'}, pg.message)
        return {'FINISHED'}


class DESKI_OT_live_cancel(bpy.types.Operator):
    bl_idname = "object.deski_podglad_anuluj"
    bl_label = "Anuluj"
    bl_description = "Przywraca deski sprzed podgladu i konczy podglad"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.deski_podglad.active

    def execute(self, context):
        sc = context.scene; pg = sc.deski_podglad
        coll = pg.coll or sc.deski_teren.boards
        ghost_clear()
        restore(coll)
        forget(coll)
        pg.active = False; pg.message = ""; pg.coll = None
        return {'FINISHED'}


class DESKI_OT_live_refresh(bpy.types.Operator):
    bl_idname = "object.deski_podglad_odswiez"
    bl_label = "Przelicz"
    bl_description = "Przelicza podglad teraz"

    @classmethod
    def poll(cls, context):
        return context.scene.deski_podglad.active

    def execute(self, context):
        compute(context)
        return {'FINISHED'}


class DeskiLiveSettings(bpy.types.PropertyGroup):
    active: bpy.props.BoolProperty(name="Podglad aktywny", default=False)
    mode: bpy.props.EnumProperty(name="Podglad", default='SZKIC', update=live_update, items=(
        ('SZKIC', "Szkic", "Szybko: polprzezroczyste deski w kolorach stanu (zielona - przybita, zolta - jeden gwozdz, "
                           "czerwona - wyrwana); prawdziwe siatki nietkniete, bez detali", 'SHADING_WIRE', 0),
        ('PELNY', "Pełny", "Wolniej: prawdziwe siatki z petlami, bevelem i detalami", 'SHADING_TEXTURE', 1)))
    auto: bpy.props.BoolProperty(name="Przeliczaj od razu", default=True,
                                 description="Wylacz przy bardzo duzej liczbie desek - wtedy przycisk Przelicz")
    with_details: bpy.props.BoolProperty(name="Z detalami", default=True, update=live_update,
                                         description="Podglad obejmuje tez Detale (drzazgi, pekniecia, gwozdzie)")
    message: bpy.props.StringProperty(name="Wynik")
    coll: bpy.props.PointerProperty(name="Kolekcja sesji", type=bpy.types.Collection)


classes = (DeskiLiveSettings, DESKI_OT_live_start, DESKI_OT_live_accept, DESKI_OT_live_cancel, DESKI_OT_live_refresh)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.deski_podglad = bpy.props.PointerProperty(type=DeskiLiveSettings)
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)
    if _on_frame not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_on_frame)
    if not bpy.app.background:
        _GH["handler"] = bpy.types.SpaceView3D.draw_handler_add(_draw_ghost, (), 'WINDOW', 'POST_VIEW')


def unregister():
    for h, lst in ((_on_depsgraph, bpy.app.handlers.depsgraph_update_post), (_on_frame, bpy.app.handlers.frame_change_post)):
        if h in lst:
            lst.remove(h)
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    if _GH["handler"] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_GH["handler"], 'WINDOW'); _GH["handler"] = None
    del bpy.types.Scene.deski_podglad
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
