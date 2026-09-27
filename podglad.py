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


# ---------------------------------------------------------------- przeliczenie
def compute(context, final=False):
    from . import wyrwanie, detale
    sc = context.scene
    pg, pw, pd = sc.deski_podglad, sc.deski_teren, sc.deski_detale
    coll = pw.boards
    if coll is None or (pw.terrain if pw.scenario == 'TEREN' else pw.emitter) is None:
        pg.message = "Wskaz teren (albo obszar scenariusza) i kolekcje desek"; return
    _state["running"] = True
    t0 = time.time()
    try:
        restore(coll)
        over = {} if final else {"bevel_segments": min(2, pw.bevel_segments), "verify": False,
                                 "steps": min(pw.steps, 60)}
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
    bl_label = "Podglad na zywo"
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
        pg.active = True
        compute(context)
        self.report({'INFO'}, f"Podglad wlaczony ({n} desek zapamietanych). " + pg.message)
        return {'FINISHED'}


class DESKI_OT_live_accept(bpy.types.Operator):
    bl_idname = "object.deski_podglad_zatwierdz"
    bl_label = "Zatwierdz"
    bl_description = "Przelicza deski w pelnej jakosci (jak przyciski Wyrwij / Dodaj detale) i konczy podglad"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.deski_podglad.active

    def execute(self, context):
        sc = context.scene; pg = sc.deski_podglad
        compute(context, final=True)
        forget(sc.deski_teren.boards)
        pg.active = False
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
        restore(sc.deski_teren.boards)
        forget(sc.deski_teren.boards)
        pg.active = False; pg.message = ""
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
    auto: bpy.props.BoolProperty(name="Przeliczaj od razu", default=True,
                                 description="Wylacz przy bardzo duzej liczbie desek - wtedy przycisk Przelicz")
    with_details: bpy.props.BoolProperty(name="Z detalami", default=True, update=live_update,
                                         description="Podglad obejmuje tez Detale (drzazgi, pekniecia, gwozdzie)")
    message: bpy.props.StringProperty(name="Wynik")


def draw_live(layout, context):
    """ramka podgladu - rysowana na gorze panelu Wyrwanie"""
    pg = context.scene.deski_podglad
    box = layout.box()
    if not pg.active:
        box.operator(DESKI_OT_live_start.bl_idname, icon='HIDE_OFF')
        return
    box.alert = True
    box.label(text="PODGLAD NA ZYWO - zmiany widac od razu", icon='HIDE_OFF')
    box.alert = False
    r = box.row(); r.prop(pg, "with_details"); r.prop(pg, "auto")
    if not pg.auto:
        box.operator(DESKI_OT_live_refresh.bl_idname, icon='FILE_REFRESH')
    r = box.row(align=True)
    r.scale_y = 1.3
    r.operator(DESKI_OT_live_accept.bl_idname, icon='CHECKMARK')
    r.operator(DESKI_OT_live_cancel.bl_idname, icon='X')
    if pg.message:
        col = box.column(align=True)
        for part in pg.message.split(" | "):
            col.label(text=part)


classes = (DeskiLiveSettings, DESKI_OT_live_start, DESKI_OT_live_accept, DESKI_OT_live_cancel, DESKI_OT_live_refresh)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.deski_podglad = bpy.props.PointerProperty(type=DeskiLiveSettings)
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)
    if _on_frame not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_on_frame)


def unregister():
    for h, lst in ((_on_depsgraph, bpy.app.handlers.depsgraph_update_post), (_on_frame, bpy.app.handlers.frame_change_post)):
        if h in lst:
            lst.remove(h)
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    del bpy.types.Scene.deski_podglad
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
