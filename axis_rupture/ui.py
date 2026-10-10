# Panele dodatku (View3D > N > Deski): trzy kroki w kolejnosci pracy, glowne przyciski wyroznione,
# ustawienia w zwijanych podpanelach (zaawansowane domyslnie zwiniete).
import bpy
from bpy.types import Panel

SPACE = dict(bl_space_type='VIEW_3D', bl_region_type='UI', bl_category="Axis Rupture")
_ICONS = None


def icon(*names):
    """pierwsza istniejaca ikona (nazwy ikon zmieniaja sie miedzy wersjami Blendera)"""
    global _ICONS
    if _ICONS is None:
        _ICONS = set(bpy.types.UILayout.bl_rna.functions["label"].parameters["icon"].enum_items.keys())
    return next((n for n in names if n in _ICONS), 'NONE')


def split(layout):
    layout.use_property_split = True
    layout.use_property_decorate = False
    return layout


def big(layout, op, text, ic, scale=1.8, enabled=True):
    row = layout.row(align=True)
    row.scale_y = scale
    row.enabled = enabled
    row.operator(op, text=text, icon=ic, depress=True)     # podswietlony = glowna akcja
    return row


def live_active(context):
    pg = getattr(context.scene, "deski_podglad", None)
    return bool(pg and pg.active)


def live_locked(context):
    """pole kolekcji zablokowane tylko w trwajacej sesji z kolekcja (inaczej da sie wskazac nowe deski)"""
    pg = getattr(context.scene, "deski_podglad", None)
    return bool(pg and pg.active and pg.coll is not None)


# ================================================================ 1. PRZYGOTOWANIE
class DESKI_PT_prep(Panel):
    bl_label = "1  Prepare Planks"
    bl_idname = "DESKI_PT_prep"
    bl_order = 0
    bl_space_type, bl_region_type, bl_category = SPACE.values()

    def draw_header(self, context):
        self.layout.label(icon='MOD_EDGESPLIT')

    def draw(self, context):
        from . import przygotowanie as pz
        p = context.scene.deski_przygotowanie
        L = self.layout
        col = split(L.column())
        col.prop(p, "source", text="Plane")
        col.prop(p, "texture", text="Texture")
        col.prop(p, "output_name", text="Output")
        col.prop(p, "surface", text="Surface")
        L.grid_flow(row_major=True, columns=3, even_columns=True, align=True).prop(p, "layout", expand=True)
        L.separator()
        if pz.JOB["running"]:
            pz.draw_job(L, pz.DESKI_OT_split.bl_idname, 'MOD_EDGESPLIT')
            return
        big(L, pz.DESKI_OT_split.bl_idname, "Split into Pieces" if p.surface != 'DESKI' else "Split into Planks",
            'MOD_EDGESPLIT', enabled=p.source is not None)
        if p.layout == 'HERRINGBONE':
            r = L.row(); r.enabled = p.source is not None
            r.operator(pz.DESKI_OT_split_preview.bl_idname, text="Preview Split (Outlines)", icon='HIDE_OFF')


class DESKI_PT_prep_detect(Panel):
    bl_label = "Seam Detection"
    bl_parent_id = "DESKI_PT_prep"
    bl_space_type, bl_region_type, bl_category = SPACE.values()

    def draw(self, context):
        p = context.scene.deski_przygotowanie
        col = split(self.layout.column())
        if p.layout == 'IRREGULAR':
            col.prop(p, "stone_sens", text="Grout Threshold")
            col.prop(p, "stone_min", text="Min. Stone")
            col.prop(p, "stone_simplify", text="Simplify")
            return
        if p.layout == 'HERRINGBONE':
            col.prop(p, "herr_length", text="Length")
            col.prop(p, "herr_variant", text="Variant")
            col.prop(p, "herr_offset", text="Offset")
            col.prop(p, "herr_calibrate", text="Local Calibration")
            col.label(text="Length 0 = auto, angle/width auto", icon='INFO')
        else:
            col.prop(p, "straight_method", text="Method")
            sub = col.column(); sub.enabled = p.straight_method == 'MAPS'
            sub.prop(p, "straight_calibrate", text="Local Calibration")
            col.separator()
            col.prop(p, "sensitivity", text="Sensitivity")
            col.prop(p, "min_width", text="Min. Width")
            col.prop(p, "min_length", text="Min. Length")
            col.prop(p, "joint_contrast", text="Merge Contrast")
            col.prop(p, "joint_cover", text="Merge Coverage")


class DESKI_PT_prep_nails(Panel):
    bl_label = "Nails from Texture"
    bl_parent_id = "DESKI_PT_prep"
    bl_options = {'DEFAULT_CLOSED'}
    bl_space_type, bl_region_type, bl_category = SPACE.values()

    @classmethod
    def poll(cls, context):
        return context.scene.deski_przygotowanie.layout == 'STRAIGHT'

    def draw_header(self, context):
        self.layout.prop(context.scene.deski_przygotowanie, "detect_nails", text="")

    def draw(self, context):
        p = context.scene.deski_przygotowanie
        col = split(self.layout.column()); col.enabled = p.detect_nails
        col.prop(p, "nail_texture", text="Color Texture")
        col.prop(p, "nail_dark", text="Contrast")


class DESKI_PT_prep_boards(Panel):
    bl_label = "Plank Dimensions"
    bl_parent_id = "DESKI_PT_prep"
    bl_options = {'DEFAULT_CLOSED'}
    bl_space_type, bl_region_type, bl_category = SPACE.values()

    def draw(self, context):
        p = context.scene.deski_przygotowanie
        col = split(self.layout.column())
        if p.layout == 'STRAIGHT':
            col.prop(p, "max_length", text="Max. Length")
        col.prop(p, "thickness", text="Thickness")
        col.prop(p, "gap", text="Gap")
        col.prop(p, "chamfer", text="Edge Bevel")
        col.prop(p, "min_piece", text="Min. Piece")
        col.prop(p, "hide_source", text="Hide Plane")
        col.prop(p, "replace_output", text="Replace Previous Result")


# ================================================================ 2. WYRWANIE
SCEN_ICON = {'TEREN': 'MESH_GRID', 'WILGOC': 'MOD_FLUIDSIM', 'ZAPADNIECIE': 'SORT_DESC', 'WYBUCH': 'FORCE_FORCE'}


class DESKI_PT_panel(Panel):
    bl_label = "2  Tear Planks"
    bl_idname = "DESKI_PT_panel"
    bl_order = 1
    bl_space_type, bl_region_type, bl_category = SPACE.values()

    def draw_header(self, context):
        self.layout.label(icon='MOD_SIMPLEDEFORM')

    def draw(self, context):
        from . import wyrwanie, podglad
        sc = context.scene
        p, pg = sc.deski_teren, sc.deski_podglad
        L = self.layout
        L.grid_flow(row_major=True, columns=2, even_columns=True, align=True).prop(p, "scenario", expand=True)
        col = split(L.column())
        if p.scenario == 'TEREN':
            col.prop(p, "terrain", text="Terrain")
        else:
            col.prop(p, "emitter", text="Area (Empty)")
        r = col.row(); r.enabled = not live_locked(context)
        r.prop(p, "boards", text="Planks")
        L.separator()
        ready = (p.terrain if p.scenario == 'TEREN' else p.emitter) is not None and p.boards is not None
        if not pg.active:
            big(L, podglad.DESKI_OT_live_start.bl_idname, "Live Preview", 'HIDE_OFF', enabled=ready)
            r = L.row(); r.enabled = ready
            r.operator(wyrwanie.DESKI_OT_deform.bl_idname, text="Tear Now", icon='MOD_SIMPLEDEFORM')
            r = L.row(); r.enabled = p.boards is not None
            r.operator(wyrwanie.DESKI_OT_reset.bl_idname, text="Restore Original Planks", icon='LOOP_BACK')
            return
        # --- trwa podglad
        box = L.box()
        head = box.row(); head.alert = True
        head.label(text="LIVE PREVIEW", icon=icon('REC', 'RADIOBUT_ON'))
        box.row().prop(pg, "mode", expand=True)
        if pg.mode == 'SZKIC':
            leg = box.row(align=True); leg.scale_y = 0.8
            leg.label(text="nailed", icon=icon('STRIP_COLOR_04', 'SEQUENCE_COLOR_04', 'COLORSET_03_VEC'))
            leg.label(text="loose", icon=icon('STRIP_COLOR_03', 'SEQUENCE_COLOR_03', 'COLORSET_09_VEC'))
            leg.label(text="torn", icon=icon('STRIP_COLOR_01', 'SEQUENCE_COLOR_01', 'COLORSET_01_VEC'))
        else:
            box.prop(pg, "with_details", text="With Details (slower)")
        r = box.row(); r.prop(pg, "auto", text="Update Instantly")
        if not pg.auto:
            r.operator(podglad.DESKI_OT_live_refresh.bl_idname, text="Recompute", icon='FILE_REFRESH')
        if pg.message:
            msg = box.column(align=True); msg.scale_y = 0.8
            for part in pg.message.split(" | "):
                msg.label(text=part)
        row = box.row(align=True); row.scale_y = 1.8
        row.operator(podglad.DESKI_OT_live_accept.bl_idname, text="Apply", icon='CHECKMARK')
        row.operator(podglad.DESKI_OT_live_cancel.bl_idname, text="Cancel", icon='X')


class DESKI_PT_scen(Panel):
    bl_label = "Scenario Settings"
    bl_parent_id = "DESKI_PT_panel"
    bl_space_type, bl_region_type, bl_category = SPACE.values()

    @classmethod
    def poll(cls, context):
        return context.scene.deski_teren.scenario != 'TEREN'

    def draw_header(self, context):
        self.layout.label(icon=SCEN_ICON.get(context.scene.deski_teren.scenario, 'NONE'))

    def draw(self, context):
        p = context.scene.deski_teren
        col = split(self.layout.column())
        if p.scenario == 'WILGOC':
            col.prop(p, "bulge_height", text="Bulge Height")
        elif p.scenario == 'ZAPADNIECIE':
            col.prop(p, "hole_depth", text="Hole Depth")
            col.prop(p, "hang_max", text="Max. Sag (°)")
            col.prop(p, "make_ground", text="Create Bottom")
        else:
            col.prop(p, "blast_strength", text="Blast Strength")
            col.prop(p, "peel_max", text="Max. Bend (°)")
            col.prop(p, "throw_zone", text="Ejection Zone")
            col.prop(p, "crater_size", text="Crater Size")
            col.prop(p, "hole_depth", text="Crater Depth")
            col.prop(p, "make_ground", text="Create Bottom")
        col.label(text="Radius = Empty size", icon='INFO')


class DESKI_PT_nails(Panel):
    bl_label = "Nails and Tearing"
    bl_parent_id = "DESKI_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    bl_space_type, bl_region_type, bl_category = SPACE.values()

    def draw(self, context):
        p = context.scene.deski_teren
        col = split(self.layout.column())
        col.prop(p, "seed", text="Seed")
        col.prop(p, "nail_position", text="Nails from End")
        col.prop(p, "bend_min_length", text="Bend from Length")
        col.separator()
        for a, b, t in (("bulge_min", "bulge_max", "Deflection"), ("slope_min", "slope_max", "Angle at Nail"),
                        ("angle2_min", "angle2_max", "2nd Nail Break Angle")):
            sub = col.column(align=True)
            sub.prop(p, a, text=f"{t} min"); sub.prop(p, b, text="max")


class DESKI_PT_quality(Panel):
    bl_label = "Quality and Geometry"
    bl_parent_id = "DESKI_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    bl_space_type, bl_region_type, bl_category = SPACE.values()

    def draw(self, context):
        p = context.scene.deski_teren
        col = split(self.layout.column())
        col.prop(p, "segment_length", text="Segment Length")
        sub = col.column(align=True)
        sub.prop(p, "bevel_width", text="Bevel: Width")
        sub.prop(p, "bevel_segments", text="segments")
        col.prop(p, "roll_max", text="Cross Twist")
        col.prop(p, "steps", text="Simulation Steps")
        col.prop(p, "clearance", text="Clearance")
        col.prop(p, "smooth_shading", text="Smooth Arc")


class DESKI_PT_options(Panel):
    bl_label = "Options"
    bl_parent_id = "DESKI_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    bl_space_type, bl_region_type, bl_category = SPACE.values()

    def draw(self, context):
        p = context.scene.deski_teren
        col = split(self.layout.column())
        col.prop(p, "resolve_collisions", text="Separate Collisions")
        col.prop(p, "verify", text="Check Intersections")
        col.prop(p, "remove_rigidbody", text="Remove Rigid Body")


class DESKI_PT_anim(Panel):
    bl_label = "Animation"
    bl_parent_id = "DESKI_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    bl_space_type, bl_region_type, bl_category = SPACE.values()

    def draw_header(self, context):
        self.layout.label(icon='RENDER_ANIMATION')

    def draw(self, context):
        from . import animacja
        sc = context.scene; p = sc.deski_teren
        L = self.layout
        col = split(L.column())
        r = col.row(align=True); r.prop(sc, "frame_start", text="Frames"); r.prop(sc, "frame_end", text="")
        col.prop(p, "anim_step", text="Frame Step")
        col.prop(p, "anim_loops", text="Animation Loops")
        big(L, animacja.DESKI_OT_animate.bl_idname, "Animate Tearing", 'PLAY', scale=1.5,
            enabled=not live_active(context))
        L.operator(animacja.DESKI_OT_animate_clear.bl_idname, text="Remove Animation", icon='TRASH')


# ================================================================ 3. DETALE
class DESKI_PT_details(Panel):
    bl_label = "3  Plank Details"
    bl_idname = "DESKI_PT_details"
    bl_order = 2
    bl_space_type, bl_region_type, bl_category = SPACE.values()

    def draw_header(self, context):
        self.layout.label(icon='MOD_NOISE')

    def draw(self, context):
        from . import detale
        p = context.scene.deski_detale
        L = self.layout
        col = split(L.column())
        r = col.row(); r.enabled = not live_locked(context)
        r.prop(p, "boards", text="Planks")
        col.prop(p, "seed", text="Seed")
        L.separator()
        if live_active(context):
            L.label(text="Full Preview computes details", icon='INFO')
        big(L, detale.DESKI_OT_details.bl_idname, "Add Details", 'MOD_NOISE', enabled=not live_active(context))


def _feature(flag, label, props, ic='NONE', closed=True, extra=None):
    """podpanel funkcji Detali z checkboxem w naglowku"""
    def draw_header(self, context):
        self.layout.prop(context.scene.deski_detale, flag, text="")

    def draw(self, context):
        p = context.scene.deski_detale
        col = split(self.layout.column()); col.enabled = getattr(p, flag)
        for name, text in (extra(p) if extra else props):
            col.prop(p, name, text=text)
    return type("DESKI_PT_det_" + flag, (Panel,), {
        "bl_label": label, "bl_parent_id": "DESKI_PT_details", "bl_space_type": 'VIEW_3D',
        "bl_region_type": 'UI', "bl_category": "Axis Rupture", "bl_options": {'DEFAULT_CLOSED'} if closed else set(),
        "draw_header": draw_header, "draw": draw})


def _decay_props(p):
    out = [("decay_type", "Type"), ("decay_mode", "Area")]
    out.append(("decay_source", "Source (Empty)") if p.decay_mode == 'OBSZAR' else ("decay_coverage", "Coverage"))
    out += [("decay_scale", "Patch Scale"), ("decay_erode", "Edge Erosion"), ("decay_thin", "Thinning"),
            ("decay_seg", "Loop Density"), ("decay_material", "Effect Material")]
    return out


FEATURES = [
    _feature("use_jitter", "Unevenness (intact)", [("jitter_lift", "Max. Lift"), ("jitter_tilt", "Max. Tilt (°)"),
                                                        ("jitter_shift", "End Offset")]),
    _feature("use_twist_nailed", "Warp (intact)", [("twist_nailed", "Max. Twist (°)")]),
    _feature("use_splinters", "Splinters at Ends", [("splinter_chance", "Chance"), ("splinter_depth", "Depth"),
                                                     ("splinter_teeth_min", "Teeth Min"), ("splinter_teeth_max", "Teeth Max"),
                                                     ("splinter_out", "Protruding Splinter")]),
    _feature("use_warp", "Deformation (torn)", [("warp_bow", "Bend (lengths)"), ("warp_twist", "Twist (°)"),
                                                    ("warp_loops", "Loops")]),
    _feature("use_crack", "Cracks", [("crack_chance", "Chance"), ("crack_min_length", "Min. Plank Length"),
                                       ("crack_angle", "Max. Bend (°)"), ("crack_jag", "Jaggedness")]),
    _feature("use_nails", "Nails with Planks", [("nail_chance", "Chance"), ("nail_length", "Length"),
                                                ("nail_diameter", "Diameter"), ("nail_head", "Head Diameter"),
                                                ("nail_tilt", "Max. Deviation (°)")]),
    _feature("use_decay", "Burn / Rot", None, extra=_decay_props),
]

# ================================================================ 4. EFEKTY
class DESKI_PT_fx(Panel):
    bl_label = "4  Effects"
    bl_idname = "DESKI_PT_fx"
    bl_order = 3
    bl_space_type, bl_region_type, bl_category = SPACE.values()

    def draw_header(self, context):
        self.layout.label(icon='FREEZE')

    def draw(self, context):
        self.layout.label(text="Static effects on planes", icon='INFO')


class DESKI_PT_fx_ice(Panel):
    bl_label = "Cracking Ice"
    bl_parent_id = "DESKI_PT_fx"
    bl_space_type, bl_region_type, bl_category = SPACE.values()

    def draw_header(self, context):
        self.layout.label(icon='FREEZE')

    def draw(self, context):
        from . import efekty
        e = context.scene.deski_efekty
        L = self.layout
        col = split(L.column())
        col.prop(e, "ice_source", text="Ice Sheet")
        col.prop(e, "ice_impact", text="Impact (Empty)")
        col.prop(e, "ice_output", text="Output")
        big(L, efekty.DESKI_OT_ice.bl_idname, "Break Ice", 'FREEZE', enabled=e.ice_source is not None)
        col = split(L.column())
        col.prop(e, "ice_radius", text="Impact Radius")
        col.prop(e, "ice_floe", text="Floe Size")
        col.prop(e, "ice_hole", text="Ice Hole")
        col.prop(e, "ice_tilt", text="Max. Tilt (°)")
        col.prop(e, "ice_sink", text="Sinking")
        col.prop(e, "ice_bob", text="Rocking (°)")
        col.separator()
        col.prop(e, "ice_thickness", text="Thickness")
        col.prop(e, "ice_gap", text="Crack Width")
        col.prop(e, "ice_chamfer", text="Bevel")
        col.prop(e, "ice_water", text="Add Water")
        col.prop(e, "ice_replace", text="Replace Previous Result")
        col.prop(e, "ice_seed", text="Seed")


class DESKI_PT_fx_carpet(Panel):
    bl_label = "Peeling Carpet"
    bl_parent_id = "DESKI_PT_fx"
    bl_space_type, bl_region_type, bl_category = SPACE.values()

    def draw_header(self, context):
        self.layout.label(icon='MOD_CURVE')

    def draw(self, context):
        from . import efekty
        e = context.scene.deski_efekty
        L = self.layout
        col = split(L.column())
        col.prop(e, "carpet_source", text="Carpet")
        col.prop(e, "carpet_handle", text="Peel From (Empty)")
        L.row().prop(e, "carpet_mode", expand=True)
        big(L, efekty.DESKI_OT_carpet.bl_idname, "Peel", 'MOD_CURVE',
            enabled=e.carpet_source is not None and e.carpet_handle is not None)
        col = split(L.column())
        if e.carpet_mode == 'PECHERZ':
            col.prop(e, "carpet_bubble", text="Radius")
            col.prop(e, "carpet_bubble_h", text="Height")
        else:
            col.prop(e, "carpet_length", text="Peel Length")
            col.prop(e, "carpet_radius", text="Fold Radius" if e.carpet_mode == 'ZAGIECIE' else "Roll Radius")
            if e.carpet_mode == 'ZAGIECIE':
                col.prop(e, "carpet_angle", text="Flap Angle (°)")
                col.prop(e, "carpet_curl", text="End Curl")
            col.prop(e, "carpet_noise", text="Line Irregularity")
            col.prop(e, "carpet_wrinkle", text="Folds")
        col.separator()
        col.prop(e, "carpet_thickness", text="Thickness")
        col.prop(e, "carpet_seg", text="Loop Density")
        col.prop(e, "carpet_seed", text="Seed")


classes = (DESKI_PT_prep, DESKI_PT_prep_detect, DESKI_PT_prep_nails, DESKI_PT_prep_boards,
           DESKI_PT_panel, DESKI_PT_scen, DESKI_PT_nails, DESKI_PT_quality, DESKI_PT_options, DESKI_PT_anim,
           DESKI_PT_details, *FEATURES, DESKI_PT_fx, DESKI_PT_fx_ice, DESKI_PT_fx_carpet)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
