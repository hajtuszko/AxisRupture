# Panele dodatku (View3D > N > Deski): trzy kroki w kolejnosci pracy, glowne przyciski wyroznione,
# ustawienia w zwijanych podpanelach (zaawansowane domyslnie zwiniete).
import bpy
from bpy.types import Panel

SPACE = dict(bl_space_type='VIEW_3D', bl_region_type='UI', bl_category="Deski")
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


# ================================================================ 1. PRZYGOTOWANIE
class DESKI_PT_prep(Panel):
    bl_label = "1  Przygotowanie desek"
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
        col.prop(p, "source", text="Płaszczyzna")
        col.prop(p, "texture", text="Tekstura")
        col.prop(p, "output_name", text="Wynik")
        col.prop(p, "surface", text="Powierzchnia")
        L.row().prop(p, "layout", expand=True)
        L.separator()
        if pz.JOB["running"]:
            pz.draw_job(L, pz.DESKI_OT_split.bl_idname, 'MOD_EDGESPLIT')
            return
        big(L, pz.DESKI_OT_split.bl_idname, "Podziel na elementy" if p.surface != 'DESKI' else "Podziel na deski",
            'MOD_EDGESPLIT', enabled=p.source is not None)
        if p.layout == 'HERRINGBONE':
            r = L.row(); r.enabled = p.source is not None
            r.operator(pz.DESKI_OT_split_preview.bl_idname, text="Podgląd podziału (obrysy)", icon='HIDE_OFF')


class DESKI_PT_prep_detect(Panel):
    bl_label = "Wykrywanie szczelin"
    bl_parent_id = "DESKI_PT_prep"
    bl_space_type, bl_region_type, bl_category = SPACE.values()

    def draw(self, context):
        p = context.scene.deski_przygotowanie
        col = split(self.layout.column())
        if p.layout == 'HERRINGBONE':
            col.prop(p, "herr_length", text="Długość")
            col.prop(p, "herr_variant", text="Wariant")
            col.prop(p, "herr_offset", text="Przesunięcie")
            col.prop(p, "herr_calibrate", text="Kalibracja lokalna")
            col.label(text="Długość 0 = auto, kąt/szer. auto", icon='INFO')
        else:
            col.prop(p, "straight_method", text="Metoda")
            sub = col.column(); sub.enabled = p.straight_method == 'MAPS'
            sub.prop(p, "straight_calibrate", text="Kalibracja lokalna")
            col.separator()
            col.prop(p, "sensitivity", text="Czułość")
            col.prop(p, "min_width", text="Min. szerokość")
            col.prop(p, "min_length", text="Min. długość")
            col.prop(p, "joint_contrast", text="Łącz. kontrast")
            col.prop(p, "joint_cover", text="Łącz. pokrycie")


class DESKI_PT_prep_nails(Panel):
    bl_label = "Gwoździe z tekstury"
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
        col.prop(p, "nail_texture", text="Tekstura koloru")
        col.prop(p, "nail_dark", text="Kontrast")


class DESKI_PT_prep_boards(Panel):
    bl_label = "Wymiary desek"
    bl_parent_id = "DESKI_PT_prep"
    bl_options = {'DEFAULT_CLOSED'}
    bl_space_type, bl_region_type, bl_category = SPACE.values()

    def draw(self, context):
        p = context.scene.deski_przygotowanie
        col = split(self.layout.column())
        if p.layout == 'STRAIGHT':
            col.prop(p, "max_length", text="Maks. długość")
        col.prop(p, "thickness", text="Grubość")
        col.prop(p, "gap", text="Szczelina")
        col.prop(p, "chamfer", text="Faza krawędzi")
        col.prop(p, "min_piece", text="Min. kawałek")
        col.prop(p, "hide_source", text="Ukryj płaszczyznę")


# ================================================================ 2. WYRWANIE
SCEN_ICON = {'TEREN': 'MESH_GRID', 'WILGOC': 'MOD_FLUIDSIM', 'ZAPADNIECIE': 'SORT_DESC', 'WYBUCH': 'FORCE_FORCE'}


class DESKI_PT_panel(Panel):
    bl_label = "2  Wyrwanie desek"
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
            col.prop(p, "terrain", text="Teren")
        else:
            col.prop(p, "emitter", text="Obszar (Empty)")
        r = col.row(); r.enabled = not pg.active
        r.prop(p, "boards", text="Deski")
        L.separator()
        ready = (p.terrain if p.scenario == 'TEREN' else p.emitter) is not None and p.boards is not None
        if not pg.active:
            big(L, podglad.DESKI_OT_live_start.bl_idname, "Podgląd na żywo", 'HIDE_OFF', enabled=ready)
            r = L.row(); r.enabled = ready
            r.operator(wyrwanie.DESKI_OT_deform.bl_idname, text="Wyrwij od razu", icon='MOD_SIMPLEDEFORM')
            return
        # --- trwa podglad
        box = L.box()
        head = box.row(); head.alert = True
        head.label(text="PODGLĄD NA ŻYWO", icon=icon('REC', 'RADIOBUT_ON'))
        box.row().prop(pg, "mode", expand=True)
        if pg.mode == 'SZKIC':
            leg = box.row(align=True); leg.scale_y = 0.8
            leg.label(text="przybita", icon=icon('STRIP_COLOR_04', 'SEQUENCE_COLOR_04', 'COLORSET_03_VEC'))
            leg.label(text="luźna", icon=icon('STRIP_COLOR_03', 'SEQUENCE_COLOR_03', 'COLORSET_09_VEC'))
            leg.label(text="wyrwana", icon=icon('STRIP_COLOR_01', 'SEQUENCE_COLOR_01', 'COLORSET_01_VEC'))
        else:
            box.prop(pg, "with_details", text="Z detalami (wolniej)")
        r = box.row(); r.prop(pg, "auto", text="Przeliczaj od razu")
        if not pg.auto:
            r.operator(podglad.DESKI_OT_live_refresh.bl_idname, text="Przelicz", icon='FILE_REFRESH')
        if pg.message:
            msg = box.column(align=True); msg.scale_y = 0.8
            for part in pg.message.split(" | "):
                msg.label(text=part)
        row = box.row(align=True); row.scale_y = 1.8
        row.operator(podglad.DESKI_OT_live_accept.bl_idname, text="Zatwierdź", icon='CHECKMARK')
        row.operator(podglad.DESKI_OT_live_cancel.bl_idname, text="Anuluj", icon='X')


class DESKI_PT_scen(Panel):
    bl_label = "Parametry scenariusza"
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
            col.prop(p, "bulge_height", text="Wysokość wybrzuszenia")
        elif p.scenario == 'ZAPADNIECIE':
            col.prop(p, "hole_depth", text="Głębokość dziury")
            col.prop(p, "hang_max", text="Maks. zwis (°)")
            col.prop(p, "make_ground", text="Utwórz dno")
        else:
            col.prop(p, "blast_strength", text="Siła podmuchu")
            col.prop(p, "peel_max", text="Maks. odgięcie (°)")
            col.prop(p, "throw_zone", text="Strefa wyrzucenia")
            col.prop(p, "crater_size", text="Rozmiar krateru")
            col.prop(p, "hole_depth", text="Głębokość krateru")
            col.prop(p, "make_ground", text="Utwórz dno")
        col.label(text="Promień = rozmiar Empty", icon='INFO')


class DESKI_PT_nails(Panel):
    bl_label = "Gwoździe i zerwanie"
    bl_parent_id = "DESKI_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    bl_space_type, bl_region_type, bl_category = SPACE.values()

    def draw(self, context):
        p = context.scene.deski_teren
        col = split(self.layout.column())
        col.prop(p, "seed", text="Seed")
        col.prop(p, "nail_position", text="Gwoździe od końca")
        col.prop(p, "bend_min_length", text="Gięcie od długości")
        col.separator()
        for a, b, t in (("bulge_min", "bulge_max", "Ugięcie"), ("slope_min", "slope_max", "Kąt przy gwoździu"),
                        ("angle2_min", "angle2_max", "Kąt zerwania 2. gw.")):
            sub = col.column(align=True)
            sub.prop(p, a, text=f"{t} min"); sub.prop(p, b, text="max")


class DESKI_PT_quality(Panel):
    bl_label = "Jakość i geometria"
    bl_parent_id = "DESKI_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    bl_space_type, bl_region_type, bl_category = SPACE.values()

    def draw(self, context):
        p = context.scene.deski_teren
        col = split(self.layout.column())
        col.prop(p, "segment_length", text="Długość segmentu")
        sub = col.column(align=True)
        sub.prop(p, "bevel_width", text="Bevel: szerokość")
        sub.prop(p, "bevel_segments", text="segmenty")
        col.prop(p, "roll_max", text="Skręt w poprzek")
        col.prop(p, "steps", text="Kroki symulacji")
        col.prop(p, "clearance", text="Luz")
        col.prop(p, "smooth_shading", text="Gładki łuk")


class DESKI_PT_options(Panel):
    bl_label = "Opcje"
    bl_parent_id = "DESKI_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    bl_space_type, bl_region_type, bl_category = SPACE.values()

    def draw(self, context):
        p = context.scene.deski_teren
        col = split(self.layout.column())
        col.prop(p, "resolve_collisions", text="Rozsuń kolizje")
        col.prop(p, "verify", text="Sprawdź przenikania")
        col.prop(p, "remove_rigidbody", text="Usuń Rigid Body")


class DESKI_PT_anim(Panel):
    bl_label = "Animacja"
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
        r = col.row(align=True); r.prop(sc, "frame_start", text="Klatki"); r.prop(sc, "frame_end", text="")
        col.prop(p, "anim_step", text="Co ile klatek")
        col.prop(p, "anim_loops", text="Pętle animacji")
        big(L, animacja.DESKI_OT_animate.bl_idname, "Animuj wyrywanie", 'PLAY', scale=1.5,
            enabled=not live_active(context))
        L.operator(animacja.DESKI_OT_animate_clear.bl_idname, text="Usuń animację", icon='TRASH')


# ================================================================ 3. DETALE
class DESKI_PT_details(Panel):
    bl_label = "3  Detale desek"
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
        r = col.row(); r.enabled = not live_active(context)
        r.prop(p, "boards", text="Deski")
        col.prop(p, "seed", text="Seed")
        L.separator()
        if live_active(context):
            L.label(text="Podgląd Pełny liczy detale", icon='INFO')
        big(L, detale.DESKI_OT_details.bl_idname, "Dodaj detale", 'MOD_NOISE', enabled=not live_active(context))


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
        "bl_region_type": 'UI', "bl_category": "Deski", "bl_options": {'DEFAULT_CLOSED'} if closed else set(),
        "draw_header": draw_header, "draw": draw})


def _decay_props(p):
    out = [("decay_type", "Rodzaj"), ("decay_mode", "Obszar")]
    out.append(("decay_source", "Źródło (Empty)") if p.decay_mode == 'OBSZAR' else ("decay_coverage", "Pokrycie"))
    out += [("decay_scale", "Skala plam"), ("decay_erode", "Zjedzenie krawędzi"), ("decay_thin", "Pocienienie"),
            ("decay_seg", "Gęstość pętli"), ("decay_material", "Materiał z efektem")]
    return out


FEATURES = [
    _feature("use_jitter", "Nierówności nietkniętych", [("jitter_lift", "Maks. uniesienie"), ("jitter_tilt", "Maks. przechył (°)"),
                                                        ("jitter_shift", "Przesunięcie końca")]),
    _feature("use_twist_nailed", "Zwichrowanie nietkniętych", [("twist_nailed", "Maks. skręt (°)")]),
    _feature("use_splinters", "Drzazgi na końcach", [("splinter_chance", "Szansa"), ("splinter_depth", "Głębokość"),
                                                     ("splinter_teeth_min", "Zęby min"), ("splinter_teeth_max", "Zęby max"),
                                                     ("splinter_out", "Wystająca drzazga")]),
    _feature("use_warp", "Odkształcenie wyrwanych", [("warp_bow", "Wygięcie (długości)"), ("warp_twist", "Skręt (°)"),
                                                    ("warp_loops", "Pętle")]),
    _feature("use_crack", "Pęknięcia", [("crack_chance", "Szansa"), ("crack_min_length", "Min. długość deski"),
                                       ("crack_angle", "Maks. odgięcie (°)"), ("crack_jag", "Poszarpanie")]),
    _feature("use_nails", "Gwoździe z deskami", [("nail_chance", "Szansa"), ("nail_length", "Długość"),
                                                ("nail_diameter", "Średnica"), ("nail_head", "Średnica łba"),
                                                ("nail_tilt", "Maks. odchylenie (°)")]),
    _feature("use_decay", "Spalenie / zbutwienie", None, extra=_decay_props),
]

classes = (DESKI_PT_prep, DESKI_PT_prep_detect, DESKI_PT_prep_nails, DESKI_PT_prep_boards,
           DESKI_PT_panel, DESKI_PT_scen, DESKI_PT_nails, DESKI_PT_quality, DESKI_PT_options, DESKI_PT_anim,
           DESKI_PT_details, *FEATURES)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
