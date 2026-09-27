# Deski - dodatek do Blendera (4.2+ / 5.x)
# przygotowanie.py - Przygotowanie desek: podzial plaszczyzny podlogi na deski wg szczelin na teksturze
# wyrwanie.py      - Wyrwanie desek: wyginanie / wyrywanie przybitych desek przez teren
# scenariusze.py   - scenariusze etapu 1: wilgoc (wybrzuszenie), zapadniecie (dziura), wybuch (krater)
# detale.py        - Detale desek: nierownosci, drzazgi, odksztalcenie wyrwanych desek, gwozdzie
# podglad.py       - Podglad na zywo: zmiany ustawien od razu widoczne, Zatwierdz = wynik finalny
# animacja.py      - Animacja wyrywania: stan fizyczny w czasie -> klucze obiektow / absolutne shape keys
# efekty.py        - efekty statyczne: pekajacy lod, odklejanie wykladziny
# ui.py            - panele: 1 Przygotowanie, 2 Wyrwanie, 3 Detale (+ zwijane podpanele)
# Uzycie: View3D > Sidebar (N) > zakladka "Deski"
bl_info = {
    "name": "Deski: wyrwanie przez teren",
    "author": "Bella",
    "version": (1, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Deski",
    "description": "Przygotowanie desek z plaszczyzny podlogi i ich wyginanie / wyrywanie przez teren",
    "category": "Object",
}

if "bpy" in locals():
    import importlib
    importlib.reload(jodelka)
    importlib.reload(przygotowanie)
    importlib.reload(scenariusze)
    importlib.reload(wyrwanie)
    importlib.reload(detale)
    importlib.reload(podglad)
    importlib.reload(animacja)
    importlib.reload(efekty)
    importlib.reload(ui)
else:
    from . import jodelka, przygotowanie, scenariusze, wyrwanie, detale, podglad, animacja, efekty, ui

import bpy

modules = (przygotowanie, wyrwanie, detale, podglad, animacja, efekty, ui)


def register():
    for m in modules:
        m.register()


def unregister():
    for m in reversed(modules):
        m.unregister()
