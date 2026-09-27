# Deski - dodatek do Blendera (4.2+ / 5.x)
# przygotowanie.py - Przygotowanie desek: podzial plaszczyzny podlogi na deski wg szczelin na teksturze
# wyrwanie.py      - Wyrwanie desek: wyginanie / wyrywanie przybitych desek przez teren
# scenariusze.py   - scenariusze etapu 1: wilgoc (wybrzuszenie), zapadniecie (dziura), wybuch (krater)
# detale.py        - Detale desek: nierownosci, drzazgi, odksztalcenie wyrwanych desek, gwozdzie
# podglad.py       - Podglad na zywo: zmiany ustawien od razu widoczne, Zatwierdz = wynik finalny
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
else:
    from . import jodelka, przygotowanie, scenariusze, wyrwanie, detale, podglad

import bpy

modules = (przygotowanie, wyrwanie, detale, podglad)


def register():
    for m in modules:
        m.register()


def unregister():
    for m in reversed(modules):
        m.unregister()
