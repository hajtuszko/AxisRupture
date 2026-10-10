# Axis Rupture - Blender add-on (4.2+ / 5.x)
# Splits floor, wall and roof planes into planks, tiles or bricks along seams detected on the texture,
# then bends, tears and breaks them.
# przygotowanie.py - Prepare: split a plane into separate pieces along texture seams (UVs preserved)
# wyrwanie.py      - Tear: bend / tear nailed planks with a terrain or a scenario
# scenariusze.py   - scenarios: moisture (bulge), collapse (hole), blast (crater)
# detale.py        - Details: unevenness, splinters, cracks, nails, burn / rot
# podglad.py       - Live Preview: setting changes recompute instantly, Apply = final result
# animacja.py      - Animate tearing: object keys / absolute shape keys over time
# efekty.py        - static effects: cracking ice, peeling carpet
# ui.py            - panels: 1 Prepare, 2 Tear, 3 Details, 4 Effects
# Usage: View3D > Sidebar (N) > "Axis Rupture" tab
bl_info = {
    "name": "Axis Rupture",
    "author": "Patryk Malarczyk",
    "version": (1, 2, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Axis Rupture",
    "description": "Split floor, wall and roof planes into planks, tiles or bricks along texture seams, then bend, tear and break them",
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
