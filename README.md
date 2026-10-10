# Axis Rupture

Texture-aware destruction for Blender. Axis Rupture reads the seams in your plank, tile or brick texture,
cuts the plane into real pieces along them, then bends, tears and breaks those pieces with a terrain or a scenario.
The output stays game-ready: UVs preserved, no n-gons, extra loops only where a piece actually bends.

## What it does

**1. Prepare: split along texture seams**
- Detects seams from all material maps (AO, Cavity, Roughness, Normal, height) or from the brightness of a single map
- Local calibration fits every seam and joint separately, because a photo is never a perfect grid
- Layouts: rows (planks, tiles, bricks), herringbone (angle and width detected from the texture), irregular stones and cobbles
- Surface presets for planks, tiles, bricks, roof tiles and pavers: thickness, gap, bevel and nails in one click
- Detects nail heads on the color texture and uses them when tearing

**2. Tear: bend and rip the pieces**
- A terrain mesh pushes the planks up from below to its current position
- Scenarios driven by an Empty or a mesh: moisture bulge, collapse into a hole, blast crater with ejected planks
- Nails hold and let go, planks bend at the nails and break at the edge of a hole
- Collision separation, intersection check and an optional bottom mesh for holes and craters
- Live Preview in draft or full quality, with Apply, Cancel and Restore Original Planks

**3. Details**
- Unevenness and warp of intact planks
- Splinters, cracks and deformation of torn planks
- Nails torn out together with the planks
- Burn or rot in an area or in noise patches, with a darkened copy of the material

**4. Effects**
- Cracking ice: Voronoi floes around an impact point, ice hole, sinking, rocking and an optional water plane
- Peeling carpet, linoleum or wallpaper: fold, roll or bubble

**Animation:** tearing can be baked over the scene frame range as object keys and absolute shape keys.

## Requirements

Blender 4.2 or newer. Developed and tested on Blender 5.0.

## Installation

1. Clone or download this repository.
2. Build the extension package:
   ```
   blender --command extension build --source-dir axis_rupture
   ```
   This creates `axis_rupture-1.2.0.zip`.
3. In Blender: **Edit > Preferences > Get Extensions > Install from Disk…** and pick the zip.

## Usage

Open the sidebar (`N`) in the 3D Viewport and go to the **Axis Rupture** tab. The panels follow the workflow:

1. **Prepare Planks:** pick a plane with UVs and a plank texture, check the result with **Preview Split**, then **Split into Planks**.
2. **Tear Planks:** pick the plank collection and a terrain mesh or a scenario Empty, then use **Live Preview** or **Tear Now**.
3. **Plank Details:** **Add Details** to the result.
4. **Effects:** **Break Ice** or **Peel Carpet** on separate planes.

The interface is in English. Code comments and some internal identifiers are in Polish.

## License

GPL-2.0-or-later. See [LICENSE](LICENSE).

## Author

Patryk Malarczyk, Technical Artist and co-founder of Orion Belt Games.
