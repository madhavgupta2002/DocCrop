# DocCrop

Local web app for perspective-cropping photos: whiteboard timelapse series, documents, or
**several prints per page out of photos of old photo albums**. Original colours are preserved (no filter).

    python3 server.py [port]     # default 8765, then open http://localhost:8765 (binds to 127.0.0.1 only)

Requires `opencv-python` and `numpy`.

## Model
Every item is one *crop region* on a source photo. A photo can have several regions (`a`, `b`, `c`, …),
e.g. the 2–3 prints on an album page. Each region has 4 corners (TL, TR, BR, BL), an optional output
rotation and a "✓ checked" flag. State lives in `data/state.json` (v1 one-crop-per-photo state is migrated automatically).

## Keys
| Key | Action |
|---|---|
| ←/→ (↑/↓) | previous / next region |
| `0` | free mode: drag the nearest corner dot |
| `1`–`4` | zoom onto TL/TR/BR/BL and click the true corner (auto-advances to the next corner) |
| Alt+arrows | nudge the last-used corner by 1 px (Alt+Shift = 10 px) |
| `+` / `-` | zoom |
| `Enter` | mark ✓ checked and go to the next region |
| `R` | rotate the output 90° |
| `N` / `X` | add / delete a region on this photo |
| `F` | refine this region's corners |
| `P` | refresh the crop preview |

Other regions on the same photo are drawn as dashed blue outlines. Sidebar colours: green = checked,
yellow = edited, orange ⚠ = flagged by the refiner for a look.

## Getting starting corners
**Auto-crop (photo series from the same spot)** — `align.py`. Crop one photo, *Set as reference*, then
*Auto-crop new*: every new photo is registered to the reference (SIFT init + ECC on blue/edge maps) and the
reference crop is mapped onto it.

**Refine corners (prints / documents)** — `refine.py`. Start from rough corners (dragged by hand,
imported, or from a detector) and press `F` (this region) or *Refine all unedited*. The refiner snaps each
side onto the real edge of the print:

1. crop around the rough quad (+8 %) and work at ≤ 2500 px;
2. estimate the print's interior colour (median Lab inside the eroded quad);
3. along ~70 profiles perpendicular to each side, every gradient peak within ±3 % is a candidate edge,
   scored by *gradient × similarity(inside colour, interior colour) × (1 + inside/outside contrast)* —
   so the real print edge beats a parallel plastic-sleeve / pocket / page edge;
4. RANSAC line per side, intersect neighbouring lines → corners;
5. sanity checks (convex, area within ±20 %); otherwise the previous corners are kept.

Results that move more than 2 % are flagged ⚠ for review; *Use old corners* restores the pre-refine corners.
Edited or checked regions are never touched by *Refine all unedited*.

## Export / import
`Export` downloads JSON with the absolute image path, region index `n`, rotation and the 4 corners
(fractions of width/height, order TL,TR,BR,BL). `Import` merges such a file (v1 exports still import).

## Adding photos
`Add images…` opens the native macOS file dialog (multi-select) and adds the chosen photos by path.

## Running the cropper
`Run cropper…` asks for an output folder (remembered), format (lossless PNG or JPG 95) and size:
leave *Size* blank for each crop's natural full-resolution size, or give e.g. `2400x1840` to force one size
(timelapse). Output: `<photo>.png`, or `<photo>_a.png`, `<photo>_b.png` … for photos with several regions;
one subfolder per source folder when photos come from several folders.
