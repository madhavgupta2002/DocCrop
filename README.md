# whiteboard-cropper

Local web app for perspective-cropping a series of whiteboard photos so they line up for a timelapse.
Original colors are preserved (no filter).

    python3 server.py [port]     # default 8765, then open http://localhost:8765

Requires `opencv-python` and `numpy`.

Keys: ←/→ previous/next photo, `0` free mode, `1`-`4` edit TL/TR/BR/BL with zoom, `+`/`-` zoom, `P` preview.

## Export / import
`Export` downloads a JSON with the absolute local image path and the 4 corner points (fractions of width/height, order TL,TR,BR,BL) per photo.
`Import` merges such a file: known paths get their points updated, unknown paths that exist on disk are added.
State is kept in `data/state.json`.

## Adding photos + auto-crop
`Add images…` opens the native macOS file dialog (multi-select) and adds the chosen photos by path.
`Auto-crop new` registers photos that have no starting crop against the reference photo and maps the reference crop onto them;
a progress dialog is shown. `Set as reference` makes the current photo (with its current crop) the reference.

## Running the cropper
`Run cropper` opens a dialog to type or browse (native folder dialog) for the output folder (remembered in `data/state.json`),
then crops every photo from its full-resolution original (2400x1840, original colors) with a progress bar and Cancel button.
