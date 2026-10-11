# whiteboard-cropper

Local web app for perspective-cropping a series of whiteboard photos so they line up for a timelapse.
Original colors are preserved (no filter).

    python3 server.py [port]     # default 8765, then open http://localhost:8765

Requires `opencv-python` and `numpy`.

Keys: ←/→ previous/next photo, `0` free mode, `1`-`4` edit TL/TR/BR/BL with zoom, `+`/`-` zoom, `P` preview.
