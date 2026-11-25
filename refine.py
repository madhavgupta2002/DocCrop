"""Corner refinement: snap a rough quadrilateral onto the real edges of a print/document.

Technique ("classical colour-edge"):
  1. Crop the region around the rough quad (+8 % margin) and work at <= 2500 px.
  2. Estimate the object's interior colour = median Lab colour of the eroded rough quad.
  3. For every side, sample ~70 profiles perpendicular to the rough edge (band +-3 % of the crop).
     Every local gradient maximum on a profile is a candidate edge.  Score each candidate by
       gradient strength x similarity(inside colour, interior colour) x (1 + inside/outside contrast)
     so an edge whose inner side looks like the print wins over a parallel sleeve/pocket/page edge.
  4. RANSAC line fit per side through the winning points, intersect adjacent lines -> 4 corners.
  5. Sanity check (convex, area within +-20-25 % of the rough quad); otherwise keep the rough quad.

refine_quad(img, quad_px) -> (quad_px, flag) where flag is None or a short reason to review it.
"""
import numpy as np
import cv2

MAXC = 2500          # working resolution of the crop (long side)
MARGIN = 0.08        # crop margin around the rough quad, fraction of its size
BAND = 0.03          # search band either side of each rough edge, fraction of the crop
FLAG_MOVE = 0.02     # flag results whose corners moved more than this fraction of the crop


def order_quad(q):
    """Order 4 points as TL, TR, BR, BL."""
    q = np.asarray(q, float).reshape(4, 2)
    c = q.mean(0)
    q = q[np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))]
    return np.roll(q, -int(np.argmin(q.sum(1))), 0)


def _ransac_line(pts, iters=300, tol=1.5, seed=0):
    if len(pts) < 10:
        return None
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        a, b = pts[rng.choice(len(pts), 2, replace=False)]
        d = b - a
        L = np.linalg.norm(d)
        if L < 1e-6:
            continue
        n = np.array([-d[1], d[0]]) / L
        inl = np.abs((pts - a) @ n) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    vx, vy, x0, y0 = cv2.fitLine(pts[best].astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0]), np.array([vx, vy])


def _intersect(lines, fallback):
    out = []
    for k in range(4):
        if lines[k - 1] is None or lines[k] is None:
            out.append(fallback[k]); continue
        (p1, d1), (p2, d2) = lines[k - 1], lines[k]
        try:
            s = np.linalg.solve(np.array([d1, -d2]).T, p2 - p1)
            out.append(p1 + d1 * s[0])
        except np.linalg.LinAlgError:
            out.append(fallback[k])
    return np.array(out)


def refine_crop(crop, rough):
    """Core algorithm on a crop; rough = 4x2 quad in crop pixels."""
    h, w = crop.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(crop, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    g = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (0, 0), 1.5).astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    m = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(m, rough.astype(np.int32), 1)
    m = cv2.erode(m, np.ones((max(3, int(0.08 * min(h, w))),) * 2, np.uint8))
    interior = np.median(lab[m > 0], 0) if m.any() else lab.reshape(-1, 3).mean(0)
    c = rough.mean(0)
    band = int(BAND * max(h, w))
    off = np.arange(-band, band + 1)
    k_in = max(4, int(0.004 * max(h, w)))
    lines = []
    for k in range(4):
        p, r = rough[k], rough[(k + 1) % 4]
        d = r - p
        t = d / np.linalg.norm(d)
        n = np.array([-t[1], t[0]])
        if (c - (p + r) / 2) @ n < 0:
            n = -n  # n points into the print
        pts = []
        for s in np.linspace(0.1, 0.9, 70):
            xy = (p + d * s)[None] + off[:, None] * n[None]
            xi = np.clip(xy[:, 0].round().astype(int), 0, w - 1)
            yi = np.clip(xy[:, 1].round().astype(int), 0, h - 1)
            gr = np.abs(gx[yi, xi] * n[0] + gy[yi, xi] * n[1])
            thr = 0.3 * gr.max()
            best, bs = None, -1.0
            for j in range(1, len(off) - 1):
                if gr[j] < thr or gr[j] < gr[j - 1] or gr[j] < gr[j + 1]:
                    continue
                cin = lab[yi[min(len(off) - 1, j + k_in)], xi[min(len(off) - 1, j + k_in)]]
                cout = lab[yi[max(0, j - k_in)], xi[max(0, j - k_in)]]
                sc = gr[j] * (1.0 / (1.0 + np.linalg.norm(cin - interior) / 20)) * (1 + np.linalg.norm(cin - cout) / 30)
                if sc > bs:
                    bs, best = sc, xy[j]
            if best is not None:
                pts.append(best)
        lines.append(_ransac_line(np.array(pts)) if len(pts) > 10 else None)
    return _intersect(lines, rough)


def _valid(q, rough):
    q = q.astype(np.float32)
    if not np.all(np.isfinite(q)) or not cv2.isContourConvex(q.reshape(-1, 1, 2)):
        return False
    return 0.8 < cv2.contourArea(q) / max(cv2.contourArea(rough.astype(np.float32)), 1) < 1.25


def refine_quad(img, quad_px):
    """img: full-resolution BGR image, quad_px: 4x2 rough corners (TL,TR,BR,BL) in image pixels."""
    H, W = img.shape[:2]
    q = np.asarray(quad_px, float)
    mg = MARGIN * (q.max(0) - q.min(0))
    x0, y0 = np.maximum(0, q.min(0) - mg).astype(int)
    x1, y1 = np.minimum([W, H], q.max(0) + mg).astype(int)
    s = min(1.0, MAXC / max(x1 - x0, y1 - y0))
    crop = cv2.resize(img[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    rough = (q - [x0, y0]) * s
    try:
        r = refine_crop(crop, rough)
    except Exception as e:  # never lose the user's starting point
        return q, f'refine error: {e}'
    if not _valid(r, rough):
        return q, 'refine failed - kept previous corners'
    move = np.max(np.linalg.norm(r - rough, axis=1)) / max(crop.shape[:2])
    flag = f'corners moved {move * 100:.1f}% - please check' if move > FLAG_MOVE else None
    return r / s + [x0, y0], flag
