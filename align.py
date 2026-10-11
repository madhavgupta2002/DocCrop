"""Auto-cropper: registers a photo to a reference photo (SIFT init + ECC on blue/edge maps)
and maps the reference's crop quad into the photo, giving a starting point for manual correction."""
import cv2, numpy as np

W = 800  # working width

def _fmap(im, Hs):
    im = cv2.resize(im, (W, Hs), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(cv2.GaussianBlur(im, (0, 0), 1.5), cv2.COLOR_BGR2HSV)
    blue = ((hsv[..., 0] > 100) & (hsv[..., 0] < 125) & (hsv[..., 1] > 110) & (hsv[..., 2] > 70)).astype(np.float32)
    g = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 2)
    mag = np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1) ** 2)
    return blue, mag / (mag.max() + 1e-6)

class Aligner:
    def __init__(self, ref_path, ref_quad):
        self.ref = cv2.imread(ref_path)
        self.H0, self.W0 = self.ref.shape[:2]
        self.Hs = int(self.H0 * W / self.W0)
        self.ref_quad = np.float32(ref_quad)  # normalized
        rb, rm = _fmap(self.ref, self.Hs)
        self.rtarget = rb + rm * 2.0
        self.sift = cv2.SIFT_create(4000)
        g = cv2.cvtColor(cv2.resize(self.ref, (W, self.Hs)), cv2.COLOR_BGR2GRAY)
        self.rk, self.rd = self.sift.detectAndCompute(g, None)

    def _sift_init(self, im):
        g = cv2.cvtColor(cv2.resize(im, (W, self.Hs)), cv2.COLOR_BGR2GRAY)
        k, d = self.sift.detectAndCompute(g, None)
        if d is None or self.rd is None or len(k) < 10: return None
        m = cv2.BFMatcher().knnMatch(d, self.rd, k=2)
        good = [p[0] for p in m if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
        if len(good) < 10: return None
        Hm, mask = cv2.findHomography(np.float32([k[x.queryIdx].pt for x in good]), np.float32([self.rk[x.trainIdx].pt for x in good]), cv2.RANSAC, 6)
        return Hm if Hm is not None and mask.sum() >= 12 else None

    def _ecc(self, cur, H):
        Hm = H.astype(np.float32); cc = 0
        for s, it in [(0.25, 150), (0.5, 100), (1.0, 60)]:
            S = np.diag([s, s, 1]).astype(np.float32); Si = np.linalg.inv(S)
            Hl = S @ Hm @ Si
            t = cv2.GaussianBlur(cv2.resize(self.rtarget, None, fx=s, fy=s), (0, 0), 1.5)
            i = cv2.GaussianBlur(cv2.resize(cur, None, fx=s, fy=s), (0, 0), 1.5)
            try:
                cc, Hl = cv2.findTransformECC(t, i, Hl, cv2.MOTION_HOMOGRAPHY, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, it, 1e-6), None, 5)
            except cv2.error:
                return None, 0
            Hm = Si @ Hl @ S
        return Hm, cc

    def quad_for(self, path):
        """Returns (normalized quad [TL,TR,BR,BL], ecc score). Falls back to the reference quad if alignment fails."""
        im = cv2.imread(path)
        if im is None: return self.ref_quad.tolist(), 0.0
        h, w = im.shape[:2]
        blue, mag = _fmap(im, self.Hs)
        cur = blue + mag * 2.0
        inits = [self._sift_init(im)]
        cands = []
        for init in [x for x in inits if x is not None] + [np.eye(3, dtype=np.float32)]:
            Hm, cc = self._ecc(cur, np.linalg.inv(init))
            if Hm is not None: cands.append((Hm, cc))
        if not cands: return self.ref_quad.tolist(), 0.0
        Hm, cc = max(cands, key=lambda x: x[1])
        A = np.diag([W / w, self.Hs / h, 1.0]); B = np.diag([W / self.W0, self.Hs / self.H0, 1.0])
        Hfull = np.linalg.inv(B) @ np.linalg.inv(Hm) @ A  # photo px -> reference px
        q_ref_px = (self.ref_quad * [self.W0, self.H0]).reshape(-1, 1, 2).astype(np.float32)
        q = cv2.perspectiveTransform(q_ref_px, np.linalg.inv(Hfull)).reshape(-1, 2) / [w, h]
        return q.tolist(), float(cc)
