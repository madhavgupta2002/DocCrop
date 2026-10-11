import cv2, numpy as np, json, os, sys, glob
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

APP = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(APP, 'data')
SRC = os.environ.get('WB_SRC', '/Users/gupmadha/Desktop/whiteboardf/extracted')
OUT = os.environ.get('WB_OUT', '/Users/gupmadha/Desktop/whiteboardf/corrected')
PREV = os.path.join(DATA, 'previews')
QF = os.path.join(DATA, 'quads.json')
OW, OH = 2400, 1840
REF_QUAD = (np.float32([[58, 50], [888, 42], [852, 672], [215, 690]]) * 4 / [4000, 3000]).tolist()
DEFAULTS = json.load(open(os.path.join(DATA, 'defaults.json')))
names = sorted(os.path.basename(f) for f in glob.glob(os.path.join(SRC, '*.jpg')))

def ensure_preview(n):
    p = os.path.join(PREV, n)
    if not os.path.exists(p):
        im = cv2.imread(os.path.join(SRC, n)); h, w = im.shape[:2]
        cv2.imwrite(p, cv2.resize(im, (1600, int(1600 * h / w)), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return p

def load_saved():
    return json.load(open(QF)) if os.path.exists(QF) else {}

def crop(n, quad, w_out=OW, h_out=OH):
    im = cv2.imread(os.path.join(SRC, n)); h, w = im.shape[:2]
    src = (np.float32(quad) * [w, h]).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, np.float32([[0, 0], [w_out, 0], [w_out, h_out], [0, h_out]]))
    return cv2.warpPerspective(im, M, (w_out, h_out), flags=cv2.INTER_CUBIC)

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, code, body, ctype):
        self.send_response(code); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == '/':
            return self.send(200, open(os.path.join(APP, 'index.html'), 'rb').read(), 'text/html')
        if u.path == '/api/images':
            saved = load_saved()
            return self.send(200, json.dumps([{'name': n, 'quad': saved.get(n) or DEFAULTS.get(n) or REF_QUAD,
                                               'default': DEFAULTS.get(n) or REF_QUAD, 'edited': n in saved} for n in names]).encode(), 'application/json')
        if u.path.startswith('/img/'):
            n = unquote(u.path[5:]); return self.send(200, open(ensure_preview(n), 'rb').read(), 'image/jpeg')
        if u.path.startswith('/full/'):
            n = unquote(u.path[6:]); return self.send(200, open(os.path.join(SRC, n), 'rb').read(), 'image/jpeg')
        if u.path == '/api/preview':
            q = parse_qs(u.query); n = q['name'][0]; quad = json.loads(q['quad'][0])
            ok, buf = cv2.imencode('.jpg', crop(n, quad, 900, 690), [cv2.IMWRITE_JPEG_QUALITY, 90])
            return self.send(200, buf.tobytes(), 'image/jpeg')
        self.send(404, b'not found', 'text/plain')
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])) or b'{}')
        if self.path == '/api/save':
            s = load_saved()
            if body.get('quad') is None: s.pop(body['name'], None)
            else: s[body['name']] = body['quad']
            json.dump(s, open(QF, 'w')); return self.send(200, b'{}', 'application/json')
        if self.path == '/api/run':
            os.makedirs(OUT, exist_ok=True); s = load_saved()
            for n in names:
                cv2.imwrite(os.path.join(OUT, n), crop(n, s.get(n) or DEFAULTS.get(n) or REF_QUAD), [cv2.IMWRITE_JPEG_QUALITY, 95])
            return self.send(200, json.dumps({'count': len(names), 'dir': OUT}).encode(), 'application/json')
        self.send(404, b'', 'text/plain')

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f'http://localhost:{port}'); ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
