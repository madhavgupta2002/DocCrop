import cv2, numpy as np, json, os, sys, glob, hashlib, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

APP = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(APP, 'data')
PREV = os.path.join(DATA, 'previews')
STATE_F = os.path.join(DATA, 'state.json')
OW, OH = 2400, 1840
REF_QUAD = (np.float32([[58, 50], [888, 42], [852, 672], [215, 690]]) * 4 / [4000, 3000]).tolist()
LOCK = threading.RLock()

def pid(path): return hashlib.sha1(path.encode()).hexdigest()[:12]

def migrate():
    """First run: build state from the legacy name-keyed files if present."""
    src = os.environ.get('WB_SRC', '/Users/gupmadha/Desktop/whiteboardf/extracted')
    d = json.load(open(os.path.join(DATA, 'defaults.json'))) if os.path.exists(os.path.join(DATA, 'defaults.json')) else {}
    q = json.load(open(os.path.join(DATA, 'quads.json'))) if os.path.exists(os.path.join(DATA, 'quads.json')) else {}
    imgs = [{'path': p, 'default': d.get(os.path.basename(p)), 'quad': q.get(os.path.basename(p))}
            for p in sorted(glob.glob(os.path.join(src, '*.jpg')))]
    return {'images': imgs, 'settings': {}}

def load_state():
    if os.path.exists(STATE_F): return json.load(open(STATE_F))
    return migrate()

STATE = load_state()
def save_state():
    os.makedirs(DATA, exist_ok=True)
    with LOCK: json.dump(STATE, open(STATE_F, 'w'))
save_state()

def find(i):
    for o in STATE['images']:
        if pid(o['path']) == i: return o
    return None

def item_json(o):
    return {'id': pid(o['path']), 'name': os.path.basename(o['path']), 'path': o['path'],
            'quad': o.get('quad') or o.get('default') or REF_QUAD, 'default': o.get('default') or REF_QUAD,
            'edited': bool(o.get('quad')), 'missing': not os.path.exists(o['path'])}

def ensure_preview(o):
    p = os.path.join(PREV, pid(o['path']) + '.jpg')
    if not os.path.exists(p):
        os.makedirs(PREV, exist_ok=True)
        im = cv2.imread(o['path']); h, w = im.shape[:2]
        cv2.imwrite(p, cv2.resize(im, (1600, int(1600 * h / w)), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return p

def crop(path, quad, w_out=OW, h_out=OH):
    im = cv2.imread(path); h, w = im.shape[:2]
    src = (np.float32(quad) * [w, h]).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, np.float32([[0, 0], [w_out, 0], [w_out, h_out], [0, h_out]]))
    return cv2.warpPerspective(im, M, (w_out, h_out), flags=cv2.INTER_CUBIC)

def export_json():
    return {'format': 'whiteboard-cropper', 'version': 1,
            'note': 'quad = [TL,TR,BR,BL] as fractions (0-1) of image width/height; null = use auto default',
            'images': [{'path': o['path'], 'default': o.get('default'), 'quad': o.get('quad')} for o in STATE['images']]}

def import_json(data):
    if data.get('format') != 'whiteboard-cropper': raise ValueError('not a whiteboard-cropper export')
    idx = {o['path']: o for o in STATE['images']}
    added = updated = missing = 0
    with LOCK:
        for e in data.get('images', []):
            o = idx.get(e['path'])
            if o is None:
                if not os.path.exists(e['path']): missing += 1; continue
                o = {'path': e['path']}; STATE['images'].append(o); idx[e['path']] = o; added += 1
            else: updated += 1
            if e.get('default'): o['default'] = e['default']
            o['quad'] = e.get('quad')
    save_state()
    return {'added': added, 'updated': updated, 'skipped_missing_files': missing}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, code, body, ctype, extra=None):
        self.send_response(code); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(body)))
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)
    def js(self, obj, code=200): self.send(code, json.dumps(obj).encode(), 'application/json')
    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path == '/': return self.send(200, open(os.path.join(APP, 'index.html'), 'rb').read(), 'text/html')
        if u.path == '/api/state': return self.js({'images': [item_json(o) for o in STATE['images']], 'settings': STATE.get('settings', {})})
        if u.path == '/api/export':
            return self.send(200, json.dumps(export_json(), indent=1).encode(), 'application/json',
                             {'Content-Disposition': 'attachment; filename="whiteboard-cropper-export.json"'})
        if u.path.startswith('/img/'):
            o = find(u.path[5:]); return self.send(200, open(ensure_preview(o), 'rb').read(), 'image/jpeg') if o else self.send(404, b'', 'text/plain')
        if u.path.startswith('/full/'):
            o = find(u.path[6:]); return self.send(200, open(o['path'], 'rb').read(), 'image/jpeg') if o else self.send(404, b'', 'text/plain')
        if u.path == '/api/preview':
            o = find(q['id'][0]); ok, buf = cv2.imencode('.jpg', crop(o['path'], json.loads(q['quad'][0]), 900, 690), [cv2.IMWRITE_JPEG_QUALITY, 90])
            return self.send(200, buf.tobytes(), 'image/jpeg')
        self.send(404, b'not found', 'text/plain')
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])) or b'{}')
        try:
            if self.path == '/api/save':
                o = find(body['id']); o['quad'] = body.get('quad'); save_state(); return self.js({})
            if self.path == '/api/import': return self.js(import_json(body))
            if self.path == '/api/run': return self.js(run_all())
        except Exception as e:
            return self.js({'error': str(e)}, 400)
        self.send(404, b'', 'text/plain')

def run_all():
    out = STATE.get('settings', {}).get('out_dir') or os.path.join(APP, 'corrected')
    os.makedirs(out, exist_ok=True)
    for o in STATE['images']:
        cv2.imwrite(os.path.join(out, os.path.basename(o['path'])), crop(o['path'], o.get('quad') or o.get('default') or REF_QUAD), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return {'count': len(STATE['images']), 'dir': out}

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f'http://localhost:{port}'); ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
