import cv2, numpy as np, json, os, sys, glob, hashlib, threading, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import align
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
STATE.setdefault('reference', {'path': '/Users/gupmadha/Desktop/whiteboardf/extracted/20260216_085334.jpg', 'quad': REF_QUAD})
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
            'edited': bool(o.get('quad')), 'auto': bool(o.get('default')), 'missing': not os.path.exists(o['path'])}

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
    return {'format': 'doccrop', 'version': 1,
            'note': 'quad = [TL,TR,BR,BL] as fractions (0-1) of image width/height; null = use auto default',
            'images': [{'path': o['path'], 'default': o.get('default'), 'quad': o.get('quad')} for o in STATE['images']]}

def import_json(data):
    if data.get('format') not in ('doccrop', 'whiteboard-cropper'): raise ValueError('not a DocCrop export')
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

IMG_EXT = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff')
JOB = {'state': 'idle'}

def start_job(kind, total, fn):
    with LOCK:
        if JOB.get('state') == 'running': raise RuntimeError('another job is already running')
        JOB.clear(); JOB.update({'kind': kind, 'state': 'running', 'total': total, 'done': 0, 'current': '', 'errors': [], 'started': time.time(), 'cancel': False})
    def work():
        try: fn()
        except Exception as e: JOB['errors'].append(str(e))
        finally: JOB['state'] = 'cancelled' if JOB.get('cancel') else 'done'
    threading.Thread(target=work, daemon=True).start()

def add_paths(paths):
    have = {o['path'] for o in STATE['images']}; added = 0; skipped = 0
    with LOCK:
        for p in paths:
            p = os.path.abspath(p)
            if p in have or not p.lower().endswith(IMG_EXT) or not os.path.isfile(p): skipped += 1; continue
            STATE['images'].append({'path': p}); have.add(p); added += 1
        STATE['images'].sort(key=lambda o: o['path'])
    save_state()
    return {'added': added, 'skipped': skipped}

def pick_files():
    script = ['set fs to choose file with prompt "Select whiteboard photos" of type {"public.image"} with multiple selections allowed',
              'set out to ""', 'repeat with f in fs', 'set out to out & POSIX path of f & linefeed', 'end repeat', 'return out']
    r = subprocess.run(['osascript'] + [x for l in script for x in ('-e', l)], capture_output=True, text=True)
    if r.returncode != 0: return []  # cancelled
    return [l for l in r.stdout.splitlines() if l.strip()]

def autocrop(scope):
    ref = STATE['reference']
    targets = [o for o in STATE['images'] if scope == 'all' or not o.get('default')]
    al = align.Aligner(ref['path'], ref['quad'])
    def fn():
        for o in targets:
            if JOB['cancel']: break
            JOB['current'] = os.path.basename(o['path'])
            try: o['default'], _ = al.quad_for(o['path'])
            except Exception as e: JOB['errors'].append(f"{os.path.basename(o['path'])}: {e}")
            JOB['done'] += 1; save_state()
    start_job('autocrop', len(targets), fn)

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
        if u.path == '/api/state': return self.js({'images': [item_json(o) for o in STATE['images']], 'settings': {'out_dir': STATE.get('settings', {}).get('out_dir') or DEFAULT_OUT}, 'reference': STATE['reference']})
        if u.path == '/api/job': return self.js({k: v for k, v in JOB.items() if k != 'started'})
        if u.path == '/api/export':
            return self.send(200, json.dumps(export_json(), indent=1).encode(), 'application/json',
                             {'Content-Disposition': 'attachment; filename="doccrop-export.json"'})
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
            if self.path == '/api/pick': return self.js(add_paths(pick_files()))
            if self.path == '/api/add': return self.js(add_paths(body['paths']))
            if self.path == '/api/autocrop': autocrop(body.get('scope', 'new')); return self.js({})
            if self.path == '/api/cancel': JOB['cancel'] = True; return self.js({})
            if self.path == '/api/reference':
                o = find(body['id']); STATE['reference'] = {'path': o['path'], 'quad': o.get('quad') or o.get('default') or REF_QUAD}; save_state(); return self.js({})
            if self.path == '/api/run': run_all(body.get('out_dir') or STATE.get('settings', {}).get('out_dir') or DEFAULT_OUT); return self.js({})
            if self.path == '/api/pickdir': return self.js({'path': pick_dir()})
        except Exception as e:
            return self.js({'error': str(e)}, 400)
        self.send(404, b'', 'text/plain')

DEFAULT_OUT = '/Users/gupmadha/Desktop/whiteboardf/corrected'

def pick_dir():
    r = subprocess.run(['osascript', '-e', 'POSIX path of (choose folder with prompt "Select output folder")'], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None

def run_all(out):
    out = os.path.abspath(os.path.expanduser(out))
    os.makedirs(out, exist_ok=True)
    STATE.setdefault('settings', {})['out_dir'] = out; save_state()
    items = list(STATE['images'])
    seen = {}
    for o in items:  # disambiguate equal basenames from different folders
        b = os.path.basename(o['path']); seen[b] = seen.get(b, 0) + 1
    def fn():
        for o in items:
            if JOB['cancel']: break
            b = os.path.basename(o['path']); JOB['current'] = b
            name = b if seen[b] == 1 else f"{os.path.basename(os.path.dirname(o['path']))}_{b}"
            try:
                cv2.imwrite(os.path.join(out, name), crop(o['path'], o.get('quad') or o.get('default') or REF_QUAD), [cv2.IMWRITE_JPEG_QUALITY, 95])
            except Exception as e: JOB['errors'].append(f'{b}: {e}')
            JOB['done'] += 1
        JOB['out_dir'] = out
    start_job('run', len(items), fn)

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f'http://localhost:{port}'); ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
