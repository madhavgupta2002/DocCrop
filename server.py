import cv2, numpy as np, json, os, sys, glob, threading, subprocess, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

"""DocCrop - local web app for perspective-cropping photos.

Every item is one crop region on one source photo: {path, n, default, quad, rot, done, flag}.
  quad / default : [TL,TR,BR,BL] as fractions (0-1) of the photo width/height
                   (quad = your manual corners, default = automatic starting point)
  n              : index of the crop region on that photo (several prints per album page)
  rot            : extra clockwise rotation of the output (0/90/180/270)

Two ways to get starting corners:
  * Auto-crop (align.py)  - photo series shot from the same spot (e.g. whiteboard timelapse):
                            registers each photo to a reference and maps the reference crop.
  * Refine corners (refine.py) - snaps rough corners onto the real print/document edges.
"""

APP = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(APP, 'data')
PREV = os.path.join(DATA, 'previews')
STATE_F = os.path.join(DATA, 'state.json')
DEFAULT_OUT = os.environ.get('DOCCROP_OUT', os.path.join(os.path.expanduser('~'), 'Desktop', 'DocCrop-output'))
CENTER_QUAD = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
LOCK = threading.RLock()
IMG_EXT = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff')
ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def pid(o): return hashlib.sha1(f"{o['path']}#{o.get('n', 0)}".encode()).hexdigest()[:12]
def page_id(path): return hashlib.sha1(path.encode()).hexdigest()[:12]


def migrate_legacy():
    """First run: build state from the legacy name-keyed files (data/defaults.json, data/quads.json)."""
    src = os.environ.get('WB_SRC')
    d = json.load(open(os.path.join(DATA, 'defaults.json'))) if os.path.exists(os.path.join(DATA, 'defaults.json')) else {}
    q = json.load(open(os.path.join(DATA, 'quads.json'))) if os.path.exists(os.path.join(DATA, 'quads.json')) else {}
    imgs = [{'path': p, 'n': 0, 'default': d.get(os.path.basename(p)), 'quad': q.get(os.path.basename(p))}
            for p in (sorted(glob.glob(os.path.join(src, '*.jpg'))) if src else [])]
    return {'images': imgs, 'settings': {}}


def load_state():
    st = json.load(open(STATE_F)) if os.path.exists(STATE_F) else migrate_legacy()
    for o in st['images']:
        o.setdefault('n', 0)  # v1 state had one crop per photo
    st.setdefault('settings', {})
    return st


STATE = load_state()


def save_state():
    os.makedirs(DATA, exist_ok=True)
    with LOCK:
        tmp = STATE_F + '.tmp'
        json.dump(STATE, open(tmp, 'w')); os.replace(tmp, STATE_F)


def sort_items(): STATE['images'].sort(key=lambda o: (o['path'], o['n']))


def find(i):
    for o in STATE['images']:
        if pid(o) == i: return o
    return None


def find_page(pg):
    for o in STATE['images']:
        if page_id(o['path']) == pg: return o['path']
    return None


def multi(path): return sum(1 for o in STATE['images'] if o['path'] == path) > 1


def label(o):
    stem = os.path.splitext(os.path.basename(o['path']))[0]
    return f"{os.path.basename(os.path.dirname(o['path']))} / {stem}" + (f" {chr(97 + o['n'])}" if multi(o['path']) or o['n'] else '')


def item_json(o):
    return {'id': pid(o), 'page': page_id(o['path']), 'name': label(o), 'path': o['path'], 'n': o['n'],
            'quad': o.get('quad') or o.get('default') or CENTER_QUAD, 'default': o.get('default') or CENTER_QUAD,
            'rot': o.get('rot', 0), 'edited': bool(o.get('quad')), 'auto': bool(o.get('default')),
            'done': bool(o.get('done')), 'flag': o.get('flag'), 'rough': o.get('default_rough'),
            'missing': not os.path.exists(o['path'])}


CACHE = {'Cache-Control': 'max-age=86400'}


def ensure_preview(path):
    p = os.path.join(PREV, page_id(path) + '.jpg')
    if not os.path.exists(p):
        os.makedirs(PREV, exist_ok=True)
        im = cv2.imread(path); h, w = im.shape[:2]; s = 2000 / max(h, w)
        cv2.imwrite(p, cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return p


def crop_img(im, quad, rot=0, size=None, max_side=None):
    """Perspective-crop. size=(w,h) forces a fixed output size (timelapse); default = natural size."""
    h, w = im.shape[:2]
    src = (np.float32(quad) * [w, h]).astype(np.float32)
    if size:
        wo, ho = size
    else:
        tl, tr, br, bl = src
        wo = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
        ho = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if max_side:
        f = min(1.0, max_side / max(wo, ho)); wo, ho = wo * f, ho * f
    wo, ho = max(2, int(round(wo))), max(2, int(round(ho)))
    M = cv2.getPerspectiveTransform(src, np.float32([[0, 0], [wo - 1, 0], [wo - 1, ho - 1], [0, ho - 1]]))
    out = cv2.warpPerspective(im, M, (wo, ho), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    return cv2.rotate(out, ROT[rot]) if rot in ROT else out


def export_json():
    return {'format': 'doccrop', 'version': 2,
            'note': 'one entry per crop region; quad = [TL,TR,BR,BL] as fractions (0-1) of width/height; null = use default',
            'images': [{'path': o['path'], 'n': o['n'], 'default': o.get('default'), 'quad': o.get('quad'), 'rot': o.get('rot', 0)}
                       for o in STATE['images']]}


def import_json(data):
    if data.get('format') not in ('doccrop', 'whiteboard-cropper'): raise ValueError('not a DocCrop export')
    idx = {(o['path'], o['n']): o for o in STATE['images']}
    added = updated = missing = 0
    with LOCK:
        for e in data.get('images', []):
            k = (e['path'], e.get('n', 0)); o = idx.get(k)
            if o is None:
                if not os.path.exists(e['path']): missing += 1; continue
                o = {'path': e['path'], 'n': k[1]}; STATE['images'].append(o); idx[k] = o; added += 1
            else: updated += 1
            if e.get('default'): o['default'] = e['default']
            o['quad'] = e.get('quad'); o['rot'] = e.get('rot', 0)
        sort_items()
    save_state()
    return {'added': added, 'updated': updated, 'skipped_missing_files': missing}


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
    have = {o['path'] for o in STATE['images']}; added = skipped = 0
    with LOCK:
        for p in paths:
            p = os.path.abspath(p)
            if p in have or not p.lower().endswith(IMG_EXT) or not os.path.isfile(p): skipped += 1; continue
            STATE['images'].append({'path': p, 'n': 0}); have.add(p); added += 1
        sort_items()
    save_state()
    return {'added': added, 'skipped': skipped}


def add_region(item_id):
    o = find(item_id)
    with LOCK:
        n = 1 + max(x['n'] for x in STATE['images'] if x['path'] == o['path'])
        new = {'path': o['path'], 'n': n, 'default': CENTER_QUAD, 'rot': 0}
        STATE['images'].append(new); sort_items()
    save_state()
    return pid(new)


def del_region(item_id):
    o = find(item_id)
    with LOCK:
        STATE['images'].remove(o)
        for k, x in enumerate(sorted([x for x in STATE['images'] if x['path'] == o['path']], key=lambda x: x['n'])):
            x['n'] = k  # keep letters contiguous per photo
        sort_items()
    save_state()


def pick_files():
    script = ['set fs to choose file with prompt "Select photos" of type {"public.image"} with multiple selections allowed',
              'set out to ""', 'repeat with f in fs', 'set out to out & POSIX path of f & linefeed', 'end repeat', 'return out']
    r = subprocess.run(['osascript'] + [x for l in script for x in ('-e', l)], capture_output=True, text=True)
    return [l for l in r.stdout.splitlines() if l.strip()] if r.returncode == 0 else []


def pick_dir():
    r = subprocess.run(['osascript', '-e', 'POSIX path of (choose folder with prompt "Select output folder")'], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


# ---------------------------------------------------------------- auto-crop (series aligned to a reference)
def autocrop(scope):
    import align
    ref = STATE.get('reference')
    if not ref: raise RuntimeError('set a reference first (crop one photo, then "Set as reference")')
    targets = [o for o in STATE['images'] if scope == 'all' or not o.get('default')]
    al = align.Aligner(ref['path'], ref['quad'])
    def fn():
        for o in targets:
            if JOB['cancel']: break
            JOB['current'] = label(o)
            try: o['default'], _ = al.quad_for(o['path'])
            except Exception as e: JOB['errors'].append(f"{label(o)}: {e}")
            JOB['done'] += 1; save_state()
    start_job('autocrop', len(targets), fn)


# ---------------------------------------------------------------- refine corners (snap to real edges)
def refine(scope, item_id=None):
    import refine as R
    if scope == 'current':
        targets = [find(item_id)]
    else:  # every region you haven't edited or checked
        targets = [o for o in STATE['images'] if not o.get('quad') and not o.get('done') and o.get('default')]
    by_page = {}
    for o in targets: by_page.setdefault(o['path'], []).append(o)
    def fn():
        for path, os_ in sorted(by_page.items()):
            if JOB['cancel']: break
            im = cv2.imread(path); h, w = im.shape[:2]
            for o in os_:
                JOB['current'] = label(o)
                start = o.get('quad') or o.get('default') or CENTER_QUAD
                q, flag = R.refine_quad(im, np.array(start, float) * [w, h])
                qf = (q / [w, h]).round(5).tolist()
                with LOCK:
                    if scope == 'current':
                        o['quad'] = qf
                    else:
                        o.setdefault('default_rough', o.get('default'))
                        o['default'] = qf
                    if flag: o['flag'] = flag
                    else: o.pop('flag', None)
                JOB['done'] += 1
            save_state()
    start_job('refine', len(targets), fn)


# ---------------------------------------------------------------- run cropper
def run_all(out, only_done=False, size=None, fmt='png'):
    out = os.path.abspath(os.path.expanduser(out))
    os.makedirs(out, exist_ok=True)
    STATE['settings'].update({'out_dir': out, 'out_size': size, 'out_fmt': fmt}); save_state()
    items = [o for o in STATE['images'] if not only_done or o.get('done')]
    by_page = {}
    for o in items: by_page.setdefault(o['path'], []).append(o)
    folders = {os.path.dirname(p) for p in by_page}
    def fn():
        for path, os_ in sorted(by_page.items()):
            if JOB['cancel']: break
            im = cv2.imread(path)
            sub = os.path.join(out, os.path.basename(os.path.dirname(path))) if len(folders) > 1 else out
            os.makedirs(sub, exist_ok=True)
            stem = os.path.splitext(os.path.basename(path))[0]
            for o in os_:
                JOB['current'] = label(o)
                try:
                    c = crop_img(im, o.get('quad') or o.get('default') or CENTER_QUAD, o.get('rot', 0), size)
                    name = stem + (f"_{chr(97 + o['n'])}" if len(os_) > 1 or o['n'] else '')
                    if fmt == 'jpg': cv2.imwrite(os.path.join(sub, name + '.jpg'), c, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    else: cv2.imwrite(os.path.join(sub, name + '.png'), c, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                except Exception as e: JOB['errors'].append(f'{label(o)}: {e}')
                JOB['done'] += 1
        JOB['out_dir'] = out
    start_job('run', len(items), fn)


_page_cache = {}


def page_img(path):
    """tiny cache of decoded full-res photos for fast previews"""
    if path not in _page_cache:
        if len(_page_cache) > 3: _page_cache.pop(next(iter(_page_cache)))
        _page_cache[path] = cv2.imread(path)
    return _page_cache[path]


def parse_size(s):
    if not s: return None
    w, h = str(s).lower().replace(' ', '').split('x')
    return int(w), int(h)


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
        if u.path == '/api/state':
            s = STATE['settings']
            return self.js({'images': [item_json(o) for o in STATE['images']], 'reference': STATE.get('reference'),
                            'settings': {'out_dir': s.get('out_dir') or DEFAULT_OUT, 'out_size': s.get('out_size'), 'out_fmt': s.get('out_fmt') or 'png'}})
        if u.path == '/api/job': return self.js({k: v for k, v in JOB.items() if k != 'started'})
        if u.path == '/api/export':
            return self.send(200, json.dumps(export_json(), indent=1).encode(), 'application/json',
                             {'Content-Disposition': 'attachment; filename="doccrop-export.json"'})
        if u.path.startswith('/img/'):
            p = find_page(u.path[5:]); return self.send(200, open(ensure_preview(p), 'rb').read(), 'image/jpeg', CACHE) if p else self.send(404, b'', 'text/plain')
        if u.path.startswith('/full/'):
            p = find_page(u.path[6:]); return self.send(200, open(p, 'rb').read(), 'image/jpeg', CACHE) if p else self.send(404, b'', 'text/plain')
        if u.path == '/api/preview':
            o = find(q['id'][0])
            c = crop_img(page_img(o['path']), json.loads(q['quad'][0]), int(q.get('rot', ['0'])[0]), parse_size(STATE['settings'].get('out_size')), max_side=900)
            ok, buf = cv2.imencode('.jpg', c, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return self.send(200, buf.tobytes(), 'image/jpeg')
        self.send(404, b'not found', 'text/plain')
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])) or b'{}')
        try:
            if self.path == '/api/save':
                o = find(body['id'])
                if 'quad' in body: o['quad'] = body.get('quad')
                if 'rot' in body: o['rot'] = int(body['rot']) % 360
                if 'done' in body: o['done'] = bool(body['done'])
                if body.get('use_rough') and o.get('default_rough'): o['quad'] = o['default_rough']
                save_state(); return self.js({})
            if self.path == '/api/import': return self.js(import_json(body))
            if self.path == '/api/pick': return self.js(add_paths(pick_files()))
            if self.path == '/api/add': return self.js(add_paths(body['paths']))
            if self.path == '/api/addregion': return self.js({'id': add_region(body['id'])})
            if self.path == '/api/delregion': del_region(body['id']); return self.js({})
            if self.path == '/api/autocrop': autocrop(body.get('scope', 'new')); return self.js({})
            if self.path == '/api/refine': refine(body.get('scope', 'unedited'), body.get('id')); return self.js({})
            if self.path == '/api/cancel': JOB['cancel'] = True; return self.js({})
            if self.path == '/api/reference':
                o = find(body['id']); STATE['reference'] = {'path': o['path'], 'quad': o.get('quad') or o.get('default') or CENTER_QUAD}
                save_state(); return self.js({})
            if self.path == '/api/run':
                run_all(body.get('out_dir') or STATE['settings'].get('out_dir') or DEFAULT_OUT, body.get('only_done', False),
                        parse_size(body.get('size')), body.get('fmt', 'png'))
                return self.js({})
            if self.path == '/api/pickdir': return self.js({'path': pick_dir()})
        except Exception as e:
            return self.js({'error': str(e)}, 400)
        self.send(404, b'', 'text/plain')


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f'http://localhost:{port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
