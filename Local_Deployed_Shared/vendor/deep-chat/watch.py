"""watch.py — health checks for vendor/deep-chat

Deep Chat 2.5.1 (MIT) as its published self-contained ESM bundle, copied from
the npm tarball's `dist/deepChat.bundle.js` byte for byte. These checks pin the
copy, the one export conceptual/ relies on, and that it is IMPORTED lazily
rather than script-tagged into every page load.
Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, 'deepChat.bundle.js')
# sha256 of deep-chat@2.5.1 dist/deepChat.bundle.js. An upgrade changes this
# on purpose: re-copy from the tarball, update README + this hash together.
SHA256 = '6844eaf354e19d50dcdd8ee78968d6ae32ccd4b9f15a883902d60e93d69caf84'


def check_imports():
    assert os.path.isfile(BUNDLE), 'missing deepChat.bundle.js'
    assert os.path.isfile(os.path.join(HERE, 'LICENSE')), 'missing LICENSE (MIT requires it)'


def check_public_api():
    with open(BUNDLE, 'rb') as fh:
        raw = fh.read()
    assert hashlib.sha256(raw).hexdigest() == SHA256, 'bundle bytes changed (edited or partial copy?)'
    text = raw.decode('utf-8')
    assert 'customElements.define("deep-chat"' in text, 'bundle no longer registers <deep-chat>'
    # The bundle must stay self-contained: a bare `import` would 404 statically.
    assert 'from"' not in text[:2000] and not text.lstrip().startswith('import'), \
        'bundle has static imports; it must be self-contained'


def check_invariants():
    webroot = os.path.abspath(os.path.join(HERE, '..', '..'))
    index = os.path.join(webroot, 'index.html')
    with open(index, encoding='utf-8') as fh:
        html = fh.read()
    # 387 KB: loaded by conceptual/conceptual_chat.js on first open, never on boot.
    assert 'deepChat.bundle.js' not in html, 'index.html script-tags the Deep Chat bundle; import() it lazily'
    for bad in ('unpkg.com/deep-chat', 'cdn.jsdelivr.net/npm/deep-chat'):
        assert bad not in html, f'index.html loads Deep Chat from a CDN ({bad})'


if __name__ == '__main__':
    for fn in (check_imports, check_public_api, check_invariants):
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
