"""watch.py — health checks for styles/conceptual (Concept Chat page frame)

The page CSS may use only tokens every theme defines and no raw colours; the
selectors it styles must still exist in index.html. The chat's INNER styles
are conceptual/conceptual_chat_theme.js and are checked there.
Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STYLES = os.path.abspath(os.path.join(HERE, '..'))
CSS = os.path.join(HERE, 'conceptual-chat.css')


def _strip(css):
    return re.sub(r'/\*.*?\*/', '', css, flags=re.S)


def check_imports():
    assert os.path.isfile(CSS), 'missing conceptual-chat.css'


def check_public_api():
    with open(os.path.join(STYLES, '..', 'index.html'), encoding='utf-8') as fh:
        html = fh.read()
    for sel in ('page-concept-chat', 'concept-chat-root', 'cc-head', 'cc-new'):
        assert sel in html, f'index.html no longer has {sel}; this stylesheet styles nothing'


def check_invariants():
    with open(CSS, encoding='utf-8') as fh:
        css = _strip(fh.read())
    assert not re.search(r'#[0-9a-fA-F]{3,8}\b', css.replace('#page-concept-chat', '').replace('#concept-chat-root', '')), \
        'raw hex colour in conceptual-chat.css'
    assert not re.search(r'rgba?\(\s*\d', css), 'raw rgb() colour in conceptual-chat.css'
    with open(os.path.join(STYLES, 'variables.css'), encoding='utf-8') as fh:
        tokens = _strip(fh.read())
    themed, shared = [], set()
    for sel, body in re.findall(r'(:root[^{]*)\{([^}]*)\}', tokens):
        names = set(re.findall(r'(--[a-z0-9-]+)\s*:', body))
        (themed.append(names) if 'data-theme' in sel else shared.update(names))
    known = set.intersection(*themed) | shared | {'--dd-topbar-h'}
    missing = sorted(set(re.findall(r'var\((--[a-z0-9-]+)', css)) - known)
    assert not missing, f'tokens not defined in every theme: {missing}'


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
