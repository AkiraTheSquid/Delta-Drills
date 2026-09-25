"""watch.py — health checks for conceptual (the Concept Chat tab)

Every way this tab has broken, or would break, is SILENT: a page id that
doesn't match the tab name leaves a blank app; a missing switchTab line never
mounts the chat; a token not defined in all three themes drops the colour; an
SVG without xmlns draws nothing; an @import in auxiliaryStyle is refused.
These checks pin each one by reading the source.
Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.abspath(os.path.join(HERE, '..'))


def _read(*parts):
    with open(os.path.join(WEB, *parts), encoding='utf-8') as fh:
        return fh.read()


def _theme_tokens():
    """Tokens every theme resolves: declared in ALL three `[data-theme]` blocks,
    or in the theme-independent bare `:root {` block (fonts, sizes)."""
    css = re.sub(r'/\*.*?\*/', '', _read('styles', 'variables.css'), flags=re.S)
    themed, shared = [], set()
    for sel, body in re.findall(r'(:root[^{]*)\{([^}]*)\}', css):
        names = set(re.findall(r'(--[a-z0-9-]+)\s*:', body))
        (themed.append(names) if 'data-theme' in sel else shared.update(names))
    assert len(themed) >= 3, 'could not find the three theme blocks in variables.css'
    return set.intersection(*themed) | shared


def check_imports():
    for f in ('conceptual_chat.js', 'conceptual_chat_theme.js'):
        assert os.path.isfile(os.path.join(HERE, f)), f'missing {f}'
    assert os.path.isfile(os.path.join(WEB, 'vendor', 'deep-chat', 'deepChat.bundle.js')), 'missing vendored Deep Chat'


def check_public_api():
    html = _read('index.html')
    assert 'data-tab="concept-chat"' in html, 'no Concept Chat tab button'
    for needle in ('id="page-concept-chat"', 'id="concept-chat-root"', 'id="concept-chat-new"'):
        assert needle in html, f'index.html lost {needle}'
    theme, main, app = (html.find(f'src="{s}') for s in
                        ('conceptual/conceptual_chat_theme.js', 'conceptual/conceptual_chat.js', 'app.js'))
    assert -1 < theme < main, 'conceptual_chat_theme.js must load before conceptual_chat.js'
    assert main > -1 and (app == -1 or main < app), 'conceptual_chat.js must be tagged (before app.js)'
    assert 'styles/conceptual/conceptual-chat.css' in html, 'page stylesheet not linked'
    js = _read('app.js')
    assert re.search(r'tabName === "concept-chat"\)\s*window\.DDConceptualChat\?\.open\(\)', js), \
        'app.js switchTab no longer opens the chat on arrival'
    main_js = _read('conceptual', 'conceptual_chat.js')
    assert 'window.DDConceptualChat = ' in main_js, 'DDConceptualChat not published'
    assert '"/api/conceptual/chat"' in main_js, 'endpoint path drifted from backend app/conceptual/router.py'
    assert 'import(BUNDLE_URL)' in main_js, 'bundle must be import()-ed lazily'


def check_invariants():
    theme_js = _read('conceptual', 'conceptual_chat_theme.js')
    main_js = _read('conceptual', 'conceptual_chat.js')
    # 1. No raw colours: every colour is a var(--token).
    for name, src in (('theme', theme_js), ('main', main_js)):
        assert not re.search(r'#[0-9a-fA-F]{3,8}\b(?![-\w])', re.sub(r'&#\d+;', '', src)), f'{name}: raw hex colour'
        assert not re.search(r'rgba?\(\s*\d', src), f'{name}: raw rgb() colour'
    # 2. Every token used exists in all three themes (or is layout's topbar height).
    known = _theme_tokens() | {'--dd-topbar-h'}
    used = set(re.findall(r'var\((--[a-z0-9-]+)', re.sub(r'/\*.*?\*/', '', theme_js, flags=re.S)))
    missing = sorted(used - known)
    assert not missing, f'tokens not defined in every theme: {missing}'
    # 3. Deep Chat parses svg.content as image/svg+xml: xmlns or no icon.
    for svg in re.findall(r'<svg[^>]*>', theme_js):
        assert 'xmlns' in svg or '${NS}' in svg, f'svg without xmlns: {svg[:60]}'
    # 4. auxiliaryStyle is a constructable stylesheet: @import is refused.
    assert '@import' not in theme_js.split('const AUX', 1)[-1].split('`;', 1)[0], '@import in auxiliaryStyle'
    # 5. The server owns prompt/model/keys; nothing of that on the client.
    for bad in ('"system"', "'system'", 'gpt-', 'sk-', 'OPENAI'):
        assert bad not in main_js, f'client carries {bad!r}: the server owns prompt/model/keys'


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
