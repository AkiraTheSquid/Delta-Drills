#!/usr/bin/env python3
"""Serve the frontend the way production serves it.

Two behaviours, both there so what you test locally is what ships:

1. `Cache-Control: no-store`, so the browser always fetches fresh JS/CSS
   instead of serving a stale cached copy of the file you just edited.

2. The SPA rewrite from Local_Deployed_Shared/vercel.json. Production sets
   `cleanUrls` and rewrites `/((?!arena-book(?:/|$)).+)` to `/`, which is what
   makes the pathname deep links work: solo-route.js reads `location.pathname`
   and shows one page with no app chrome (`/diagnostic`, `/knowledge-graph`,
   `/notebooks`, `/practice`...). A plain SimpleHTTPRequestHandler answers 404
   for every one of them, so those routes could only ever be tested by
   deploying — the same class of gap as the backend that was not there.

   🔴 `arena-book` IS EXCLUDED, exactly as in vercel.json. It is a real
   directory of real HTML files (the ARENA book pages); rewriting it to `/`
   would serve the app shell in place of every page in it.
"""

import http.server
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 5173


class AppHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def translate_path(self, path):
        """Fall back to index.html for a path that names no file on disk.

        Deliberately narrow: a request that DOES resolve to a file or a
        directory is served as itself, so this cannot mask a genuinely missing
        asset — a mistyped `src=` still 404s here, which is how a broken script
        tag is meant to look. Only a bare pathname with no file behind it is
        treated as a route into the app.
        """
        local = super().translate_path(path)
        if os.path.exists(local):
            return local
        route = path.split("?", 1)[0].split("#", 1)[0].strip("/")
        # No extension = a route, not an asset. `/foo.js` that is missing stays
        # a 404; `/diagnostic` becomes the app.
        #
        # 🔴 A DELIBERATE DIVERGENCE FROM PRODUCTION, and the only one. Vercel
        # rewrites a dotted pathname too, so `/foo.bar/baz` loads the app there
        # and 404s here. Kept because this is a DEV server and the dot is what
        # keeps a mistyped asset honest: without it `src="app.jss"` would serve
        # index.html with a 200 and the failure would surface as a mystery
        # syntax error instead of a missing file. The app has no dotted routes;
        # if it ever gets one, this is the line to change.
        # 🔴 THE BOUNDARY IS A PATH SEGMENT, not a prefix — `arena-book(?:/|$)`
        # in vercel.json. A bare `startswith` also excludes `/arena-bookish`,
        # which production WOULD rewrite, and local/production parity is the
        # only reason this method exists.
        in_book = route == "arena-book" or route.startswith("arena-book/")
        if route and not in_book and "." not in route.rsplit("/", 1)[-1]:
            return super().translate_path("/index.html")
        return local


with http.server.HTTPServer(("", PORT), AppHandler) as httpd:
    print(f"Serving on http://localhost:{PORT} (no-cache, SPA rewrite)")
    httpd.serve_forever()
