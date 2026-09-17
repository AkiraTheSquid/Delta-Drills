#!/usr/bin/env python3
"""
Minimal Chrome DevTools Protocol client — enough to load a page at a FIXED
viewport, run JavaScript inside it, and photograph it.

Why not Playwright or Puppeteer: this repo has no package.json and no build
step, and installing Playwright pulls down a second copy of Chromium. Chrome is
already running with --remote-debugging-port=9222 (that is what `chrome_dev`
starts), and everything this tool needs from a browser is four CDP methods.
Dependencies: websocket-client and nothing else that is not already installed.

🔴 EVERY CAPTURE PINS THE VIEWPORT FIRST. A design comparison between two pages
measured at two different window widths is noise, not signal: the reading
column is a percentage, headings clamp, media queries flip, and the ToC rail
this tool exists to compare disappears below 1180px on our side. So the tab
sets device metrics before it navigates, and every number downstream is only
comparable because of that.
"""

import base64
import json
import time
import urllib.parse
import urllib.request

import websocket  # websocket-client

DEFAULT_PORT = 9222
DEFAULT_VIEWPORT = (1440, 900)


class CdpError(RuntimeError):
    pass


def _http(port, path, method="GET"):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method)
    with urllib.request.urlopen(req, timeout=10) as res:
        body = res.read().decode("utf-8", "replace")
    return json.loads(body) if body.strip().startswith(("{", "[")) else body


def browser_is_up(port=DEFAULT_PORT):
    try:
        _http(port, "/json/version")
        return True
    except Exception:
        return False


class Tab:
    """One page target. Not thread-safe and does not need to be."""

    def __init__(self, ws_url, target_id, port):
        # 🔴 NO Origin HEADER. websocket-client derives one from the URL, and
        # Chrome refuses a DevTools socket that arrives with an origin it was
        # not launched to allow ("Handshake status 403 Forbidden"). Suppressing
        # it is what every CDP client does; the alternative is relaunching the
        # user's browser with --remote-allow-origins.
        self._ws = websocket.create_connection(ws_url, timeout=45, suppress_origin=True)
        self._id = 0
        self.target_id = target_id
        self.port = port
        self.send("Page.enable")
        self.send("Runtime.enable")
        # 🔴 NO CACHE. A design capture that answers from a cached stylesheet is
        # measuring the last edit but one, silently — and the whole point of the
        # tool is to re-measure after every change to that stylesheet.
        self.send("Network.enable")
        self.send("Network.setCacheDisabled", cacheDisabled=True)

    # -- protocol ---------------------------------------------------------

    def send(self, method, **params):
        self._id += 1
        want = self._id
        self._ws.send(json.dumps({"id": want, "method": method, "params": params}))
        deadline = time.time() + 45
        while time.time() < deadline:
            msg = json.loads(self._ws.recv())
            if msg.get("id") != want:
                continue  # an event; this client has no use for the stream
            if "error" in msg:
                raise CdpError(f"{method}: {msg['error'].get('message')}")
            return msg.get("result", {})
        raise CdpError(f"{method}: timed out")

    # -- page -------------------------------------------------------------

    def set_viewport(self, width, height, scale=1):
        self.send(
            "Emulation.setDeviceMetricsOverride",
            width=width,
            height=height,
            deviceScaleFactor=scale,
            mobile=False,
        )

    def navigate(self, url, ready=None, settle=1.2, timeout=45):
        """Go there and come back when the page is actually usable.

        The load event is not the signal worth waiting for — our notebook page
        renders its cells from a fetch that starts after load, and LessWrong is
        a React app whose first paint is empty. `ready` is a JS expression that
        the caller knows means 'the thing I am about to measure exists'.
        """
        self.send("Page.navigate", url=url)
        self.wait_for("document.readyState === 'complete'", timeout=timeout)
        if ready:
            self.wait_for(ready, timeout=timeout)
        time.sleep(settle)  # fonts, images, and the last layout pass

    def wait_for(self, expr, timeout=30, poll=0.25):
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            try:
                last = self.evaluate(expr)
                if last:
                    return True
            except CdpError as err:
                last = str(err)
            time.sleep(poll)
        raise CdpError(f"waiting for `{expr}` timed out (last value: {last!r})")

    def evaluate(self, expr, await_promise=False):
        res = self.send(
            "Runtime.evaluate",
            expression=expr,
            returnByValue=True,
            awaitPromise=await_promise,
            allowUnsafeEvalBlockedByCSP=False,
        )
        if res.get("exceptionDetails"):
            detail = res["exceptionDetails"]
            text = detail.get("exception", {}).get("description") or detail.get("text")
            raise CdpError(text)
        return res.get("result", {}).get("value")

    def screenshot(self, path, clip=None, full=False):
        params = {"format": "png"}
        if full and not clip:
            metrics = self.send("Page.getLayoutMetrics")
            size = metrics.get("cssContentSize") or metrics.get("contentSize")
            clip = {"x": 0, "y": 0, "width": size["width"], "height": size["height"]}
        if clip:
            params["clip"] = {**clip, "scale": clip.get("scale", 1)}
            params["captureBeyondViewport"] = True
        data = self.send("Page.captureScreenshot", **params)["data"]
        with open(path, "wb") as handle:
            handle.write(base64.b64decode(data))
        return path

    def close(self):
        try:
            self._ws.close()
        except Exception:
            pass
        try:
            _http(self.port, f"/json/close/{self.target_id}")
        except Exception:
            pass


def open_tab(port=DEFAULT_PORT, url="about:blank"):
    """A NEW tab every time. Reusing whatever the human left open would both
    measure their scroll position and destroy their page."""
    try:
        target = _http(port, f"/json/new?url={urllib.parse.quote(url, safe='')}", "PUT")
    except Exception:
        target = _http(port, f"/json/new?url={urllib.parse.quote(url, safe='')}")
    if not isinstance(target, dict) or "webSocketDebuggerUrl" not in target:
        raise CdpError(f"could not open a tab on port {port}: {target!r}")
    return Tab(target["webSocketDebuggerUrl"], target["id"], port)
