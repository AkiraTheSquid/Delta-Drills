#!/usr/bin/env python3
"""
Daily Modal spend check for Delta Drills notebook kernels.

Reads Modal's own billing (the workspace's metered cost, before the plan's
free credits — the same number the in-app ceiling in `app/modal_spend.py`
compares against) and raises an alert when either:

  * yesterday (UTC) cost more than DD_MODAL_DAILY_ALERT_USD (default 1.50), or
  * the month so far projects past DD_MODAL_PACE_ALERT_USD (default 30).

An alert is a file every Claude Code session prints at start
(`dd_modal_spend_alert`, a SessionStart hook) plus a Telegram message to
Seth's phone through the Delta Note relay's bot. Only a run that READ the bill
and found it ok clears the alert; a failed read changes nothing and exits 1,
because "could not look" is not "nothing to see".

Run by the `dd-modal-spend-check.timer` systemd user timer under the backend
venv (the only interpreter here with `modal`). Credentials: ~/.modal.toml.

    dd_modal_spend_check.py                         # the real check
    dd_modal_spend_check.py --daily-threshold 0     # force an alert (test)
    dd_modal_spend_check.py --no-push               # file only, no phone
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

STATE_DIR = Path(os.environ.get("DD_STATE_DIR", Path.home() / ".local/state/delta-drills"))
ALERT_FILE = STATE_DIR / "modal-spend-alert.md"
LAST_RUN = STATE_DIR / "modal-spend-last-run"
RELAY_ENV = Path(os.environ.get("DD_ALERT_RELAY_ENV",
                                Path.home() / ".config/delta-note/alert-relay.env"))
# A busy first morning must not project the whole month from a few hours.
MIN_PACE_DAYS = 3.0


def read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                out[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


def read_bill(now: datetime) -> dict:
    import modal
    workspace = modal.Workspace.from_context()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(days=1)
    rows = workspace.billing.report(start=yesterday, end=today, resolution="d")
    month = workspace.billing.summary()
    return {
        "yesterday_usd": float(sum(r.cost for r in rows)),
        "mtd_usd": float(month.metered_cost),
        "billed_usd": float(month.billed_cost),
    }


def judge(bill: dict, now: datetime, daily_limit: float, pace_limit: float) -> tuple[float, list[str]]:
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elapsed_days = max((now - month_start).total_seconds() / 86400, MIN_PACE_DAYS)
    days_in_month = calendar.monthrange(now.year, now.month)[1]
    pace = bill["mtd_usd"] / elapsed_days * days_in_month
    reasons = []
    # >= so a forced 0 limit alerts even on a $0 day (the test path).
    if bill["yesterday_usd"] >= daily_limit:
        reasons.append(f"yesterday cost ${bill['yesterday_usd']:.2f} (limit ${daily_limit:.2f})")
    if pace >= pace_limit:
        reasons.append(f"month projects ${pace:.2f} (limit ${pace_limit:.2f})")
    return pace, reasons


def telegram(text: str) -> str:
    cfg = read_env_file(RELAY_ENV)
    token_file = Path(os.path.expanduser(cfg.get("TG_TOKEN_FILE", "")))
    token = read_env_file(token_file).get("TG_BOT_TOKEN", "") if cfg.get("TG_TOKEN_FILE") else ""
    chats = [c.strip() for c in cfg.get("TG_CHAT_IDS", "").split(",") if c.strip()]
    if not token or not chats:
        return f"no Telegram config in {RELAY_ENV}"
    failures = []
    for chat in chats:
        data = urllib.parse.urlencode({"chat_id": chat, "text": text,
                                       "disable_web_page_preview": "true"}).encode()
        try:
            with urllib.request.urlopen(
                    f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=20) as resp:
                if not json.loads(resp.read()).get("ok"):
                    failures.append(chat)
        except Exception as exc:  # never echo the URL: it carries the token
            failures.append(f"{chat} ({type(exc).__name__})")
    return f"push failed for {', '.join(failures)}" if failures else "pushed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--daily-threshold", type=float,
                        default=float(os.environ.get("DD_MODAL_DAILY_ALERT_USD", "1.50")))
    parser.add_argument("--pace-threshold", type=float,
                        default=float(os.environ.get("DD_MODAL_PACE_ALERT_USD", "30")))
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    try:
        bill = read_bill(now)
    except Exception as exc:
        print(f"modal spend: billing unreadable, alert state unchanged: {exc}", file=sys.stderr)
        return 1
    pace, reasons = judge(bill, now, args.daily_threshold, args.pace_threshold)
    report = (f"=== {now.isoformat(timespec='seconds')} ===\n"
              f"yesterday (UTC): ${bill['yesterday_usd']:.2f}\n"
              f"month to date:   ${bill['mtd_usd']:.2f} metered, ${bill['billed_usd']:.2f} after credits\n"
              f"month pace:      ${pace:.2f}\n")
    print(report, end="")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LAST_RUN.write_text(now.isoformat(timespec="seconds") + "\n")

    was_open = ALERT_FILE.exists()
    if reasons:
        ALERT_FILE.write_text(
            "# \U0001f534 DELTA DRILLS MODAL SPEND ALERT\n\n"
            + "".join(f"- {r}\n" for r in reasons)
            + f"\nNotebook kernels (Modal sandboxes) are spending faster than the "
              f"${args.pace_threshold:.0f}/month\nbudget allows. Check `modal billing report --for \"this month\" -r d` and\n"
              "`/api/practice/kernel/status` (spend block). Clears on the next ok run of\n"
              "`ops/modal-spend/dd_modal_spend_check.py`.\n\n"
            "```\n" + report + "```\n")
        print("ALERT: " + "; ".join(reasons))
        if not args.no_push:
            print(telegram("Delta Drills Modal spend: " + "; ".join(reasons)
                           + f". Month so far ${bill['mtd_usd']:.2f}."))
        return 2
    if was_open:
        ALERT_FILE.unlink(missing_ok=True)
        print("ok — alert cleared")
        if not args.no_push:
            print(telegram(f"Delta Drills Modal spend back under limits "
                           f"(month so far ${bill['mtd_usd']:.2f})."))
    else:
        print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
