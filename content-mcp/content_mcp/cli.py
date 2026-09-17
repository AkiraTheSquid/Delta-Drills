"""`dd-content` — the terminal face of the same tools the MCP exposes.

Subcommands and flags are GENERATED from `ops.REGISTRY`, so the CLI can never
offer a tool the MCP lacks or take an argument the MCP does not. Claude Code
can drive either one; a human or a shell script wants this one.

    dd-content status
    dd-content login --password ...
    dd-content lesson_read --kc numpy.random-seeding
    dd-content drill_search --kc einops.merge-axes --limit 5
    dd-content pipeline_check

Long text arguments (`markdown`, `answer_code`, a JSON `override`) are awkward
on a command line, so every op also accepts `--json-args '<object>'` and any
value of `-` is read from stdin.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import ops

ALIASES = {
    "login": "content_login",
    "logout": "content_logout",
    "status": "content_status",
    "check": "pipeline_check",
    "backup": "backup_status",
    "restore": "backup_restore",
}


def _add_arguments(parser: argparse.ArgumentParser, spec) -> None:
    for name, schema in spec.schema()["properties"].items():
        kind = schema.get("type")
        help_text = schema.get("description", "")
        flag = f"--{name.replace('_', '-')}"
        # No flag is argparse-required, even a required parameter: --json-args
        # has to be able to supply everything on its own, and a long
        # `markdown` or `answer_code` is exactly the value nobody wants to
        # type as a flag. Required-ness is enforced once, in `ops.call`, after
        # both sources are merged.
        if name in spec.required:
            help_text = (help_text + " [required]").strip()
        if kind == "boolean":
            parser.add_argument(flag, dest=name, action="store_true", default=None, help=help_text)
        elif kind == "array":
            parser.add_argument(flag, dest=name, nargs="*", help=help_text)
        elif kind == "integer":
            parser.add_argument(flag, dest=name, type=int, help=help_text)
        elif kind == "object":
            parser.add_argument(flag, dest=name, help=(help_text + " (JSON object)").strip())
        else:
            parser.add_argument(flag, dest=name, help=help_text)


def _coerce(spec, namespace: argparse.Namespace) -> dict:
    properties = spec.schema()["properties"]
    arguments: dict = {}
    for name, schema in properties.items():
        value = getattr(namespace, name, None)
        if value is None:
            continue
        if value == "-":
            value = sys.stdin.read()
        if schema.get("type") == "object" and isinstance(value, str):
            value = json.loads(value)
        if schema.get("type") == "array" and schema.get("items", {}).get("type") == "integer":
            value = [int(v) for v in value]
        arguments[name] = value
    return arguments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dd-content",
        description="Edit Delta Drills course content — lessons, concept graph, drill bank.",
    )
    parser.add_argument("--raw", action="store_true", help="Print compact JSON instead of indented.")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    subparsers.add_parser("tools", help="List every available tool and what it needs.")
    subparsers.add_parser("serve", help="Run the MCP server on stdio (what Claude Code launches).")
    setpw = subparsers.add_parser("set-password", help="Set or change the shared editing password.")
    setpw.add_argument("--password", required=True, help="The NEW password.")
    setpw.add_argument("--current", help="The current password. Required once one is set.")

    for spec in sorted(ops.REGISTRY.values(), key=lambda s: s.name):
        sub = subparsers.add_parser(spec.name, help=spec.summary)
        sub.add_argument("--json-args", dest="json_args",
                         help="All arguments as one JSON object; merged under explicit flags.")
        _add_arguments(sub, spec)

    for alias, target in ALIASES.items():
        sub = subparsers.add_parser(alias, help=f"Alias for {target}.")
        sub.add_argument("--json-args", dest="json_args")
        _add_arguments(sub, ops.REGISTRY[target])

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    namespace = parser.parse_args(argv)
    command = namespace.command

    if not command:
        parser.print_help()
        return 1

    if command == "serve":
        from .server import main as serve_main

        return serve_main()

    if command == "tools":
        print(ops.as_json(ops.catalogue()))
        return 0

    if command == "set-password":
        from . import auth

        try:
            target = auth.write_password(namespace.password, namespace.current)
        except auth.AuthError as err:
            print(f"AuthError: {err}", file=sys.stderr)
            return 1
        print(f"Password digest written to {target}")
        return 0

    name = ALIASES.get(command, command)
    spec = ops.REGISTRY[name]
    arguments = json.loads(namespace.json_args) if getattr(namespace, "json_args", None) else {}
    arguments.update(_coerce(spec, namespace))

    try:
        payload = ops.call(name, arguments)
    except Exception as err:
        print(f"{type(err).__name__}: {err}", file=sys.stderr)
        return 1

    print(json.dumps(payload, default=str) if namespace.raw else ops.as_json(payload))
    # A pipeline run that reports ok=False should fail the shell, not just print.
    if isinstance(payload, dict) and payload.get("ok") is False:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
