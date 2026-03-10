#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
THIS_DIR_ONLY = REPO_DIR / "This-Directory-Only"
STATUS_PATH = THIS_DIR_ONLY / "chatgpt" / "validator_health.txt"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    validate = load_module("delta_validate_function_bank", THIS_DIR_ONLY / "scripts" / "validate_function_bank.py")
    code_runner = validate.load_code_runner()

    cases = [
        {
            "name": "valid_numpy_fixture_passes",
            "starter_code": "import numpy as np\n\ndef solve():\n    Z = np.random.randint(0,10,(5,5))\n    # Write your solution here\n    return None\n",
            "test_case": {
                "setup_code": "Z = np.random.randint(0,10,(5,5))",
                "call": "solve()",
                "expected_expr": "Z.argmax(axis=1)",
            },
            "expected_reasons": [],
        },
        {
            "name": "undefined_fixture_names_fail",
            "starter_code": "import numpy as np\nimport einops\nfrom einops import repeat\n\ndef solve():\n    b = 2\n    # Write your solution here\n    return None\n",
            "test_case": {
                "setup_code": "b = 2",
                "call": "solve()",
                "expected_expr": "einops.repeat(cls, 'b d -> b t d', t=t)",
            },
            "expected_reasons": ["expected_expr_undefined_names"],
        },
        {
            "name": "valid_repeat_fixture_passes",
            "starter_code": "import numpy as np\nimport einops\nfrom einops import repeat\n\ndef solve():\n    b = 2\n    t = 4\n    cls = np.arange(b * 3).reshape(b, 3)\n    # Write your solution here\n    return None\n",
            "test_case": {
                "setup_code": "b = 2\nt = 4\ncls = np.arange(b * 3).reshape(b, 3)",
                "call": "solve()",
                "expected_expr": "einops.repeat(cls, 'b d -> b t d', t=t)",
            },
            "expected_reasons": [],
        },
        {
            "name": "syntax_error_is_rejected",
            "starter_code": "import numpy as np\n\ndef solve():\n    # Write your solution here\n    return None\n",
            "test_case": {
                "setup_code": "Z = np.arange(4",
                "call": "solve()",
                "expected_expr": "Z",
            },
            "expected_reasons": ["setup_code_syntax_error"],
        },
        {
            "name": "invalid_assignment_expression_is_rejected",
            "starter_code": "import numpy as np\n\nZ = np.zeros(10)\nZ.flags.writeable = False\n\ndef solve():\n    # Write your solution here\n    return None\n",
            "test_case": {
                "setup_code": "Z = np.zeros(10)\nZ.flags.writeable = False",
                "call": "solve()",
                "expected_expr": "Z[0] = 1",
            },
            "expected_reasons": ["expected_expr_syntax_error"],
        },
        {
            "name": "deprecated_numpy_alias_is_rejected",
            "starter_code": "import numpy as np\nfrom io import StringIO\n\ns = StringIO('1,2\\n3,4')\n\ndef solve():\n    # Write your solution here\n    return None\n",
            "test_case": {
                "setup_code": "from io import StringIO\ns = StringIO('1,2\\n3,4')\nZ = np.genfromtxt(s, delimiter=',', dtype=np.int)",
                "call": "solve()",
                "expected_expr": "Z",
            },
            "expected_reasons": ["expected_expr_execution_failed"],
        },
        {
            "name": "missing_external_dependency_is_rejected",
            "starter_code": "import numpy as np\n\ndef solve():\n    # Write your solution here\n    return None\n",
            "test_case": {
                "setup_code": "import scipy\nimport scipy.spatial\nZ = np.random.random((10,2))\nD = scipy.spatial.distance.cdist(Z,Z)",
                "call": "solve()",
                "expected_expr": "D",
            },
            "expected_reasons": ["expected_expr_execution_failed"],
        },
        {
            "name": "undefined_helper_import_is_rejected",
            "starter_code": "import numpy as np\n\nZ = np.arange(10)\n\ndef solve():\n    # Write your solution here\n    return None\n",
            "test_case": {
                "setup_code": "Z = np.arange(10)",
                "call": "solve()",
                "expected_expr": "sliding_window_view(Z, window_shape=3)",
            },
            "expected_reasons": ["expected_expr_undefined_names"],
        },
        {
            "name": "visual_placeholder_define_is_replaced_correctly",
            "starter_code": "import numpy as np\nimport einops\nfrom einops import rearrange, reduce, repeat, einsum\n\narr = np.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4)\n\ndef solve():\n    # Write your solution here - define out\n    return None\n",
            "test_case": {
                "setup_code": "arr = np.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4)",
                "call": "solve()",
                "expected_expr": "einops.rearrange(arr, 'b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2)",
            },
            "expected_reasons": [],
        },
    ]

    failures: list[str] = []
    status_lines: list[str] = []
    for case in cases:
        reasons, _details = validate._validate_test_case(  # noqa: SLF001
            code_runner,
            case["starter_code"],
            case["test_case"],
        )
        passed = sorted(reasons) == sorted(case["expected_reasons"])
        status_lines.append(
            f"{case['name']}: {'0' if passed else '1'} | expected={case['expected_reasons']} | got={reasons}"
        )
        if not passed:
            failures.append(f"{case['name']}: expected {case['expected_reasons']}, got {reasons}")

    if failures:
        STATUS_PATH.write_text("1\n" + "\n".join(status_lines) + "\n", encoding="utf-8")
        print("Validator regression failures:")
        for failure in failures:
            print(f"- {failure}")
        sys.exit(1)

    STATUS_PATH.write_text("0\n" + "\n".join(status_lines) + "\n", encoding="utf-8")
    print(f"Validator regression checks passed: {len(cases)} cases")


if __name__ == "__main__":
    main()
