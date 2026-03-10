import os
import sys
import getpass
from typing import Optional

from openai import OpenAI
import json
import tempfile
import hashlib
import time
import uuid


VERBOSITY = "high"  # Options: "low", "medium", "high"


def read_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(file_path: str, text: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)


def atomic_write_text(path: str, text: str) -> None:
    """Write text to a temporary file in the same directory, then atomically replace target.

    This prevents readers from observing partially written files and works on Windows & POSIX.
    """
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _read_model_type_file(base_dir: str) -> Optional[str]:
    """Read desired model name from gpt_model_type.txt if present and non-empty."""
    try:
        path = os.path.join(base_dir, "gpt_model_type.txt")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                value = f.read().strip()
                if value:
                    return value
    except Exception:
        pass
    return None


def get_configured_model(base_dir: str) -> str:
    """Return model from OPENAI_MODEL env var, else gpt_model_type.txt, else default."""
    env_value = os.environ.get("OPENAI_MODEL")
    if env_value:
        env_value = env_value.strip()
        if env_value:
            return env_value
    file_value = _read_model_type_file(base_dir)
    if file_value:
        return file_value
    return "gpt-5-pro"


def load_api_key(base_dir: str) -> Optional[str]:
    # 1) Environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key.strip()

    # 2) Local files in priority order
    candidates = ("api_key.txt", ".openai_key", ".env")
    for name in candidates:
        path = os.path.join(base_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            if name == ".env":
                # Parse minimal .env looking for OPENAI_API_KEY=...
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.split("=")[0].strip() == "OPENAI_API_KEY":
                            value = "=".join(line.split("=")[1:]).strip().strip('"').strip("'")
                            if value:
                                return value
            else:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        return content
        except Exception:
            # Ignore and try next source
            continue

    return None


def prompt_and_store_api_key(base_dir: str) -> Optional[str]:
    try:
        entered = getpass.getpass("Enter your OPENAI_API_KEY: ").strip()
    except Exception:
        # Fallback to visible input if getpass is unavailable
        entered = input("Enter your OPENAI_API_KEY: ").strip()

    if not entered:
        return None

    # Save for future runs
    try:
        with open(os.path.join(base_dir, "api_key.txt"), "w", encoding="utf-8") as f:
            f.write(entered)
    except Exception:
        pass

    return entered


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(base_dir, "prompt.txt")
    output_path = os.path.join(base_dir, "output.txt")
    meta_path = os.path.join(base_dir, "output.meta.json")
    lock_path = os.path.join(base_dir, "output.lock")
    completion_state_path = os.path.join(base_dir, "completion_state.txt")
    simple_mode = os.environ.get("AI_PROMPT_SIMPLE_MODE", "0").strip() in {"1", "true", "True"}

    # Create lock as early as possible so watchers know a run is in progress (unless simple mode)
    run_id = str(uuid.uuid4())
    if not simple_mode:
        try:
            open(lock_path, "w", encoding="utf-8").close()
        except Exception:
            # If we cannot create a lock, continue but still attempt to write outputs atomically
            pass

    # Mark run as in-progress: completion_state = 0
    try:
        if simple_mode:
            write_text(completion_state_path, "0")
        else:
            atomic_write_text(completion_state_path, "0")
    except Exception:
        pass

    # Defaults in case of early failure
    answer = ""
    completion_id = None
    finish_reason = None
    server_created = None

    # Read prompt (convert early failures to error outputs rather than exiting)
    try:
        prompt = read_text(prompt_path).strip()
    except FileNotFoundError as e:
        print("prompt.txt not found in the current folder.", file=sys.stderr)
        finish_reason = "error"
        answer = f"ERROR: {e}"
        # Write and exit
        if simple_mode:
            write_text(output_path, answer)
            # Mark run as complete: completion_state = 1
            try:
                write_text(completion_state_path, "1")
            except Exception:
                pass
        else:
            atomic_write_text(output_path, answer)
            try:
                atomic_write_text(meta_path, json.dumps({
                    "run_id": run_id,
                    "model": get_configured_model(base_dir),
                    "created_at": int(time.time()),
                    "server_created": server_created,
                    "completion_id": completion_id,
                    "finish_reason": finish_reason,
                    "output_sha256": sha256_str(answer),
                }, ensure_ascii=False))
            except Exception:
                pass
            # Mark run as complete: completion_state = 1
            try:
                atomic_write_text(completion_state_path, "1")
            except Exception:
                pass
            try:
                if os.path.exists(lock_path):
                    os.remove(lock_path)
            except Exception:
                pass
        return

    if not prompt:
        print("prompt.txt is empty.", file=sys.stderr)
        finish_reason = "error"
        answer = "ERROR: prompt.txt is empty."
        if simple_mode:
            write_text(output_path, answer)
            # Mark run as complete: completion_state = 1
            try:
                write_text(completion_state_path, "1")
            except Exception:
                pass
        else:
            atomic_write_text(output_path, answer)
            try:
                atomic_write_text(meta_path, json.dumps({
                    "run_id": run_id,
                    "model": get_configured_model(base_dir),
                    "created_at": int(time.time()),
                    "server_created": server_created,
                    "completion_id": completion_id,
                    "finish_reason": finish_reason,
                    "output_sha256": sha256_str(answer),
                }, ensure_ascii=False))
            except Exception:
                pass
            # Mark run as complete: completion_state = 1
            try:
                atomic_write_text(completion_state_path, "1")
            except Exception:
                pass
            try:
                if os.path.exists(lock_path):
                    os.remove(lock_path)
            except Exception:
                pass
        return

    model = get_configured_model(base_dir)
    api_key = load_api_key(base_dir)
    if not api_key:
        print(
            "No API key found in environment, api_key.txt, .openai_key, or .env.",
            file=sys.stderr,
        )
        print("You will be prompted to enter your key once; it will be saved to api_key.txt.", file=sys.stderr)
        api_key = prompt_and_store_api_key(base_dir)
        if not api_key:
            print("No API key provided.", file=sys.stderr)
            finish_reason = "error"
            answer = "ERROR: No API key provided."
            atomic_write_text(output_path, answer)
            try:
                atomic_write_text(meta_path, json.dumps({
                    "run_id": run_id,
                    "model": model,
                    "created_at": int(time.time()),
                    "server_created": server_created,
                    "completion_id": completion_id,
                    "finish_reason": finish_reason,
                    "output_sha256": sha256_str(answer),
                }, ensure_ascii=False))
            except Exception:
                pass
            # Mark run as complete: completion_state = 1
            try:
                atomic_write_text(completion_state_path, "1")
            except Exception:
                pass
            try:
                if os.path.exists(lock_path):
                    os.remove(lock_path)
            except Exception:
                pass
            return

    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        finish_reason = "error"
        answer = f"ERROR: {e}"
        if simple_mode:
            write_text(output_path, answer)
            # Mark run as complete: completion_state = 1
            try:
                write_text(completion_state_path, "1")
            except Exception:
                pass
        else:
            atomic_write_text(output_path, answer)
            try:
                atomic_write_text(meta_path, json.dumps({
                    "run_id": run_id,
                    "model": model,
                    "created_at": int(time.time()),
                    "server_created": server_created,
                    "completion_id": completion_id,
                    "finish_reason": finish_reason,
                    "output_sha256": sha256_str(answer),
                }, ensure_ascii=False))
            except Exception:
                pass
            # Mark run as complete: completion_state = 1
            try:
                atomic_write_text(completion_state_path, "1")
            except Exception:
                pass
            try:
                if os.path.exists(lock_path):
                    os.remove(lock_path)
            except Exception:
                pass
        return
    try:
        # Try Responses API first (some models only support v1/responses)
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
                temperature=1,
            )
            completion_id = getattr(resp, "id", None)
            server_created = getattr(resp, "created", None)
            # Prefer the SDK's convenience property when available
            answer = getattr(resp, "output_text", None) or ""
            finish_reason = None
            if not answer:
                # Fallback attempt to extract text from structured output
                try:
                    first_output = getattr(resp, "output", None)
                    if isinstance(first_output, list) and first_output:
                        first_content = getattr(first_output[0], "content", None)
                        if isinstance(first_content, list) and first_content:
                            maybe_text = getattr(first_content[0], "text", None)
                            if isinstance(maybe_text, str):
                                answer = maybe_text
                except Exception:
                    pass
        except Exception:
            # Fallback to Chat Completions for models/endpoints that require it
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1,
            )
            completion_id = getattr(completion, "id", None)
            server_created = getattr(completion, "created", None)
            if completion.choices:
                finish_reason = completion.choices[0].finish_reason
                answer = completion.choices[0].message.content or ""
            else:
                finish_reason = None
                answer = ""
    except Exception as e:
        finish_reason = "error"
        answer = f"ERROR: {e}"

    # Write outputs according to mode
    if simple_mode:
        write_text(output_path, answer)
        # Mark run as complete: completion_state = 1
        try:
            write_text(completion_state_path, "1")
        except Exception:
            pass
    else:
        atomic_write_text(output_path, answer)
        meta = {
            "run_id": run_id,
            "model": model,
            "created_at": int(time.time()),
            "server_created": server_created,
            "completion_id": completion_id,
            "finish_reason": finish_reason,
            "output_sha256": sha256_str(answer),
        }
        try:
            atomic_write_text(meta_path, json.dumps(meta, ensure_ascii=False))
        except Exception:
            # Avoid failing the run if metadata can't be written
            pass
        # Mark run as complete: completion_state = 1
        try:
            atomic_write_text(completion_state_path, "1")
        except Exception:
            pass
        # Remove lock last to signal completion
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()


