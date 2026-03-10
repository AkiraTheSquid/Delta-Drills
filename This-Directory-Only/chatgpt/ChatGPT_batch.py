import os
import re
import sys
import asyncio
from typing import Optional, List, Tuple, Dict, Any, Union
from datetime import datetime, time as dt_time
import time
import tempfile
import hashlib
import json

from openai import AsyncOpenAI
import tiktoken

try:
    # Newer OpenAI python SDK exception hierarchy
    from openai import (
        RateLimitError,
        APIConnectionError,
        APITimeoutError,
        APIError,
    )
except Exception:  # pragma: no cover
    RateLimitError = Exception  # type: ignore
    APIConnectionError = Exception  # type: ignore
    APITimeoutError = Exception  # type: ignore
    APIError = Exception  # type: ignore


# ============================================================================
# Helper functions from ChatGPT.py
# ============================================================================

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
    """Return model from gpt_model_type.txt, else OPENAI_MODEL env var, else default."""
    file_value = _read_model_type_file(base_dir)
    if file_value:
        return file_value

    env_value = os.environ.get("OPENAI_MODEL")
    if env_value:
        env_value = env_value.strip()
        if env_value:
            return env_value

    return "gpt-4o-mini"


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


def parse_settings(base_dir: str) -> Dict[str, Any]:
    """Parse settings.txt for model/temperature overrides (and optional per-minute token limit)."""
    settings_path = os.path.join(base_dir, "settings.txt")
    model_override: Optional[str] = None
    models_override: Optional[List[str]] = None
    model_cycle: Optional[str] = None
    temperature_override: Optional[Union[int, float]] = None
    limit_override: Optional[int] = None

    if not os.path.isfile(settings_path):
        return {
            "model": model_override,
            "models": models_override,
            "model_cycle": model_cycle,
            "temperature": temperature_override,
            "limit": limit_override,
        }

    try:
        for line in read_text(settings_path).splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, raw_value = stripped.split("=", 1)
            key = key.strip()
            value = raw_value.strip().strip('"').strip("'")

            if key == "model" and value:
                model_override = value
            elif key == "models" and value:
                # Comma-separated list of models, e.g.:
                # models="gpt-4o-mini,gpt-4.1-mini,gpt-4o"
                parsed = [m.strip() for m in value.split(",") if m.strip()]
                models_override = parsed or None
            elif key == "model_cycle" and value:
                # Supported: "round_robin" (cycle models per prompt), "fallback" (try in order on errors)
                model_cycle = value.strip().lower()
            elif key == "temperature" and value:
                try:
                    temperature_override = float(value)
                except ValueError:
                    temperature_override = None
            elif key == "limit" and value:
                # Per-minute token budget. Keep conservative to avoid sitting on the org TPM ceiling.
                try:
                    parsed = int(value)
                    limit_override = max(1, min(parsed, 190_000))
                except ValueError:
                    limit_override = None
    except Exception:
        return {
            "model": model_override,
            "models": models_override,
            "model_cycle": model_cycle,
            "temperature": temperature_override,
            "limit": limit_override,
        }

    return {
        "model": model_override,
        "models": models_override,
        "model_cycle": model_cycle,
        "temperature": temperature_override,
        "limit": limit_override,
    }


def cleanup_output_dir(output_dir: str, keep_prompt_nums: set[int]) -> None:
    """Remove stale output files that are not part of the current batch."""
    if not os.path.isdir(output_dir):
        return

    pattern = re.compile(r"^(\d+)_output\.txt$")
    for filename in os.listdir(output_dir):
        match = pattern.match(filename)
        if not match:
            continue

        try:
            prompt_num = int(match.group(1))
        except ValueError:
            continue

        if prompt_num in keep_prompt_nums:
            continue

        file_path = os.path.join(output_dir, filename)
        try:
            os.remove(file_path)
        except Exception as exc:
            print(f"Warning: Could not delete {filename}: {exc}", file=sys.stderr)


# ============================================================================
# Usage tracking and rate limiting
# ============================================================================

class UsageTracker:
    """Tracks API usage for rate limiting."""
    
    def __init__(self, usage_file: str):
        self.usage_file = usage_file
        self.tokens_within_min = 0
        self.requests_within_min = 0
        self.minute_token_limit_reached = 0
        # Conservative per-minute token budget. Default 190k (below common 200k org TPM cap).
        self.minute_token_limit = 190_000
        self.date = ""
        self.last_reset_time = None
        
    def load(self):
        """Load usage data from usage.md file."""
        if not os.path.exists(self.usage_file):
            # Initialize with defaults
            self._reset_for_new_day()
            return
            
        try:
            # Backward-compatible parsing: accept older files that included within-day fields.
            raw = read_text(self.usage_file).strip()
            lines = raw.splitlines() if raw else []

            kv: Dict[str, str] = {}
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, raw_value = stripped.split("=", 1)
                kv[key.strip()] = raw_value.strip()

            def _int_or_0(value: Optional[str]) -> int:
                try:
                    return int((value or "").strip() or 0)
                except ValueError:
                    return 0

            self.tokens_within_min = _int_or_0(kv.get("tokens_within_min"))
            self.requests_within_min = _int_or_0(kv.get("requests_within_min"))
            self.minute_token_limit_reached = _int_or_0(kv.get("minute_token_limit_reached"))
            self.date = (kv.get("Date") or kv.get("date") or "").strip()

            # Optional persisted last reset epoch for more accurate minute resets
            self.last_reset_time = None
            last_reset_raw = kv.get("LastResetEpoch")
            if last_reset_raw is not None:
                try:
                    self.last_reset_time = float(last_reset_raw.strip())
                except ValueError:
                    self.last_reset_time = None

            # Check if we need to reset for new day (we still reset the file date,
            # but we do NOT enforce any within-day token limits).
            today = datetime.now().strftime("%Y-%m-%d")
            if self.date != today:
                self._reset_for_new_day()
            else:
                # If last_reset_time is missing or stale, make sure minute window resets promptly
                if self.last_reset_time is None:
                    # Force a reset on the next check
                    self.last_reset_time = time.time() - 60
        except Exception as e:
            print(f"Warning: Could not parse usage.md: {e}", file=sys.stderr)
            self._reset_for_new_day()
    
    def _reset_for_new_day(self):
        """Reset counters for a new day."""
        self.tokens_within_min = 0
        self.requests_within_min = 0
        self.minute_token_limit_reached = 0
        self.date = datetime.now().strftime("%Y-%m-%d")
        self.last_reset_time = time.time()
    
    def check_and_reset_minute(self):
        """Check if 60 seconds have passed and reset minute counters if so."""
        if self.last_reset_time is None:
            self.last_reset_time = time.time()
            return
            
        elapsed = time.time() - self.last_reset_time
        if elapsed >= 60:
            self.tokens_within_min = 0
            self.requests_within_min = 0
            self.minute_token_limit_reached = 0
            self.last_reset_time = time.time()
    
    def save(self):
        """Save usage data to usage.md file."""
        content = f"""tokens_within_min = {self.tokens_within_min}
requests_within_min = {self.requests_within_min}
minute_token_limit_reached = {self.minute_token_limit_reached}
Date = {self.date}
LastResetEpoch = {self.last_reset_time if self.last_reset_time is not None else time.time()}
"""
        atomic_write_text(self.usage_file, content)
    
    def add_usage(self, tokens: int, requests: int):
        """Add usage and update limit flags."""
        self.tokens_within_min += tokens
        self.requests_within_min += requests
        
        if self.tokens_within_min > self.minute_token_limit:
            self.minute_token_limit_reached = 1
    
    async def wait_if_needed(self, estimated_tokens: int, num_requests: int):
        """Wait if adding this batch would exceed per-minute rate limits."""
        # Check minute limits
        if (self.tokens_within_min + estimated_tokens > self.minute_token_limit or 
            self.requests_within_min + num_requests > 500):
            
            if self.last_reset_time:
                elapsed = time.time() - self.last_reset_time
                wait_time = max(0, 60 - elapsed)
                if wait_time > 0:
                    print(f"Rate limit approaching. Waiting {wait_time:.1f} seconds...", file=sys.stderr)
                    await asyncio.sleep(wait_time)
                    self.check_and_reset_minute()


# ============================================================================
# Token counting with tiktoken
# ============================================================================

def count_tokens(text: str, model: str) -> int:
    """Count tokens in text using tiktoken for the specified model."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))

def _is_rpd_exhausted(exc: BaseException) -> bool:
    """
    Detect 'requests per day (RPD)' exhaustion from RateLimitError messages.

    When RPD is exhausted, waiting/retrying is pointless until the daily window resets,
    so we should fail fast to avoid multi-hour runs.
    """
    msg = str(exc) or ""
    msg_l = msg.lower()
    return ("requests per day" in msg_l) or ("(rpd)" in msg_l)

def _normalize_model_list(models: List[str]) -> List[str]:
    """De-duplicate models while preserving order."""
    seen = set()
    out: List[str] = []
    for m in models:
        m = (m or "").strip()
        if not m or m in seen:
            continue
        seen.add(m)
        out.append(m)
    return out


# ============================================================================
# Async batch processing
# ============================================================================

async def process_single_prompt(
    client: AsyncOpenAI,
    models_to_try: List[str],
    temperature: float,
    prompt_num: int,
    prompt_text: str,
    output_dir: str
) -> Tuple[int, int, int]:
    """Process a single prompt and return (prompt_num, input_tokens, output_tokens).
    
    Returns (prompt_num, 0, 0) on error.
    """
    output_path = os.path.join(output_dir, f"{prompt_num}_output.txt")
    
    # Retry/backoff makes large runs (hundreds of prompts) much more stable.
    # Hard cap: never allow more than 6 total attempts (max_retries + 1).
    # Even if OPENAI_MAX_RETRIES is set higher in the environment, we clamp it.
    max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "5"))
    max_retries = max(0, min(max_retries, 5))
    base_delay = float(os.environ.get("OPENAI_RETRY_BASE_DELAY_SECONDS", "1.0"))
    max_delay = float(os.environ.get("OPENAI_RETRY_MAX_DELAY_SECONDS", "30.0"))

    last_err: Optional[BaseException] = None
    models_to_try = _normalize_model_list(models_to_try) or ["gpt-4o-mini"]
    model_index = 0
    for attempt in range(max_retries + 1):
        try:
            model = models_to_try[model_index]
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=temperature,
            )

            if response.choices:
                answer = response.choices[0].message.content or ""
            else:
                answer = ""

            output_tokens = count_tokens(answer, model)
            atomic_write_text(output_path, answer)
            return (prompt_num, count_tokens(prompt_text, model), output_tokens)

        except RateLimitError as e:
            # 429: be conservative, but do NOT keep retrying forever.
            last_err = e
            if _is_rpd_exhausted(e):
                # Switch model if possible; only label RPD if all models fail.
                if model_index + 1 < len(models_to_try):
                    prev = models_to_try[model_index]
                    model_index += 1
                    print(
                        f"429 (RPD) on model {prev!r}; switching to {models_to_try[model_index]!r} for this prompt",
                        file=sys.stderr,
                    )
                    continue
                raise RuntimeError("request per day limit reached")
            if attempt >= max_retries:
                break
            # Only one 60s retry for 429s; otherwise we can spend minutes per prompt.
            if attempt >= 1:
                break
            wait_s = 60.0
            print(
                f"Rate limit on prompt {prompt_num} (attempt {attempt+1}/{max_retries+1}): {e}. "
                f"Retrying in {wait_s:.1f}s...",
                file=sys.stderr,
            )
            await asyncio.sleep(wait_s)

        except (APIConnectionError, APITimeoutError, APIError) as e:
            # Transient (non-429): retry with exponential backoff + jitter.
            last_err = e
            if attempt >= max_retries:
                break
            delay = min(max_delay, base_delay * (2 ** attempt))
            # Small deterministic jitter (no random import needed)
            jitter = (prompt_num % 10) * 0.1
            wait_s = delay + jitter
            print(
                f"Transient error on prompt {prompt_num} (attempt {attempt+1}/{max_retries+1}): {e}. "
                f"Retrying in {wait_s:.1f}s...",
                file=sys.stderr,
            )
            await asyncio.sleep(wait_s)

        except Exception as e:
            # Unknown error: don't spin forever; treat as final.
            last_err = e
            break

    error_msg = f"ERROR: {last_err}" if last_err is not None else "ERROR: Unknown error"
    atomic_write_text(output_path, error_msg)
    print(f"Error processing prompt {prompt_num}: {error_msg}", file=sys.stderr)
    return (prompt_num, 0, 0)


async def process_batch(
    prompts: List[Tuple[int, str, int]],
    base_dir: str,
    models: List[str],
    model_cycle: str,
    temperature: float,
    client: AsyncOpenAI,
) -> List[Tuple[int, int, int]]:
    """Process all prompts in parallel.
    
    Returns list of (prompt_num, input_tokens, output_tokens) tuples.
    """
    output_dir = os.path.join(base_dir, "outputs")
    
    # IMPORTANT: launching hundreds of requests fully in-parallel tends to cause
    # connection resets/timeouts and rate-limit spikes. Cap concurrency.
    max_concurrency = int(os.environ.get("OPENAI_MAX_CONCURRENCY", "50"))
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    models = _normalize_model_list(models) or ["gpt-4o-mini"]
    mode = (model_cycle or "fallback").strip().lower()
    if mode not in {"fallback", "round_robin"}:
        mode = "fallback"

    async def _guarded(num: int, text: str, prompt_index: int) -> Tuple[int, int, int]:
        async with semaphore:
            if mode == "round_robin" and len(models) > 1:
                start = prompt_index % len(models)
                models_to_try = models[start:] + models[:start]
            else:
                models_to_try = models
            return await process_single_prompt(client, models_to_try, temperature, num, text, output_dir)

    tasks = [_guarded(num, text, idx) for num, text, idx in prompts]
    return await asyncio.gather(*tasks, return_exceptions=False)


# ============================================================================
# Main batch processing logic
# ============================================================================

async def main_async():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(base_dir, "prompts")
    outputs_dir = os.path.join(base_dir, "outputs")
    usage_file = os.path.join(base_dir, "usage.md")
    completion_state_file = os.path.join(base_dir, "completion_state.txt")

    # NOTE: We must NEVER mark completion_state.txt as "1" when we exit early due to
    # a fatal error (no API key, prompt too long, etc). Several pipeline steps use
    # this file as a readiness signal and would otherwise read partial outputs.
    atomic_write_text(completion_state_file, "0")

    try:
        # Ensure directories exist
        os.makedirs(prompts_dir, exist_ok=True)
        os.makedirs(outputs_dir, exist_ok=True)

        settings = parse_settings(base_dir)
        # Per-minute token budget: keep conservative to avoid crashing on TPM limit.
        minute_token_limit = settings.get("limit") if isinstance(settings, dict) else None
        if not isinstance(minute_token_limit, int) or minute_token_limit <= 0:
            minute_token_limit = 190_000

        # Load API key and model
        api_key = load_api_key(base_dir)
        if not api_key:
            print("ERROR: No API key found in environment, api_key.txt, .openai_key, or .env.", file=sys.stderr)
            atomic_write_text(completion_state_file, "FATAL")
            raise RuntimeError("No API key found.")

        if settings.get("model"):
            model = settings["model"]
            print(f"Using model from settings.txt: {model}", file=sys.stderr)
        else:
            model = get_configured_model(base_dir)
            print(f"Using model: {model}", file=sys.stderr)

        configured_models = settings.get("models")
        if isinstance(configured_models, list) and configured_models:
            models = _normalize_model_list([str(m) for m in configured_models])
        else:
            models = _normalize_model_list([model])

        model_cycle = settings.get("model_cycle") if isinstance(settings, dict) else None
        if not isinstance(model_cycle, str) or not model_cycle.strip():
            model_cycle = "fallback"
        model_cycle = model_cycle.strip().lower()
        if model_cycle not in {"fallback", "round_robin"}:
            print(f"Warning: Unrecognized model_cycle={model_cycle!r}; defaulting to 'fallback'", file=sys.stderr)
            model_cycle = "fallback"

        print(f"Models (from settings.txt): {' -> '.join(models)}", file=sys.stderr)
        print(f"Model selection mode: {model_cycle}", file=sys.stderr)

        temperature = float(settings.get("temperature")) if settings.get("temperature") is not None else 1.0
        print(f"Temperature: {temperature}", file=sys.stderr)

        # Scan for all numbered prompt files matching "<num>_prompt.txt"
        prompt_files = []
        pattern = re.compile(r"^(\d+)_prompt\.txt$")
        for filename in os.listdir(prompts_dir):
            match = pattern.match(filename)
            if not match:
                continue
            prompt_num = int(match.group(1))
            prompt_path = os.path.join(prompts_dir, filename)
            prompt_files.append((prompt_num, prompt_path))

        prompt_files.sort(key=lambda x: x[0])

        prompts_with_tokens = []
        for prompt_index, (prompt_num, prompt_path) in enumerate(prompt_files):
            try:
                raw_prompt = read_text(prompt_path)
            except Exception as e:
                print(f"Warning: Could not read prompt {prompt_num}: {e}", file=sys.stderr)
                continue
            prompt_text = raw_prompt.strip()
            if not prompt_text:
                print(f"Skipping empty prompt file {prompt_num}_prompt.txt.", file=sys.stderr)
                continue

            # Token estimation model: first configured model (good enough for budgeting)
            estimate_model = models[0] if models else model
            prompt_tokens = count_tokens(prompt_text, estimate_model)
            if prompt_tokens > 500_000:
                print(
                    f"ERROR: Prompt {prompt_num} requires {prompt_tokens} tokens which exceeds the 500,000 token batch limit.",
                    file=sys.stderr,
                )
                atomic_write_text(completion_state_file, "FATAL")
                raise RuntimeError(f"Prompt {prompt_num} exceeds token limit.")

            prompts_with_tokens.append((prompt_num, prompt_text, prompt_tokens, prompt_index))

        if not prompts_with_tokens:
            print("No non-empty prompt files found.", file=sys.stderr)
            atomic_write_text(completion_state_file, "1")
            return

        used_prompt_nums = {num for num, _, _ in prompts_with_tokens}
        cleanup_output_dir(outputs_dir, used_prompt_nums)

        print(f"Found {len(prompts_with_tokens)} prompts to process.", file=sys.stderr)

        total_estimated_tokens = sum(tokens for _, _, tokens in prompts_with_tokens)
        print(f"Estimated input tokens across all prompts: {total_estimated_tokens}", file=sys.stderr)

        # Load usage tracker
        tracker = UsageTracker(usage_file)
        tracker.minute_token_limit = int(minute_token_limit)
        tracker.load()
        tracker.check_and_reset_minute()
        print(f"Minute token limit: {tracker.minute_token_limit}", file=sys.stderr)

        client = AsyncOpenAI(api_key=api_key)

        remaining = prompts_with_tokens.copy()
        batch_index = 1
        all_results: List[Tuple[int, int, int]] = []

        while remaining:
            tracker.check_and_reset_minute()

            batch_wall_start = time.time()
            # Split batches by BOTH token budget and request count.
            # The tracker enforces a per-minute request budget; batch sizing should
            # respect it too, otherwise a single batch can exceed the limit.
            max_requests_per_batch = int(os.environ.get("OPENAI_MAX_REQUESTS_PER_BATCH", "500"))
            batch = []
            batch_tokens = 0
            batch_token_budget = min(500_000, tracker.minute_token_limit)
            while (
                remaining
                and len(batch) < max_requests_per_batch
                and batch_tokens + remaining[0][2] <= batch_token_budget
            ):
                batch.append(remaining.pop(0))
                batch_tokens += batch[-1][2]

            if not batch:
                next_num, _, next_tokens = remaining[0]
                print(
                    f"ERROR: Prompt {next_num} requires {next_tokens} tokens which exceeds the {batch_token_budget:,} token batch limit.",
                    file=sys.stderr,
                )
                atomic_write_text(completion_state_file, "FATAL")
                raise RuntimeError(f"Prompt {next_num} exceeds token limit.")

            print(
                f"Processing batch {batch_index}: {len(batch)} prompts, estimated input tokens {batch_tokens}",
                file=sys.stderr,
            )

            await tracker.wait_if_needed(batch_tokens, len(batch))

            start_time = time.time()
            batch_results = await process_batch(
                [(num, text, idx) for num, text, _, idx in batch],
                base_dir,
                models,
                model_cycle,
                temperature,
                client,
            )
            elapsed_time = time.time() - start_time
            print(f"Batch {batch_index} completed in {elapsed_time:.2f} seconds.", file=sys.stderr)

            batch_input_tokens = sum(inp for _, inp, _ in batch_results)
            batch_output_tokens = sum(out for _, _, out in batch_results)
            batch_total_tokens = batch_input_tokens + batch_output_tokens

            all_results.extend(batch_results)

            tracker.add_usage(batch_total_tokens, len(batch))
            tracker.save()

            print(
                f"Batch {batch_index} tokens: {batch_total_tokens} (input: {batch_input_tokens}, output: {batch_output_tokens})",
                file=sys.stderr,
            )
            print(
                f"Usage updated. Minute: {tracker.tokens_within_min} tokens, {tracker.requests_within_min} requests",
                file=sys.stderr,
            )

            if remaining:
                elapsed_since_batch_start = time.time() - batch_wall_start
                sleep_seconds = max(0, 60 - elapsed_since_batch_start)
                if sleep_seconds > 0:
                    print(f"Waiting {sleep_seconds:.1f} seconds before next batch...", file=sys.stderr)
                    await asyncio.sleep(sleep_seconds)
                batch_index += 1

        total_input_tokens = sum(inp for _, inp, _ in all_results)
        total_output_tokens = sum(out for _, _, out in all_results)
        total_tokens = total_input_tokens + total_output_tokens

        print(
            f"All batches complete. Total tokens: {total_tokens} (input: {total_input_tokens}, output: {total_output_tokens})",
            file=sys.stderr,
        )
        atomic_write_text(completion_state_file, "1")
    except BaseException:
        # Leave completion_state as 0/FATAL so callers do not treat this run as successful.
        raise


def main():
    """Entry point for the batch processor."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

