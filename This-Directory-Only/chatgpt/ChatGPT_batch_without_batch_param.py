
import os
import re
import sys
import asyncio
from typing import Optional, List, Tuple, Dict, Any, Union
from datetime import datetime
import time
import tempfile
import hashlib
import json

from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APITimeoutError, APIError
import tiktoken

def _is_rpd_exhausted(exc: BaseException) -> bool:
    msg = str(exc) or ""
    msg_l = msg.lower()
    return ("requests per day" in msg_l) or ("(rpd)" in msg_l)

def _model_chain(start_model: str) -> List[str]:
    """
    Per-prompt model fallback chain for this run (does NOT edit settings.txt).

    Order requested:
      gpt-4o-mini -> gpt-4.1-mini -> gpt-4o
    If start_model is something else, we still try it first then the fallbacks.
    """
    fallbacks = ["gpt-4.1-mini", "gpt-4o"]
    seen = set()
    chain: List[str] = []
    for m in [start_model, *fallbacks]:
        if m and m not in seen:
            seen.add(m)
            chain.append(m)
    return chain or [start_model]


# ============================================================================
# Helper functions from ChatGPT_batch.py
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
        # On Windows, antivirus or other processes can transiently lock the file.
        # Retry a few times on PermissionError before failing.
        last_exc: Optional[Exception] = None
        for attempt in range(5):
            try:
                os.replace(tmp_path, path)
                last_exc = None
                break
            except PermissionError as exc:  # pragma: no cover - platform specific
                last_exc = exc
                time.sleep(0.1 * (attempt + 1))
        if last_exc:
            raise last_exc
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


def parse_settings(base_dir: str) -> Dict[str, Any]:
    """Parse settings.txt for model/temperature/logprobs toggles."""
    settings_path = os.path.join(base_dir, "settings.txt")
    logprobs_enabled = False
    top_logprobs = 5
    model_override: Optional[str] = None
    temperature_override: Optional[Union[int, float]] = None
    num_prompts_in_batch: Optional[int] = None
    limit_override: Optional[int] = None

    if not os.path.isfile(settings_path):
        return {
            "logprobs_enabled": logprobs_enabled,
            "top_logprobs": top_logprobs,
            "model": model_override,
            "temperature": temperature_override,
            "num_prompts_in_batch": num_prompts_in_batch,
            "limit": limit_override,
        }

    def _norm_bool(val: str) -> bool:
        val = val.strip().lower()
        return val in {"1", "true", "yes", "on"}

    try:
        for line in read_text(settings_path).splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, raw_value = stripped.split("=", 1)
            key = key.strip()
            value = raw_value.strip().strip('"').strip("'")

            if key == "logprobs":
                logprobs_enabled = _norm_bool(value)
            elif key == "top_logprobs":
                if value:
                    try:
                        top_logprobs = max(1, int(value))
                    except ValueError:
                        top_logprobs = 5
            elif key == "model":
                if value:
                    model_override = value
            elif key == "temperature":
                if value:
                    try:
                        temperature_override = float(value)
                    except ValueError:
                        temperature_override = None
            elif key == "num_prompts_in_batch":
                if value:
                    try:
                        parsed = int(value)
                        if parsed > 0:
                            num_prompts_in_batch = parsed
                    except ValueError:
                        num_prompts_in_batch = None
            elif key == "limit":
                if value:
                    try:
                        parsed_limit = int(value)
                        if parsed_limit > 0:
                            limit_override = parsed_limit
                    except ValueError:
                        limit_override = None
    except Exception:
        # Fall back to defaults if parsing fails
        return {
            "logprobs_enabled": logprobs_enabled,
            "top_logprobs": top_logprobs,
            "model": model_override,
            "temperature": temperature_override,
            "num_prompts_in_batch": num_prompts_in_batch,
            "limit": limit_override,
        }

    return {
        "logprobs_enabled": logprobs_enabled,
        "top_logprobs": top_logprobs,
        "model": model_override,
        "temperature": temperature_override,
        "num_prompts_in_batch": num_prompts_in_batch,
        "limit": limit_override,
    }


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
    
    def __init__(self, usage_file: str, minute_token_limit: int = 190000):
        self.usage_file = usage_file
        self.tokens_within_min = 0
        self.requests_within_min = 0
        self.minute_token_limit_reached = 0
        self.date = ""
        self.last_reset_time = None
        # Allow overriding the per-minute token limit from settings.txt, but keep it conservative.
        # We clamp to 190k to avoid riding the org's 200k TPM ceiling.
        self.minute_token_limit = max(1, min(int(minute_token_limit), 190000))
        
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


# ============================================================================
# Async processing with limited concurrency (no OpenAI batch API)
# ============================================================================

async def process_single_prompt(
    client: AsyncOpenAI,
    model: str,
    prompt_num: int,
    prompt_text: str,
    output_dir: str,
    temperature: float,
    semaphore: asyncio.Semaphore,
    tracker: UsageTracker,
    tracker_lock: asyncio.Lock,
    prompt_tokens: int,
    logprobs_enabled: bool,
    top_logprobs: int,
    logprobs_dir: Optional[str],
) -> Tuple[int, int, int]:
    """Process a single prompt using chat completions with semaphore limiting.
    
    Returns (prompt_num, input_tokens, output_tokens). On error returns (prompt_num, 0, 0).
    """
    output_path = os.path.join(output_dir, f"{prompt_num}_output.txt")
    logprobs_path = None
    if logprobs_enabled and logprobs_dir:
        logprobs_path = os.path.join(logprobs_dir, f"{prompt_num}_logprobs.txt")
    
    async with semaphore:
        # Rate-limit check based on estimated input tokens; serialize tracker updates to avoid races.
        async with tracker_lock:
            tracker.check_and_reset_minute()
            await tracker.wait_if_needed(prompt_tokens, 1)
        
        request_kwargs = {"messages": [{"role": "user", "content": prompt_text}], "temperature": temperature}
        if logprobs_enabled:
            request_kwargs["logprobs"] = True
            request_kwargs["top_logprobs"] = top_logprobs

        # Retry policy:
        # - Hard cap: 6 total attempts (never unbounded).
        # - RateLimitError (429):
        #   - Switch models in order: 4o-mini -> 4.1-mini -> 4o.
        #   - If 429 persists on last model, we fail fast.
        # - Other transient errors: short backoff (seconds), up to the 6-attempt cap.
        models_to_try = _model_chain(model)
        last_err: Optional[BaseException] = None
        response = None
        model_index = 0
        for attempt in range(6):
            try:
                request_kwargs["model"] = models_to_try[model_index]
                response = await client.chat.completions.create(**request_kwargs)
                last_err = None
                break
            except RateLimitError as e:
                last_err = e
                # First response to any 429: switch models if possible, without assuming RPD/TPM/RPM.
                if model_index + 1 < len(models_to_try):
                    model_index += 1
                    print(
                        f"429 on model {models_to_try[model_index-1]!r}; switching to {models_to_try[model_index]!r} for this prompt",
                        file=sys.stderr,
                    )
                    continue
                # Last model also failed: decide what to report.
                if _is_rpd_exhausted(e):
                    raise RuntimeError("request per day limit reached")
                raise RuntimeError("rate limit reached (likely shared per-minute or other bucket)")
            except (APIConnectionError, APITimeoutError, APIError) as e:
                last_err = e
                if attempt >= 5:
                    break
                # Short backoff; don't sleep 60s for non-429 transients.
                wait_s = min(30.0, 1.0 * (attempt + 1))
                print(
                    f"Transient error on prompt {prompt_num} (attempt {attempt+1}/6): {e}. Retrying in {wait_s:.1f}s...",
                    file=sys.stderr,
                )
                await asyncio.sleep(wait_s)
            except Exception as e:
                last_err = e
                break

        if response is None:
            error_msg = f"ERROR: {last_err}" if last_err is not None else "ERROR: Unknown error"
            atomic_write_text(output_path, error_msg)
            if logprobs_path is not None:
                atomic_write_text(logprobs_path, error_msg)
            print(f"Error processing prompt {prompt_num}: {error_msg}", file=sys.stderr)
            return (prompt_num, 0, 0)

        # Success path
        if response.choices:
            answer = response.choices[0].message.content or ""
        else:
            answer = ""

        model_used = str(request_kwargs.get("model") or model)
        output_tokens = count_tokens(answer, model_used)
        atomic_write_text(output_path, answer)
        if logprobs_path is not None:
            try:
                lp = getattr(response.choices[0], "logprobs", None)
                if lp is not None:
                    if hasattr(lp, "model_dump"):
                        lp_payload = lp.model_dump()
                    else:
                        lp_payload = lp
                    atomic_write_text(
                        logprobs_path, json.dumps(lp_payload, ensure_ascii=False, indent=2)
                    )
            except Exception as logprob_exc:
                atomic_write_text(logprobs_path, f"ERROR extracting logprobs: {logprob_exc}")

        # Track usage (input + output) after success
        async with tracker_lock:
            tracker.add_usage(prompt_tokens + output_tokens, 1)
            tracker.save()
            print(
                f"Prompt {prompt_num} tokens: total={prompt_tokens + output_tokens} (input={prompt_tokens}, output={output_tokens})",
                file=sys.stderr,
            )
            print(
                f"Usage now minute: {tracker.tokens_within_min} tokens, {tracker.requests_within_min} requests",
                file=sys.stderr,
            )

        return (prompt_num, prompt_tokens, output_tokens)


async def process_all_prompts_concurrently(
    prompts: List[Tuple[int, str, int]],
    base_dir: str,
    model: str,
    client: AsyncOpenAI,
    concurrency: int = 6,
    logprobs_enabled: bool = False,
    top_logprobs: int = 5,
    temperature: float = 1.0,
    minute_token_limit: int = 190000,
) -> List[Tuple[int, int, int]]:
    """Process all prompts with a fixed concurrency limit."""
    output_dir = os.path.join(base_dir, "outputs")
    logprobs_dir = os.path.join(base_dir, "logprobs") if logprobs_enabled else None
    semaphore = asyncio.Semaphore(concurrency)
    tracker = UsageTracker(
        os.path.join(base_dir, "usage.md"),
        minute_token_limit=minute_token_limit,
    )
    tracker.load()
    tracker.check_and_reset_minute()
    tracker_lock = asyncio.Lock()

    tasks = [
        asyncio.create_task(
            process_single_prompt(
                client,
                model,
                prompt_num,
                prompt_text,
                output_dir,
                temperature,
                semaphore,
                tracker,
                tracker_lock,
                prompt_tokens,
                logprobs_enabled,
                top_logprobs,
                logprobs_dir,
            )
        )
        for prompt_num, prompt_text, prompt_tokens in prompts
    ]

    results = await asyncio.gather(*tasks, return_exceptions=False)
    return results


# ============================================================================
# Main processing logic
# ============================================================================

async def main_async():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(base_dir, "prompts")
    outputs_dir = os.path.join(base_dir, "outputs")
    logprobs_dir = os.path.join(base_dir, "logprobs")
    completion_state_file = os.path.join(base_dir, "completion_state.txt")

    atomic_write_text(completion_state_file, "0")

    try:
        # Ensure directories exist
        os.makedirs(prompts_dir, exist_ok=True)
        os.makedirs(outputs_dir, exist_ok=True)
        settings = parse_settings(base_dir)
        logprobs_enabled = bool(settings.get("logprobs_enabled", False))
        top_logprobs = int(settings.get("top_logprobs", 5) or 5)
        settings_model = settings.get("model")
        settings_temperature = settings.get("temperature")
        settings_num_prompts = settings.get("num_prompts_in_batch")
        settings_limit = settings.get("limit")
        if settings_num_prompts is not None:
            try:
                concurrency = max(1, int(settings_num_prompts))
                print(
                    f"Max concurrent prompts from settings.txt num_prompts_in_batch={settings_num_prompts} -> {concurrency}",
                    file=sys.stderr,
                )
            except (TypeError, ValueError):
                concurrency = 4
                print(
                    f"num_prompts_in_batch value {settings_num_prompts!r} invalid; defaulting to {concurrency}",
                    file=sys.stderr,
                )
        else:
            concurrency = 4
            print(
                f"num_prompts_in_batch not set; defaulting to {concurrency}",
                file=sys.stderr,
            )
        if settings_limit is not None:
            try:
                minute_token_limit = max(1, min(int(settings_limit), 190000))
                print(
                    f"Minute token limit from settings.txt: {minute_token_limit}",
                    file=sys.stderr,
                )
            except (TypeError, ValueError):
                minute_token_limit = 190_000
                print(
                    f"limit value {settings_limit!r} invalid; defaulting to {minute_token_limit}",
                    file=sys.stderr,
                )
        else:
            minute_token_limit = 190_000
            print(
                f"limit not set; defaulting to {minute_token_limit}",
                file=sys.stderr,
            )
        temperature = float(settings_temperature) if settings_temperature is not None else 1.0
        if logprobs_enabled:
            os.makedirs(logprobs_dir, exist_ok=True)

        # Load API key and model
        api_key = load_api_key(base_dir)
        if not api_key:
            print("ERROR: No API key found in environment, api_key.txt, .openai_key, or .env.", file=sys.stderr)
            sys.exit(1)

        if settings_model:
            model = settings_model
            print(f"Using model from settings.txt: {model}", file=sys.stderr)
        else:
            model = get_configured_model(base_dir)
            print(f"Using model: {model}", file=sys.stderr)
        print(f"Temperature: {temperature}", file=sys.stderr)
        print(f"Max concurrent prompts: {concurrency}", file=sys.stderr)

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
        for prompt_num, prompt_path in prompt_files:
            try:
                raw_prompt = read_text(prompt_path)
            except Exception as e:
                print(f"Warning: Could not read prompt {prompt_num}: {e}", file=sys.stderr)
                continue
            prompt_text = raw_prompt.strip()
            if not prompt_text:
                print(f"Skipping empty prompt file {prompt_num}_prompt.txt.", file=sys.stderr)
                continue

            prompt_tokens = count_tokens(prompt_text, model)
            if prompt_tokens > minute_token_limit:
                print(
                    f"ERROR: Prompt {prompt_num} requires {prompt_tokens} tokens which exceeds the {minute_token_limit:,} token limit.",
                    file=sys.stderr,
                )
                sys.exit(1)

            prompts_with_tokens.append((prompt_num, prompt_text, prompt_tokens))

        if not prompts_with_tokens:
            print("No non-empty prompt files found.", file=sys.stderr)
            sys.exit(0)

        used_prompt_nums = {num for num, _, _ in prompts_with_tokens}
        cleanup_output_dir(outputs_dir, used_prompt_nums)

        print(f"Found {len(prompts_with_tokens)} prompts to process.", file=sys.stderr)
        total_estimated_tokens = sum(tokens for _, _, tokens in prompts_with_tokens)
        print(f"Estimated input tokens across all prompts: {total_estimated_tokens}", file=sys.stderr)
        print(
            f"Processing with up to {concurrency} concurrent chat completion calls (no batch API).",
            file=sys.stderr,
        )
        if logprobs_enabled:
            print(f"logprobs enabled with top_logprobs={top_logprobs}; outputs in {logprobs_dir}", file=sys.stderr)
        else:
            print("logprobs disabled (toggle via settings.txt).", file=sys.stderr)

        client = AsyncOpenAI(api_key=api_key)
        print(f"Model fallback chain (this run only): {' -> '.join(_model_chain(model))}", file=sys.stderr)

        start_time = time.time()
        results = await process_all_prompts_concurrently(
            prompts_with_tokens,
            base_dir,
            model,
            client,
            concurrency=concurrency,
            logprobs_enabled=logprobs_enabled,
            top_logprobs=top_logprobs,
            temperature=temperature,
            minute_token_limit=minute_token_limit,
        )
        elapsed = time.time() - start_time
        print(f"All prompts completed in {elapsed:.2f} seconds (wall clock).", file=sys.stderr)

        total_input_tokens = sum(inp for _, inp, _ in results)
        total_output_tokens = sum(out for _, _, out in results)
        total_tokens = total_input_tokens + total_output_tokens

        print(
            f"Totals: {total_tokens} tokens (input: {total_input_tokens}, output: {total_output_tokens})",
            file=sys.stderr,
        )
    finally:
        atomic_write_text(completion_state_file, "1")


def main():
    """Entry point for the concurrent non-batch processor."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()



