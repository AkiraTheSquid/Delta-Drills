<!-- 8a708d36-c0ce-4c58-a310-bfdc2437e627 9521790d-3489-4ba6-8eb7-deb3ab39d6de -->
# Async Batch OpenAI Processing

## Overview

Create `ChatGPT_batch.py` to process numbered prompt files (1-100) in parallel using Python's `asyncio` and `AsyncOpenAI`, with intelligent rate limiting based on tiktoken token counting.

## Implementation Details

### Core Structure

- New script: `pdf_2_problem/chatgpt/ChatGPT_batch.py`
- Reuse existing helpers from `ChatGPT.py`: `load_api_key()`, `get_configured_model()`
- Input: `prompts/N_prompt.txt` (scan 1-100, process non-empty files)
- Output: `outputs/N_output.txt`
- Usage tracking: `usage.md`

### Rate Limiting Logic

**Before batch execution:**

1. Scan and collect all non-empty prompt files (1-100)
2. Use tiktoken to estimate input tokens for all prompts
3. Check if batch estimate > 300k tokens → refuse and exit with error
4. Load current usage from `usage.md`:

- Check if date changed → reset `tokens_within_day` to 0
- Check if >60 seconds passed → reset `tokens_within_min` and `requests_within_min` to 0

5. If adding batch would exceed minute limits (500k tokens or 500 requests):

- Wait/sleep until 60 seconds have passed since last reset

6. If adding batch would exceed day limits (5M tokens):

- Wait/sleep until next day (midnight)

**After batch execution:**

1. Count actual output tokens using tiktoken
2. Update `usage.md` with total tokens (input + output) and request count
3. Set `minute_token_limit_reached = 1` if tokens_within_min > 500,000
4. Set `day_token_limit_reached = 1` if tokens_within_day > 5,000,000

### Async Implementation

- Use `AsyncOpenAI` client
- Create async task for each prompt file
- Use `asyncio.gather()` to execute all requests in parallel
- Each task reads prompt, calls API, writes output atomically
- Handle errors per-task (write error to output file, continue with others)

### Key Files Referenced

- Leverage `ChatGPT.py` lines 64-75 (model config), 77-110 (API key loading)
- Parse `usage.md` format (6 lines: tokens_within_min, requests_within_min, tokens_within_day, minute_token_limit_reached, day_token_limit_reached, Date)
- Use atomic writes (lines 27-43) for output files

### Dependencies

- Add to `requirements.txt`: `tiktoken`, `openai` (async support)

### To-dos

- [ ] Create ChatGPT_batch.py with basic structure, imports, and helper functions from ChatGPT.py
- [ ] Implement functions to parse and update usage.md with rate limit tracking
- [ ] Add tiktoken-based token counting for prompts and completions
- [ ] Implement pre-batch validation and wait logic for rate limits (minute/day, 300k batch max)
- [ ] Implement async batch processing with AsyncOpenAI and asyncio.gather()
- [ ] Test the batch script with sample prompts and verify rate limiting works correctly