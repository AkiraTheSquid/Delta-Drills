# ChatGPT Batch Processing

This script (`ChatGPT_batch.py`) allows you to process multiple OpenAI API requests in parallel with automatic rate limiting.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure your API key is configured (same as for `ChatGPT.py`):
   - Set `OPENAI_API_KEY` environment variable, or
   - Create `api_key.txt` with your key

## Usage

1. Create numbered prompt files in the `prompts/` directory:
   - `prompts/1_prompt.txt`
   - `prompts/2_prompt.txt`
   - ... up to `prompts/100_prompt.txt`

2. Run the batch processor:
   ```bash
   python ChatGPT_batch.py
   ```

3. Results will be written to:
   - `outputs/1_output.txt`
   - `outputs/2_output.txt`
   - etc.

## Features

### Parallel Processing
- All non-empty prompt files (1-100) are processed simultaneously using async/await
- Much faster than sequential processing

### Rate Limiting
The script automatically tracks and enforces OpenAI's rate limits:

- **Per-minute limits:**
  - 500,000 tokens
  - 500 requests
  
- **Per-day limits:**
  - 5,000,000 tokens

- **Per-batch limit:**
  - 300,000 input tokens (safety limit to prevent excessive single batches)

### Automatic Waiting
If a batch would exceed rate limits, the script will:
- Wait until the minute window resets (for minute limits)
- Wait until midnight (for daily limits)

### Usage Tracking
All usage is tracked in `usage.md`:
- Tokens used within the current minute
- Requests made within the current minute
- Tokens used within the current day
- Flags when limits are reached

## Model Configuration

By default, the script uses `gpt-4o-mini`. You can change this by:
- Setting `OPENAI_MODEL` environment variable, or
- Creating `gpt_model_type.txt` with the model name

## Error Handling

- If a prompt fails, the error is written to its output file
- Other prompts continue processing
- Script exits successfully even if some prompts fail

## Examples

### Process 5 prompts in parallel
```bash
# Create prompts
echo "What is 2+2?" > prompts/1_prompt.txt
echo "What is the capital of France?" > prompts/2_prompt.txt
echo "Explain quantum computing in one sentence." > prompts/3_prompt.txt
echo "What is the speed of light?" > prompts/4_prompt.txt
echo "Name three primary colors." > prompts/5_prompt.txt

# Run batch
python ChatGPT_batch.py
```

### Check usage
```bash
cat usage.md
```

