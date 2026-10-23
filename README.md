# Vision AI reCAPTCHA Solver

<div align="left">

<a href="LICENSE"></a> <img src="https://github.com/DannyLuna17/BulletDroid2/raw/image-data/license-badge.svg" alt="License: MIT" height="22" />
<a href="https://pypi.org/project/vision-ai-recaptcha-solver/"><img src="https://img.shields.io/pypi/v/vision-ai-recaptcha-solver" alt="PyPI version" /></a>
<a href="https://github.com/DannyLuna17/VisionAIRecaptchaSolver/actions/workflows/ci.yml">
<img src="https://github.com/DannyLuna17/VisionAIRecaptchaSolver/actions/workflows/ci.yml/badge.svg" alt="CI" /> [![CodeFactor](https://www.codefactor.io/repository/github/dannyluna17/VisionAIRecaptchaSolver/badge)](https://www.codefactor.io/repository/github/dannyluna17/VisionAIRecaptchaSolver) </a>
<a href="https://huggingface.co/DannyLuna/recaptcha-classification-57k"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue" alt="Model on Hugging Face" height="22" /></a>
<a href="https://huggingface.co/datasets/DannyLuna/recaptcha-57k-images-dataset"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue" alt="Dataset on Hugging Face" height="22" /></a>
<img src="https://upload.wikimedia.org/wikipedia/commons/2/21/Flag_of_Colombia.svg" alt="Colombia Flag" height="22" />

</div>

AI powered reCAPTCHA solver for reCAPTCHA v2 and v3 using YOLO based vision models (57k images dataset) and optional Vision LLM (OpenAI-compatible APIs). The library provides both synchronous and asynchronous APIs, to automate image challenge solving and token extraction.

https://github.com/user-attachments/assets/3d0de5bd-6bbd-4a1a-bb06-fd7efa9d55fd

## Key features

- Supports reCAPTCHA v2 image challenges (3x3 selection, 3x3 dynamic, 4x4 square) and v3 invisible flows
- **Vision LLM support** - Use OpenAI-compatible vision models (e.g., llama-3.2-90b-vision-instruct) for cloud-based image analysis
- Synchronous and asynchronous solver APIs with context manager support
- Configurable proxy, browser path, timeouts, and detection thresholds
- CLI for quick testing and automation
- Works in headless mode
- Automatic model warmup and background initialization
- Automatic model downloads on first run (classification + detection)
- **MCP Server** - Expose solver functionality to AI agents via Model Context Protocol

## How it works

1. `recaptcha-domain-replicator` spins up a local HTTPS server and replicates the target reCAPTCHA widget.
2. A Chromium browser session is launched and the checkbox/challenge is interacted with automatically.
3. Challenge images are downloaded and analyzed:
   - **YOLO models** (local): Classification model for 3x3 challenges, Detection model for 4x4 challenges
   - **Vision LLM** (optional, cloud): Full grid image sent to vision model with keyword prompt, returns cell numbers + confidences
4. The solver clicks the predicted tiles, submits verification, and extracts the token.

### RecaptchaDomainReplicator usage

This project uses `RecaptchaDomainReplicator` (from [`recaptcha-domain-replicator`](https://github.com/DannyLuna17/RecaptchaDomainReplicator)) to build the local replica page and launch the browser session. It creates a new replicator, starts a local HTTPS server, and returns the browser instance plus a token handle.

## Requirements

- Python 3.10 or newer
- Browser: Chromium based browser (Chrome for Testing, Edge)  
- For credential proxies: Use [Chrome for Testing](https://googlechromelabs.github.io/chrome-for-testing/#stable) or Microsoft Edge (Newer versions of Google Chrome doesn't support this)
- Admin privileges (optional): Required only for hosts file changes and port forwarding

## Installation

```bash
pip install vision-ai-recaptcha-solver
```

From source (recommended for development):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

### Optional: Vision LLM support

To use Vision LLM instead of (or alongside) YOLO:

```bash
pip install vision-ai-recaptcha-solver[vision]
```

Or with MCP server support:

```bash
pip install vision-ai-recaptcha-solver[mcp]
```

## Quick start (synchronous)

```python
from vision_ai_recaptcha_solver import RecaptchaSolver, SolverConfig

config = SolverConfig(
    timeout=200,
    # Optional: point to a specific Chromium binary
    # browser_path="path/to/chrome.exe",
    # proxy="http://username:password@ip:port",
)

with RecaptchaSolver(config) as solver:
    result = solver.solve(
        website_key="your_site_key",
        website_url="https://example.com/your-page-with-recaptcha",
    )

print(result.token)
print(result.captcha_type.value)
print(result.time_taken)
print(result.attempts)
```

### Using Vision LLM (cloud-based)

```python
from vision_ai_recaptcha_solver import RecaptchaSolver, SolverConfig

config = SolverConfig(
    # Vision LLM configuration
    vision_model="llama-3.2-90b-vision-instruct",
    vision_api_base="https://llm.datasolved.org/v1",
    vision_api_key="your-api-key",
    vision_timeout=30.0,
    
    # Optional browser settings
    headless=True,
    timeout=180,
)

with RecaptchaSolver(config) as solver:
    result = solver.solve(
        website_key="6Le-wvkSAAAAAPBMRTvw0Q4Muexq9bi0DJwx_mJ-",
        website_url="https://www.google.com/recaptcha/api2/demo"
    )
    print(result.token)
```

### Invisible reCAPTCHA (v3)

```python
from vision_ai_recaptcha_solver import RecaptchaSolver, SolverConfig

config = SolverConfig(timeout=120)

with RecaptchaSolver(config) as solver:
    result = solver.solve(
        website_key="your_site_key",
        website_url="https://example.com/login",
        is_invisible=True,
        action="login",
        is_enterprise=False,
        api_domain="google.com",
        cookies=None,
        user_agent=None,
    )

print(result.token)
```

## Async usage

```python
import asyncio
from pathlib import Path

from vision_ai_recaptcha_solver import AsyncRecaptchaSolver, SolverConfig

async def main():
    config = SolverConfig(
        timeout=180,
        server_port=8443,
        download_dir=Path("tmp1"),
    )

    async with AsyncRecaptchaSolver(config) as solver:
        result = await solver.solve(
            website_key="your_site_key",
            website_url="https://example.com/your-page-with-recaptcha",
        )
        print(result.token)

asyncio.run(main())
```

### Concurrent async solves

Each solver instance should use a distinct `server_port` and `download_dir` to avoid conflicts.

```python
import asyncio
from pathlib import Path

from vision_ai_recaptcha_solver import AsyncRecaptchaSolver, SolverConfig

async def solve_one(name: str, config: SolverConfig):
    async with AsyncRecaptchaSolver(config) as solver:
        result = await solver.solve(
            website_key="your_site_key",
            website_url="https://example.com/your-page-with-recaptcha",
        )
        return name, result

async def main():
    config_a = SolverConfig(server_port=8443, download_dir=Path("tmp-a"))
    config_b = SolverConfig(server_port=8444, download_dir=Path("tmp-b"))

    results = await asyncio.gather(
        solve_one("A", config_a),
        solve_one("B", config_b),
    )

    for name, result in results:
        print(name, result.captcha_type.value, result.time_taken)

asyncio.run(main())
```

## CLI

The package exposes a console script:

```bash
vision-ai-recaptcha-solver --help
```

Solve a captcha:

```bash
vision-ai-recaptcha-solver solve \
  --website-key "your_site_key" \
  --website-url "https://example.com/your-page-with-recaptcha" \
  --no-headless \
  --timeout 180 \
  --output json
```

Solve invisible reCAPTCHA (v3):

```bash
vision-ai-recaptcha-solver solve \
  --website-key "your_site_key" \
  --website-url "https://example.com/login" \
  --invisible \
  --action "login"
```

Run a demo against Google's test page:

```bash
vision-ai-recaptcha-solver demo
```

### With Vision LLM via CLI

Set environment variables or use config file:

```bash
export VISION_MODEL="llama-3.2-90b-vision-instruct"
export VISION_API_BASE="https://llm.datasolved.org/v1"
export VISION_API_KEY="your-api-key"

vision-ai-recaptcha-solver solve \
  --website-key "your_site_key" \
  --website-url "https://example.com/page"
```

## MCP Server (for AI agents)

The solver can be exposed as an MCP (Model Context Protocol) server for AI agents:

```bash
# Start MCP server
python -m vision_ai_recaptcha_solver.mcp_server
```

Or configure in your MCP client (e.g., `.opencode/mcp.json`):

```json
{
  "mcpServers": {
    "vision-ai-recaptcha-solver": {
      "command": "python",
      "args": ["-m", "vision_ai_recaptcha_solver.mcp_server"],
      "cwd": "/path/to/VisionAIRecaptchaSolver",
      "env": {
        "PYTHONPATH": "/path/to/VisionAIRecaptchaSolver/src"
      }
    }
  }
}
```

Available MCP tools:
- `solve_recaptcha` - Sync solve with full config
- `solve_recaptcha_async` - Async solve
- `demo_recaptcha` - Run demo on Google's test page
- `get_solver_info` - Get capabilities and configuration

## Configuration reference

`SolverConfig` controls runtime behavior and validation.

### YOLO / Core options

| Option | Default | Description |
| --- | --- | --- |
| `model_path` | `None` | Path to the classification ONNX model. If `None`, auto downloads from Hugging Face. |
| `detection_model_path` | `None` | Path to detection model. If `None`, uses `yolo12x.pt`. |
| `download_dir` | `Path("tmp")` | Directory for downloaded images and temp files. |
| `server_port` | `8443` | Local HTTPS port used by the replicator. |
| `proxy` | `None` | Proxy URL in `protocol://[user:pass@]host:port` format. |
| `browser_path` | `None` | Path to Chromium executable. Auto detected if omitted. |
| `headless` | `False` | Run browser in headless mode. |
| `timeout` | `300.0` | Max seconds to wait for token extraction. |
| `max_attempts` | `12` | Max attempts for solving a challenge. |
| `human_delay_mean` | `0.2` | Mean delay in seconds between interactions. |
| `human_delay_sigma` | `0.1` | Standard deviation for human like delays. |
| `log_level` | `"WARNING"` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). |
| `persist_html` | `False` | Persist the generated HTML replicas. |
| `verbose` | `False` | Enable verbose output. |

### Detection thresholds

| Option | Default | Description |
| --- | --- | --- |
| `conf_threshold` | `0.7` | Tile classification confidence threshold. |
| `min_confidence_threshold` | `0.2` | Minimum confidence for top 3 tiles (3x3). |
| `fourth_cell_threshold` | `0.7` | Confidence threshold to include 4th cell. |
| `detection_conf_threshold` | `0.6` | Confidence threshold for 4x4 detection. |
| `default_timeout` | `10.0` | Default timeout for browser operations. |
| `image_download_retries` | `3` | Image download retries. |
| `image_download_retry_delay` | `1.0` | Base delay between retries. |

### Vision LLM options

| Option | Default | Description |
| --- | --- | --- |
| `vision_model` | `None` | Vision LLM model name (e.g., `llama-3.2-90b-vision-instruct`). Enables cloud-based analysis. |
| `vision_api_base` | `None` | OpenAI-compatible API base URL (e.g., `https://llm.datasolved.org/v1`). |
| `vision_api_key` | `None` | API key for the vision LLM provider. |
| `vision_timeout` | `30.0` | Timeout for vision LLM API requests in seconds. |

### Other options

| Option | Default | Description |
| --- | --- | --- |
| `register_signal_handlers` | `True` | Register SIGINT/SIGTERM handlers for cleanup. |
| `cleanup_tmp_on_close` | `True` | Delete `download_dir` when solver closes. |

### Solve parameters

Both `RecaptchaSolver.solve()` and `AsyncRecaptchaSolver.solve()` support:

- `website_key`: reCAPTCHA site key (required)
- `website_url`: URL containing the challenge (required)
- `is_invisible`: Whether this is invisible/v3 (default `False`)
- `action`: Action name for invisible reCAPTCHA (optional)
- `is_enterprise`: Whether this is enterprise reCAPTCHA (default `False`)
- `api_domain`: `google.com` or `recaptcha.net` (default `google.com`)
- `bypass_domain_check`: Skip domain verification (default `True`)
- `use_ssl`: Use HTTPS for local server (default `True`)
- `cookies`: Optional list of cookie dictionaries to set before solving
- `user_agent`: Optional custom user agent string

## Result object

`SolveResult` is returned for successful solves:

- `token`: reCAPTCHA response token
- `cookies`: cookies from the browser session
- `time_taken`: wall clock time in seconds
- `captcha_type`: one of `dynamic_3x3`, `selection_3x3`, `square_4x4`, `invisible`, `no_challenge`, `unknown`
- `attempts`: number of solve attempts

## Model assets

- Classification model: [`recaptcha_classification_57k.onnx`](https://huggingface.co/DannyLuna/recaptcha-classification-57k)
  - Trained on a [57k images dataset](https://huggingface.co/datasets/DannyLuna/recaptcha-57k-images-dataset).
  - Auto downloaded on first use from Hugging Face if not present.
  - Override with `SolverConfig(model_path=...)`.
- Detection model: `yolo12x.pt`
  - Loaded by `ultralytics` (auto downloaded if not present).
  - Override with `SolverConfig(detection_model_path=...)`.

## Example scripts

The repo includes two runnable demos that hit Google's public test page and show token extraction timing and captcha type detection. Treat them as templates.

- `demo.py`: synchronous solve with `RecaptchaSolver`
  - Run: `python demo.py`
- `demo_async.py`: async workflows with `AsyncRecaptchaSolver`
  - Option 1 runs two solves concurrently
  - Option 2 runs a solve in the background while other tasks execute
  - Run: `python demo_async.py`

## Development

Run tests:

```bash
pytest
```

Type checking:

```bash
mypy src/
```

Linting:

```bash
ruff check src/
```

## Project layout

- `src/vision_ai_recaptcha_solver/` - core library
- `src/vision_ai_recaptcha_solver/captcha/` - handlers for challenge types
- `src/vision_ai_recaptcha_solver/detector/` - YOLO model integration + Vision LLM detector
- `src/vision_ai_recaptcha_solver/models/` - model assets
- `tests/` - unit tests
- `demo.py`, `demo_async.py` - example scripts
- `.opencode/mcp.json` - MCP server configuration

## Troubleshooting

- `browser_path does not exist`: Pass a valid Chromium executable path in `SolverConfig(browser_path=...)`. On Windows, the path should end with `.exe`.
- `Invalid proxy URL format`: Proxy must be `protocol://[user:pass@]host:port`. Supported protocols: `http`, `https`, `socks4`, `socks5`. [] means optional.
- `CaptchaNotFoundError` or `ElementNotFoundError`: Confirm `website_url` points to a page that renders the reCAPTCHA iframe and that the `website_key` matches that page.
- `Port or temp directory conflicts`: When running multiple solvers, set unique `server_port` and `download_dir` values, or let them auto-select by not setting those fields explicitly.
- Vision LLM errors: Ensure `vision_api_base`, `vision_api_key`, and `vision_model` are correctly set. Check API endpoint compatibility (OpenAI-compatible `/chat/completions`).

## Legal & Responsible Use

RecaptchaDomainReplicator is provided for educational and research purposes. Use responsibly and comply with all applicable laws and terms of service.

## License

This project is licensed under the [MIT License](LICENSE).