# AGENTS.md

This file provides guidance for AI agents working with the VisionAIRecaptchaSolver codebase.

## Project Overview

**VisionAIRecaptchaSolver** is an AI-powered reCAPTCHA v2/v3 solver for Python. It uses:
- **YOLO models** (ultralytics) for local image classification/detection
- **Vision LLM** (optional) for cloud-based image analysis via OpenAI-compatible APIs
- **DrissionPage** for browser automation
- **recaptcha-domain-replicator** for captcha iframe handling

## Architecture

```
src/vision_ai_recaptcha_solver/
├── solver.py              # Sync RecaptchaSolver (main entry point)
├── async_solver.py        # Async AsyncRecaptchaSolver
├── config.py              # SolverConfig dataclass
├── constants.py           # Selectors, timeouts, defaults
├── types.py               # CaptchaType, SolveResult, DetectionResult, mappings
├── exceptions.py          # Custom exception hierarchy
├── logging_config.py      # Logging setup
├── resource_allocation.py # Port/process management
├── utils.py               # Human delays, helpers
├── detector/
│   ├── yolo_detector.py   # YOLO-based classification/detection
│   ├── vision_detector.py # Vision LLM detector (OpenAI-compatible API)
│   ├── grid_utils.py      # Grid cell calculations
│   └── __init__.py
├── captcha/
│   ├── base_handler.py    # Abstract BaseCaptchaHandler
│   ├── dynamic_handler.py # Dynamic 3x3 (images refresh)
│   ├── selection_handler.py # Selection 3x3 (one-time)
│   ├── square_handler.py  # Square 4x4 (object detection)
│   ├── image_utils.py     # Download, composite, load images
│   └── __init__.py
├── browser/
│   ├── navigation.py      # Iframe handling, clicking, target extraction
│   └── __init__.py
└── __main__.py            # CLI entry point
```

## Key Classes

### SolverConfig
Configuration dataclass with all solver options:
- **YOLO**: `model_path`, `detection_model_path`
- **Vision LLM**: `vision_model`, `vision_api_base`, `vision_api_key`, `vision_timeout`
- **Browser**: `browser_path`, `headless`, `proxy`
- **Thresholds**: `conf_threshold`, `min_confidence_threshold`, `fourth_cell_threshold`, `detection_conf_threshold`
- **Behavior**: `timeout`, `max_attempts`, `human_delay_mean`, `human_delay_sigma`

### RecaptchaSolver (sync) / AsyncRecaptchaSolver (async)
Main solver classes. Usage:
```python
from vision_ai_recaptcha_solver import RecaptchaSolver, SolverConfig

config = SolverConfig(
    vision_model="llama-3.2-90b-vision-instruct",
    vision_api_base="https://llm.datasolved.org/v1",
    vision_api_key="your-api-key",
    headless=True,
)

with RecaptchaSolver(config) as solver:
    result = solver.solve(
        website_key="6Le-wvkSAAAAAPBMRTvw0Q4Muexq9bi0DJwx_mJ-",
        website_url="https://example.com/page-with-recaptcha"
    )
    print(result.token)
```

### Captcha Types
- `DYNAMIC_3X3` - Images refresh after each click
- `SELECTION_3X3` - One-time selection, then verify
- `SQUARE_4X4` - 4x4 grid with object detection
- `INVISIBLE` - reCAPTCHA v3 (no challenge)
- `NO_CHALLENGE` - Checkbox only

### Vision LLM Integration
When `vision_model`, `vision_api_base`, and `vision_api_key` are configured:
1. Target keyword extracted from captcha challenge text
2. Full grid image sent to vision LLM with prompt
3. LLM returns cell numbers + confidences
4. Falls back to YOLO if vision detector fails

Supported providers: Any OpenAI-compatible API (Datasolved, OpenRouter, etc.)

## Development Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Type checking
mypy src/

# Linting
ruff check src/

# Run CLI demo
python -m vision_ai_recaptcha_solver demo --headless

# Solve via CLI
python -m vision_ai_recaptcha_solver solve -k SITE_KEY -u URL
```

## Common Tasks

### Adding a New Captcha Type
1. Add to `CaptchaType` enum in `types.py`
2. Create handler in `captcha/new_handler.py` extending `BaseCaptchaHandler`
3. Register in `solver.py` and `async_solver.py` `_handlers` dict
4. Add detection logic in `detector/` if needed

### Modifying Detection Thresholds
Edit defaults in `SolverConfig` or pass via config:
```python
config = SolverConfig(
    conf_threshold=0.5,        # Lower = more detections
    min_confidence_threshold=0.1,
    fourth_cell_threshold=0.6,
    detection_conf_threshold=0.4,
)
```

### Using Custom YOLO Models
```python
config = SolverConfig(
    model_path="path/to/classification.onnx",
    detection_model_path="path/to/detection.pt",
)
```

## Testing

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_config.py -v

# With coverage
pytest tests/ --cov=vision_ai_recaptcha_solver
```

## Dependencies

Core:
- `recaptcha-domain-replicator>=1.0.6` - Captcha replication
- `ultralytics>=8.3` - YOLO models
- `opencv-python>=4.8.0` - Image processing
- `numpy>=1.24.0` - Arrays
- `aiohttp>=3.9.0` - Async HTTP (vision LLM)
- `requests>=2.28.0` - Sync HTTP
- `click>=8.1.0` - CLI

Dev:
- `pytest>=7.0`, `pytest-asyncio>=0.21`
- `ruff>=0.1.0`, `mypy>=1.0`

## Environment

- Python 3.10+
- Chromium-based browser (Chrome for Testing, Edge)
- Optional: Proxy with credentials for enterprise

## Error Handling

Key exceptions:
- `ModelNotFoundError` - Model files missing
- `LowConfidenceError` - Detection confidence too low
- `CaptchaNotFoundError` - Iframe/element not found
- `TokenExtractionError` - Token not obtained
- `UnsupportedCaptchaError` - Unknown captcha type

All inherit from `RecaptchaSolverError`.

## MCP Server

See `.opencode/mcp.json` for MCP server configuration.

### Configuration (`.opencode/mcp.json`)
```json
{
  "mcpServers": {
    "vision-ai-recaptcha-solver": {
      "command": "python",
      "args": ["-m", "vision_ai_recaptcha_solver.mcp_server"],
      "cwd": "${workspaceFolder}",
      "env": { "PYTHONPATH": "${workspaceFolder}/src" }
    }
  }
}
```

### Available MCP Tools
- `solve_recaptcha` - Sync solve with full config options
- `solve_recaptcha_async` - Async solve for better performance
- `demo_recaptcha` - Run demo on Google's test reCAPTCHA
- `get_solver_info` - Get capabilities and configuration info

### Running MCP Server
```bash
python -m vision_ai_recaptcha_solver.mcp_server
```

## Dependencies