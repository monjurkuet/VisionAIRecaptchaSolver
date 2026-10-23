"""MCP Server for VisionAIRecaptchaSolver.

This module provides an MCP (Model Context Protocol) server that exposes
the reCAPTCHA solver functionality to AI agents.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    TextContent,
)

from vision_ai_recaptcha_solver import (
    RecaptchaSolver,
    AsyncRecaptchaSolver,
    SolverConfig,
)
from vision_ai_recaptcha_solver.types import CaptchaType, SolveResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("vision_ai_recaptcha_solver.mcp")


class MCPCaptchaSolver:
    """MCP wrapper for the reCAPTCHA solver."""

    def __init__(self):
        self._sync_solver: RecaptchaSolver | None = None
        self._async_solver: AsyncRecaptchaSolver | None = None
        self._config: SolverConfig | None = None

    def create_config(self, config_dict: dict[str, Any]) -> SolverConfig:
        """Create a SolverConfig from a dictionary."""
        # Handle vision config - convert empty strings to None
        vision_fields = ["vision_model", "vision_api_base", "vision_api_key", "vision_timeout"]
        for field in vision_fields:
            if field in config_dict and config_dict[field] == "":
                config_dict[field] = None

        return SolverConfig(**config_dict)

    def get_sync_solver(self, config: SolverConfig | None = None) -> RecaptchaSolver:
        """Get or create the synchronous solver."""
        if self._sync_solver is None or config is not None:
            if self._sync_solver:
                self._sync_solver.close()
            self._sync_solver = RecaptchaSolver(config or self._config or SolverConfig())
            if config:
                self._config = config
        return self._sync_solver

    async def get_async_solver(self, config: SolverConfig | None = None) -> AsyncRecaptchaSolver:
        """Get or create the asynchronous solver."""
        if self._async_solver is None or config is not None:
            if self._async_solver:
                await self._async_solver.close()
            self._async_solver = AsyncRecaptchaSolver(config or self._config or SolverConfig())
            if config:
                self._config = config
        return self._async_solver

    def solve_sync(
        self,
        website_key: str,
        website_url: str,
        config: SolverConfig | None = None,
        **kwargs,
    ) -> SolveResult:
        """Solve a reCAPTCHA synchronously."""
        solver = self.get_sync_solver(config)
        return solver.solve(website_key=website_key, website_url=website_url, **kwargs)

    async def solve_async(
        self,
        website_key: str,
        website_url: str,
        config: SolverConfig | None = None,
        **kwargs,
    ) -> SolveResult:
        """Solve a reCAPTCHA asynchronously."""
        solver = await self.get_async_solver(config)
        return await solver.solve(website_key=website_key, website_url=website_url, **kwargs)

    def close(self):
        """Close all solvers."""
        if self._sync_solver:
            self._sync_solver.close()
            self._sync_solver = None
        if self._async_solver:
            self._async_solver = None


# Global solver instance
_solver = MCPCaptchaSolver()

# Create MCP server using MCPServer (mcp 2.x API)
app = MCPServer(
    name="vision-ai-recaptcha-solver",
    version="1.0.5",
    description="AI-powered reCAPTCHA v2/v3 solver using YOLO and Vision LLM",
)


@app.tool()
def solve_recaptcha(
    website_key: str,
    website_url: str,
    is_invisible: bool = False,
    action: str | None = None,
    is_enterprise: bool = False,
    api_domain: str = "google.com",
    config: dict[str, Any] | None = None,
) -> CallToolResult:
    """Solve a reCAPTCHA v2/v3 challenge and return the token."""
    try:
        solver_config = None
        if config:
            solver_config = _solver.create_config(config)

        result = _solver.solve_sync(
            website_key=website_key,
            website_url=website_url,
            is_invisible=is_invisible,
            action=action,
            is_enterprise=is_enterprise,
            api_domain=api_domain,
            config=solver_config,
        )

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "token": result.token,
                            "time_taken": result.time_taken,
                            "captcha_type": result.captcha_type.value,
                            "attempts": result.attempts,
                            "cookies": result.cookies,
                        },
                        indent=2,
                    ),
                )
            ]
        )
    except Exception as e:
        logger.exception(f"Error in solve_recaptcha")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True,
        )


@app.tool()
async def solve_recaptcha_async(
    website_key: str,
    website_url: str,
    is_invisible: bool = False,
    action: str | None = None,
    is_enterprise: bool = False,
    api_domain: str = "google.com",
    config: dict[str, Any] | None = None,
) -> CallToolResult:
    """Solve a reCAPTCHA v2/v3 challenge asynchronously."""
    try:
        solver_config = None
        if config:
            solver_config = _solver.create_config(config)

        result = await _solver.solve_async(
            website_key=website_key,
            website_url=website_url,
            is_invisible=is_invisible,
            action=action,
            is_enterprise=is_enterprise,
            api_domain=api_domain,
            config=solver_config,
        )

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "token": result.token,
                            "time_taken": result.time_taken,
                            "captcha_type": result.captcha_type.value,
                            "attempts": result.attempts,
                            "cookies": result.cookies,
                        },
                        indent=2,
                    ),
                )
            ]
        )
    except Exception as e:
        logger.exception(f"Error in solve_recaptcha_async")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True,
        )


@app.tool()
def demo_recaptcha(
    headless: bool = True,
    config: dict[str, Any] | None = None,
) -> CallToolResult:
    """Run a demo solve using Google's test reCAPTCHA."""
    try:
        solver_config = None
        if config:
            solver_config = _solver.create_config(config)

        if solver_config is None:
            solver_config = SolverConfig(headless=headless, timeout=180.0)
        else:
            solver_config.headless = headless

        result = _solver.solve_sync(
            website_key="6Le-wvkSAAAAAPBMRTvw0Q4Muexq9bi0DJwx_mJ-",
            website_url="https://www.google.com/recaptcha/api2/demo",
            config=solver_config,
        )

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "token": result.token[:50] + "...",
                            "time_taken": result.time_taken,
                            "captcha_type": result.captcha_type.value,
                            "attempts": result.attempts,
                            "message": "Demo completed successfully!",
                        },
                        indent=2,
                    ),
                )
            ]
        )
    except Exception as e:
        logger.exception(f"Error in demo_recaptcha")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True,
        )


@app.tool()
def get_solver_info() -> CallToolResult:
    """Get information about the solver capabilities and configuration."""
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "name": "VisionAIRecaptchaSolver",
                        "version": "1.0.5",
                        "description": "AI-powered reCAPTCHA v2/v3 solver using YOLO and Vision LLM",
                        "capabilities": [
                            "reCAPTCHA v2 image challenges (3x3 dynamic, 3x3 selection, 4x4 square)",
                            "reCAPTCHA v3 invisible flow",
                            "YOLO-based local detection",
                            "Vision LLM (OpenAI-compatible) for cloud-based analysis",
                            "Proxy support (HTTP/HTTPS/SOCKS4/SOCKS5)",
                            "Headless browser mode",
                            "Sync and async APIs",
                        ],
                        "captcha_types": [t.value for t in CaptchaType],
                        "vision_llm_support": True,
                        "default_vision_model": "llama-3.2-90b-vision-instruct",
                        "default_vision_api_base": "https://llm.datasolved.org/v1",
                    },
                    indent=2,
                ),
            )
        ]
    )


async def main():
    """Run the MCP server."""
    logger.info("Starting VisionAIRecaptchaSolver MCP server")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())