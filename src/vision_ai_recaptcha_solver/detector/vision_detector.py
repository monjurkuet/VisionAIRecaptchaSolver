"""Vision LLM detector for captcha image classification."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp
import cv2
import numpy as np

from vision_ai_recaptcha_solver.config import SolverConfig
from vision_ai_recaptcha_solver.detector.grid_utils import calculate_4x4_cells
from vision_ai_recaptcha_solver.exceptions import DetectionError, ModelNotFoundError
from vision_ai_recaptcha_solver.types import (
    COCO_TARGET_MAPPINGS,
    CaptchaType,
    DetectionResult,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class VisionLLMConfig:
    """Configuration for the Vision LLM detector."""

    model: str
    api_base: str
    api_key: str
    timeout: float = 30.0


class VisionLLMDetector:
    """Detector for reCAPTCHA challenges using a Vision LLM.

    Uses a vision-language model to analyze captcha images and identify
    which grid cells contain the target object.
    """

    # Target keywords that the vision model should recognize
    TARGET_KEYWORDS = {
        "bicycle", "bicycles", "bridge", "bridges", "bus", "buses",
        "car", "cars", "automobile", "taxi", "taxis",
        "chimney", "chimneys", "crosswalk", "crosswalks",
        "hydrant", "hydrants", "fire hydrant",
        "motorcycle", "motorcycles", "mountain", "mountains",
        "palm", "palm tree", "palm trees",
        "stair", "stairs", "tractor", "tractors",
        "traffic", "traffic light", "traffic lights",
    }

    def __init__(
        self,
        config: VisionLLMConfig,
        logger: logging.Logger | None = None,
        conf_threshold: float = 0.7,
        fourth_cell_threshold: float = 0.7,
        detection_conf_threshold: float = 0.6,
    ) -> None:
        """Initialize the Vision LLM detector.

        Args:
            config: Vision LLM configuration.
            logger: Logger instance. If None, creates a new one.
            conf_threshold: Confidence threshold for tile classification.
            fourth_cell_threshold: Threshold to include a 4th cell in selection.
            detection_conf_threshold: Confidence threshold for 4x4 detection model.
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.conf_threshold = conf_threshold
        self.fourth_cell_threshold = fourth_cell_threshold
        self.detection_conf_threshold = detection_conf_threshold

        # Validate configuration
        if not self.config.api_key:
            raise ModelNotFoundError("vision_api_key is required for VisionLLMDetector")

        self.logger.debug(f"Initialized VisionLLMDetector with model: {self.config.model}")

    def _encode_image(self, image: NDArray[np.uint8]) -> str:
        """Encode an image as base64 for the vision API.

        Args:
            image: Image as numpy array (BGR format from cv2).

        Returns:
            Base64 encoded image string.
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Encode as PNG
        success, buffer = cv2.imencode(".png", image_rgb)
        if not success:
            raise DetectionError("Failed to encode image as PNG")

        return base64.b64encode(buffer).decode("utf-8")

    def _build_prompt(self, keyword: str, grid_size: int) -> str:
        """Build the prompt for the vision LLM.

        Args:
            keyword: The target object keyword (e.g., "bus", "traffic light").
            grid_size: The grid size (3 for 3x3, 4 for 4x4).

        Returns:
            The prompt string.
        """
        return (
            f"You are analyzing a reCAPTCHA {grid_size}x{grid_size} grid image. "
            f"Your task is to identify which grid cells contain a '{keyword}'.\n\n"
            f"The image is divided into a {grid_size}x{grid_size} grid, with cells numbered "
            f"from 1 to {grid_size * grid_size} (left to right, top to bottom).\n\n"
            f"Please respond with a JSON object containing:\n"
            f"- 'cells': array of cell numbers (1-indexed) that contain the '{keyword}'\n"
            f"- 'confidences': array of confidence scores (0.0-1.0) for each cell in the same order\n\n"
            f"Example response: {{\"cells\": [1, 5, 9], \"confidences\": [0.95, 0.87, 0.92]}}\n\n"
            f"Only include cells where you are confident the '{keyword}' is present. "
            f"If no cells contain the '{keyword}', return empty arrays."
        )

    async def _call_vision_api(
        self,
        image_b64: str,
        prompt: str,
    ) -> dict[str, Any]:
        """Call the vision LLM API.

        Args:
            image_b64: Base64 encoded image.
            prompt: The prompt for the vision model.

        Returns:
            API response as a dictionary.
        """
        url = f"{self.config.api_base.rstrip('/')}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 500,
        }

        timeout = aiohttp.ClientTimeout(total=self.config.timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise DetectionError(
                        f"Vision API error ({response.status}): {error_text}"
                    )
                return await response.json()

    def _parse_response(self, response: dict[str, Any]) -> tuple[list[int], list[float]]:
        """Parse the vision LLM response.

        Args:
            response: API response.

        Returns:
            Tuple of (cell_numbers, confidences).
        """
        try:
            content = response["choices"][0]["message"]["content"]
            # Try to extract JSON from the response
            import json

            # Find JSON in the response (might be wrapped in code blocks)
            import re

            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(content)

            cells = data.get("cells", [])
            confidences = data.get("confidences", [])

            # Validate
            if not isinstance(cells, list) or not isinstance(confidences, list):
                raise DetectionError("Invalid response format: cells and confidences must be arrays")

            if len(cells) != len(confidences):
                raise DetectionError("Invalid response format: cells and confidences length mismatch")

            return cells, confidences

        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise DetectionError(f"Failed to parse vision LLM response: {e}") from e

    async def _analyze_grid(
        self,
        image: NDArray[np.uint8],
        keyword: str,
        grid_size: int,
    ) -> tuple[list[int], list[float]]:
        """Analyze a grid image using the vision LLM.

        Args:
            image: The captcha image.
            keyword: Target keyword.
            grid_size: Grid size (3 or 4).

        Returns:
            Tuple of (cell_numbers, confidences).
        """
        image_b64 = self._encode_image(image)
        prompt = self._build_prompt(keyword, grid_size)

        response = await self._call_vision_api(image_b64, prompt)
        return self._parse_response(response)

    def _normalize_keyword(self, keyword: str) -> str:
        """Normalize keyword for matching.

        Args:
            keyword: Raw keyword from captcha.

        Returns:
            Normalized keyword.
        """
        keyword = keyword.strip().lower()
        # Handle multi-word targets
        if keyword in self.TARGET_KEYWORDS:
            return keyword
        # Try to match common variations
        for target in self.TARGET_KEYWORDS:
            if target in keyword or keyword in target:
                return target
        return keyword

    async def classify_tiles_with_confidence(
        self,
        main_image: NDArray[np.uint8],
        grid_size: int,
        target_class: int,  # Not used for vision LLM, but kept for interface compatibility
    ) -> list[tuple[int, float]]:
        """Classify all tiles in a grid and return their confidences.

        For the vision LLM, we use the keyword directly instead of target_class.

        Args:
            main_image: The main captcha image.
            grid_size: Number of cells per row (3 or 4).
            target_class: Ignored for vision LLM (uses keyword from context).

        Returns:
            List of (cell_num, confidence) tuples for all cells.
        """
        # Note: This method is called from handlers that have the keyword
        # We need to get the keyword from somewhere - for now we'll raise an error
        # and the handlers need to be updated to use the new method
        raise NotImplementedError(
            "VisionLLMDetector requires keyword-based analysis. "
            "Use analyze_with_keyword instead."
        )

    async def analyze_with_keyword(
        self,
        main_image: NDArray[np.uint8],
        keyword: str,
        grid_size: int,
    ) -> list[tuple[int, float]]:
        """Analyze grid using a keyword.

        Args:
            main_image: The main captcha image.
            keyword: Target keyword (e.g., "bus", "traffic light").
            grid_size: Grid size (3 or 4).

        Returns:
            List of (cell_num, confidence) tuples for all cells.
        """
        normalized_keyword = self._normalize_keyword(keyword)
        self.logger.debug(f"Analyzing {grid_size}x{grid_size} grid for target: '{normalized_keyword}'")

        cells, confidences = await self._analyze_grid(main_image, normalized_keyword, grid_size)

        # Convert to full grid results (all cells with confidences)
        total_cells = grid_size * grid_size
        results: list[tuple[int, float]] = []

        # Create mapping from cell to confidence
        cell_conf_map = dict(zip(cells, confidences, strict=True))

        for cell_num in range(1, total_cells + 1):
            conf = cell_conf_map.get(cell_num, 0.0)
            results.append((cell_num, conf))
            self.logger.debug(f"Tile {cell_num}: {normalized_keyword} conf {conf:.2f}")

        return results

    def get_target_class(self, keyword: str) -> int | None:
        """Get the class index for a target keyword (compatibility method).

        Args:
            keyword: Target keyword.

        Returns:
            Dummy class index (always 0) since vision LLM doesn't use class indices.
        """
        normalized = self._normalize_keyword(keyword)
        if normalized in self.TARGET_KEYWORDS:
            return 0  # Dummy class index
        return None

    def get_coco_target_class(self, keyword: str) -> int | None:
        """Get the COCO class index for a target keyword (compatibility method).

        Args:
            keyword: Target keyword.

        Returns:
            COCO class index, or None if not supported for detection.
        """
        normalized = self._normalize_keyword(keyword)
        return COCO_TARGET_MAPPINGS.get(normalized)

    async def detect_for_grid(
        self,
        image: NDArray[np.uint8],
        target_class: int,
        grid_size: int = 450,
        conf_threshold: float | None = None,
    ) -> list[int]:
        """Detect objects and map them to grid cells.

        For vision LLM, we need the keyword. Since this method is called from
        SquareCaptchaHandler which has the keyword, we need a different approach.
        This is kept for interface compatibility but will raise if used directly.

        Args:
            image: The captcha image.
            target_class: COCO class index (ignored for vision LLM).
            grid_size: Total size of the grid in pixels.
            conf_threshold: Confidence threshold.

        Returns:
            List of 1-indexed cell numbers containing targets.
        """
        raise NotImplementedError(
            "VisionLLMDetector.detect_for_grid requires keyword. "
            "Use detect_for_grid_with_keyword instead."
        )

    async def detect_for_grid_with_keyword(
        self,
        image: NDArray[np.uint8],
        keyword: str,
        grid_size: int = 450,
        conf_threshold: float | None = None,
    ) -> list[int]:
        """Detect objects using a keyword and map them to grid cells.

        Args:
            image: The captcha image.
            keyword: Target keyword.
            grid_size: Total size of the grid in pixels (default 450 for 4x4).
            conf_threshold: Confidence threshold.

        Returns:
            List of 1-indexed cell numbers containing targets.
        """
        normalized_keyword = self._normalize_keyword(keyword)
        self.logger.debug(f"Detecting 4x4 grid for target: '{normalized_keyword}'")

        cells, confidences = await self._analyze_grid(image, normalized_keyword, 4)

        threshold = conf_threshold or self.detection_conf_threshold
        filtered_cells = [c for c, conf in zip(cells, confidences, strict=True) if conf >= threshold]

        self.logger.debug(f"Vision LLM detected cells: {filtered_cells}")
        return sorted(filtered_cells)

    # Synchronous wrapper methods for use in synchronous solver
    def analyze_with_keyword_sync(
        self,
        main_image: NDArray[np.uint8],
        keyword: str,
        grid_size: int,
    ) -> list[tuple[int, float]]:
        """Synchronous wrapper for analyze_with_keyword.

        Args:
            main_image: The main captcha image.
            keyword: Target keyword (e.g., "bus", "traffic light").
            grid_size: Grid size (3 or 4).

        Returns:
            List of (cell_num, confidence) tuples for all cells.
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If there's already a running loop, we need to run in a thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.analyze_with_keyword(main_image, keyword, grid_size),
                    )
                    return future.result()
            else:
                return asyncio.run(self.analyze_with_keyword(main_image, keyword, grid_size))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.analyze_with_keyword(main_image, keyword, grid_size))

    def detect_for_grid_with_keyword_sync(
        self,
        image: NDArray[np.uint8],
        keyword: str,
        grid_size: int = 450,
        conf_threshold: float | None = None,
    ) -> list[int]:
        """Synchronous wrapper for detect_for_grid_with_keyword.

        Args:
            image: The captcha image.
            keyword: Target keyword.
            grid_size: Total size of the grid in pixels (default 450 for 4x4).
            conf_threshold: Confidence threshold.

        Returns:
            List of 1-indexed cell numbers containing targets.
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.detect_for_grid_with_keyword(image, keyword, grid_size, conf_threshold),
                    )
                    return future.result()
            else:
                return asyncio.run(self.detect_for_grid_with_keyword(image, keyword, grid_size, conf_threshold))
        except RuntimeError:
            return asyncio.run(self.detect_for_grid_with_keyword(image, keyword, grid_size, conf_threshold))


def create_vision_detector(config: SolverConfig, logger: logging.Logger | None = None) -> VisionLLMDetector | None:
    """Create a VisionLLMDetector if vision configuration is provided.

    Args:
        config: Solver configuration.
        logger: Logger instance.

    Returns:
        VisionLLMDetector instance or None if not configured.
    """
    if not config.vision_model or not config.vision_api_base or not config.vision_api_key:
        return None

    vision_config = VisionLLMConfig(
        model=config.vision_model,
        api_base=config.vision_api_base,
        api_key=config.vision_api_key,
        timeout=config.vision_timeout,
    )

    return VisionLLMDetector(
        config=vision_config,
        logger=logger,
        conf_threshold=config.conf_threshold,
        fourth_cell_threshold=config.fourth_cell_threshold,
        detection_conf_threshold=config.detection_conf_threshold,
    )