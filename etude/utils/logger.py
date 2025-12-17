# etude/utils/logger.py

"""
Logging system for Etude project.

Usage:
    from etude.utils.logger import logger

    logger.info("Device initialized")
    logger.warn("File not found, using default")
    logger.error("Failed to load model")
    logger.debug("Tensor shape: (32, 128)")  # Only shown when level=DEBUG
    logger.stage(1, "Extracting features")
    logger.substep("Loading audio file")
    logger.skip("Already processed")
    logger.success("Pipeline completed")

Environment Variables:
    ETUDE_LOG_LEVEL: DEBUG, INFO, WARN, ERROR (default: INFO)
    ETUDE_NO_COLOR: Set to disable colored output
"""

import os
import sys
from enum import IntEnum
from typing import Optional


class LogLevel(IntEnum):
    """Log levels in ascending order of severity."""
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40


class EtudeLogger:
    """
    Singleton logger for the Etude project.

    Provides consistent log formatting across all modules while maintaining
    compatibility with the existing print-based output style.
    """

    _instance: Optional["EtudeLogger"] = None

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[94m",     # Blue
        "WARN": "\033[93m",     # Yellow
        "ERROR": "\033[91m",    # Red
        "SUCCESS": "\033[92m",  # Green
        "SKIP": "\033[90m",     # Gray
        "STAGE": "\033[95m",    # Magenta
        "RESET": "\033[0m",
    }

    def __new__(cls) -> "EtudeLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._level = self._get_level_from_env()
        self._use_color = self._should_use_color()

    def _get_level_from_env(self) -> LogLevel:
        """Get log level from environment variable."""
        level_str = os.environ.get("ETUDE_LOG_LEVEL", "INFO").upper()
        level_map = {
            "DEBUG": LogLevel.DEBUG,
            "INFO": LogLevel.INFO,
            "WARN": LogLevel.WARN,
            "WARNING": LogLevel.WARN,
            "ERROR": LogLevel.ERROR,
        }
        return level_map.get(level_str, LogLevel.INFO)

    def _should_use_color(self) -> bool:
        """Determine if colored output should be used."""
        # Disable color if explicitly requested
        if os.environ.get("ETUDE_NO_COLOR"):
            return False
        # Disable color if not a TTY (e.g., piped output)
        if not sys.stdout.isatty():
            return False
        # Disable color on Windows unless using Windows Terminal
        if sys.platform == "win32" and "WT_SESSION" not in os.environ:
            return False
        return True

    def set_level(self, level: str) -> None:
        """
        Set the logging level programmatically.

        Args:
            level: One of 'DEBUG', 'INFO', 'WARN', 'ERROR'
        """
        level_map = {
            "DEBUG": LogLevel.DEBUG,
            "INFO": LogLevel.INFO,
            "WARN": LogLevel.WARN,
            "WARNING": LogLevel.WARN,
            "ERROR": LogLevel.ERROR,
        }
        self._level = level_map.get(level.upper(), LogLevel.INFO)

    def set_color(self, enabled: bool) -> None:
        """Enable or disable colored output."""
        self._use_color = enabled

    def _colorize(self, text: str, color_key: str) -> str:
        """Apply color to text if colors are enabled."""
        if self._use_color and color_key in self.COLORS:
            return f"{self.COLORS[color_key]}{text}{self.COLORS['RESET']}"
        return text

    def _log(self, level: LogLevel, prefix: str, message: str,
             color_key: str, file=None, newline_before: bool = False) -> None:
        """Internal logging method."""
        if level < self._level:
            return

        output = file or sys.stdout
        formatted_prefix = self._colorize(prefix, color_key)

        if newline_before:
            print(file=output)
        print(f"{formatted_prefix} {message}", file=output)

    # === Primary logging methods ===

    def debug(self, message: str) -> None:
        """Log a debug message. Only shown when level is DEBUG."""
        self._log(LogLevel.DEBUG, "[DEBUG]", message, "DEBUG")

    def info(self, message: str) -> None:
        """Log an info message."""
        self._log(LogLevel.INFO, "[INFO]", message, "INFO")

    def warn(self, message: str) -> None:
        """Log a warning message."""
        self._log(LogLevel.WARN, "[WARN]", message, "WARN")

    def error(self, message: str) -> None:
        """Log an error message to stderr."""
        self._log(LogLevel.ERROR, "[ERROR]", message, "ERROR", file=sys.stderr)

    # === Semantic logging methods ===

    def success(self, message: str) -> None:
        """Log a success message."""
        self._log(LogLevel.INFO, "[SUCCESS]", message, "SUCCESS")

    def skip(self, message: str) -> None:
        """Log a skip message (operation was skipped)."""
        self._log(LogLevel.INFO, "[SKIP]", message, "SKIP")

    def resume(self, message: str) -> None:
        """Log a resume message (resuming from checkpoint)."""
        self._log(LogLevel.INFO, "[RESUME]", message, "INFO")

    # === Structural logging methods ===

    def stage(self, number: int, name: str) -> None:
        """
        Log a stage header with decorative borders.

        Args:
            number: Stage number (1, 2, 3, ...)
            name: Stage name/description
        """
        if self._level > LogLevel.INFO:
            return

        header = f" Stage {number}: {name} "
        border = "=" * 25
        formatted = f"\n{border}{header}{border}"

        if self._use_color:
            formatted = self._colorize(formatted, "STAGE")

        print(formatted)

    def step(self, message: str) -> None:
        """
        Log a step message with a bullet point prefix.

        Args:
            message: The step description
        """
        if self._level > LogLevel.INFO:
            return

        prefix = " • "
        print(f"{prefix}{message}")

    def substep(self, message: str, indent: int = 1) -> None:
        """
        Log a substep message with indentation.

        Args:
            message: The substep description
            indent: Indentation level (1 = "    | ", 2 = "        | ", etc.)
        """
        if self._level > LogLevel.INFO:
            return

        prefix = "    " * indent + "| "
        print(f"{prefix}{message}")

    def section(self, title: str) -> None:
        """
        Log a section header (smaller than stage).

        Args:
            title: Section title
        """
        if self._level > LogLevel.INFO:
            return

        header = f" {title} "
        border = "-" * 20
        formatted = f"\n{border}{header}{border}"
        print(formatted)

    # === Report formatting ===

    def report_header(self, title: str) -> None:
        """Print a report header with borders."""
        if self._level > LogLevel.INFO:
            return

        border = "=" * 25
        print(f"\n{border} {title} {border}")

    def report_separator(self, width: int = 75) -> None:
        """Print a separator line."""
        if self._level > LogLevel.INFO:
            return
        print("=" * width)


# Global singleton instance
logger = EtudeLogger()
