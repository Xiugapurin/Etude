# etude/utils/logger.py

"""
Logging system for Etude project.

## Method Selection Guide (Scheme A: Strict Layering)

| Method      | Purpose                        | Format              |
|-------------|--------------------------------|---------------------|
| stage()     | Pipeline major stage header    | ===== Stage N: X =====
| step()      | Action declaration (BEFORE)    | " • Message"  (no punctuation)
| substep()   | Action in progress (DURING)    | "    | Message..."  (with ...)
| info()      | Result notification (AFTER)    | "[INFO] Message."  (with period)
| success()   | Final success message          | "[SUCCESS] Message."
| warn()      | Warning message                | "[WARN] Message."
| error()     | Error message (to stderr)      | "[ERROR] Message."
| skip()      | Skipped operation              | "[SKIP] Message."
| debug()     | Debug info (verbose only)      | "[DEBUG] Message"

## Progress Bar Compatible Methods

Use these methods inside tqdm loops to avoid breaking progress bar display:

| Method          | Purpose                                    |
|-----------------|--------------------------------------------|
| progress_warn() | Warning message (tqdm compatible)          |
| progress_info() | Info message (tqdm compatible)             |
| progress_skip() | Skip message (tqdm compatible)             |

## Format Conventions

1. Capitalization: Always start with uppercase letter
2. Colons: Use before paths - "Saved to: {path}"
3. Periods: Use for completed actions - "Download complete."
4. Ellipsis: Use for in-progress actions - "Processing..."
5. Paths: Use .resolve() for full paths in user-facing messages

## Examples

    # Action declaration (step) - no punctuation
    logger.step("Preparing source audio")
    logger.step(f"Loading model from: {path}")

    # In-progress detail (substep) - with ...
    logger.substep("Initializing separator...")
    logger.substep(f"Processing {name}...")

    # Completion notification (info) - with period
    logger.info("Download complete.")
    logger.info(f"Saved to: {path.resolve()}")

## Environment Variables

    LOG_LEVEL: DEBUG, INFO, WARN, ERROR (default: INFO)
    NO_COLOR: Set to disable colored output
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
        level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
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
        if os.environ.get("NO_COLOR"):
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

    def is_debug(self) -> bool:
        """Check if the current log level is DEBUG."""
        return self._level == LogLevel.DEBUG

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
        formatted = f"\n{border}{header}{border}\n"

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

    # === Report formatting ===

    def report_header(self, title: str) -> None:
        """Print a report header with borders."""
        if self._level > LogLevel.INFO:
            return

        border = "=" * 40
        formatted = f"\n{border} {title} {border}\n"
        if self._use_color:
            formatted = self._colorize(formatted, "STAGE")
        print(formatted)

    def report_section(self, title: str) -> None:
        """
        Print a section header.

        Args:
            title: Section title
        """
        if self._level > LogLevel.INFO:
            return

        header = f" {title} "
        border = "-" * 40
        formatted = f"\n{border}{header}{border}\n"
        if self._use_color:
            formatted = self._colorize(formatted, "INFO")
        print(formatted)

    def report_separator(self, width: int = 75) -> None:
        """Print a separator line."""
        if self._level > LogLevel.INFO:
            return

        formatted = f"\n{'=' * width}\n"
        if self._use_color:
            formatted = self._colorize(formatted, "STAGE")
        print(formatted)

    # === Progress bar compatible logging ===

    def _tqdm_write(self, message: str, file=None) -> None:
        """Write message using tqdm.write() to avoid progress bar interference."""
        try:
            from tqdm import tqdm
            tqdm.write(message, file=file or sys.stdout)
        except ImportError:
            print(message, file=file or sys.stdout)

    def progress_warn(self, message: str) -> None:
        """Log a warning message compatible with tqdm progress bars."""
        if self._level > LogLevel.WARN:
            return
        formatted_prefix = self._colorize("[WARN]", "WARN")
        self._tqdm_write(f"{formatted_prefix} {message}")

    def progress_info(self, message: str) -> None:
        """Log an info message compatible with tqdm progress bars."""
        if self._level > LogLevel.INFO:
            return
        formatted_prefix = self._colorize("[INFO]", "INFO")
        self._tqdm_write(f"{formatted_prefix} {message}")

    def progress_skip(self, message: str) -> None:
        """Log a skip message compatible with tqdm progress bars."""
        if self._level > LogLevel.INFO:
            return
        formatted_prefix = self._colorize("[SKIP]", "SKIP")
        self._tqdm_write(f"{formatted_prefix} {message}")


# Global singleton instance
logger = EtudeLogger()
