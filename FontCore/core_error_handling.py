#!/usr/bin/env python3
"""
Enhanced error handling with detailed context tracking.

Provides structured error information for font processing operations,
including context, recoverability, and exception details.

Usage:
    from FontCore.core_error_handling import ErrorContext, ErrorInfo, ErrorTracker

    # Create error info
    try:
        process_font(path)
    except Exception as e:
        error = ErrorInfo.from_exception(
            context=ErrorContext.LOADING,
            filepath=path,
            exception=e,
            message="Failed to load font file"
        )
        tracker.add_error(error)

    # Check error summary
    summary = tracker.get_summary()
    print(f"Encountered {summary['total_errors']} errors")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from pathlib import Path
import traceback
from datetime import datetime
from FontCore.core_logging_config import get_logger

logger = get_logger(__name__)


class ErrorContext(Enum):
    """
    Error context categories for precise failure point identification.

    Each context represents a distinct phase of font processing where
    errors can occur. Used for categorizing errors and determining
    appropriate recovery strategies.
    """

    # File I/O operations
    FILE_IO = "file_io"  # Disk read/write errors
    LOADING = "loading"  # Font file loading/parsing from disk
    SAVING = "saving"  # Font file save operation

    # Font structure parsing
    PARSING = "parsing"  # Name table extraction
    VALIDATION = "validation"  # Value validation (e.g., invalid nameID)

    # Font table operations
    NAME_TABLE = "name_table"  # Name table operations
    STAT_TABLE = "stat_table"  # STAT table interaction
    FVAR_TABLE = "fvar_table"  # fvar table interaction
    CFF_TABLE = "cff_table"  # CFF/CFF2 table operations
    OS2_TABLE = "os2_table"  # OS/2 table operations

    # Business logic
    CONSTRUCTION = "construction"  # Name construction logic
    POLICY = "policy"  # Name policy application

    # Other
    UNKNOWN = "unknown"  # Unknown or unspecified context

    @property
    def is_recoverable_by_default(self) -> bool:
        """
        Check if errors in this context are typically recoverable.

        Recoverable errors allow processing to continue with other files.
        Non-recoverable errors typically require user intervention.
        """
        non_recoverable = {
            ErrorContext.FILE_IO,  # Filesystem issues
            ErrorContext.UNKNOWN,  # Unknown problems
        }
        return self not in non_recoverable

    @property
    def severity(self) -> str:
        """Get default severity level for this context."""
        critical = {
            ErrorContext.FILE_IO,
            ErrorContext.LOADING,
            ErrorContext.SAVING,
        }
        warning = {
            ErrorContext.VALIDATION,
            ErrorContext.POLICY,
        }

        if self in critical:
            return "critical"
        elif self in warning:
            return "warning"
        else:
            return "error"


class ErrorSeverity(Enum):
    """Error severity levels."""

    DEBUG = "debug"  # Minor issues, logged only
    INFO = "info"  # Informational, no action needed
    WARNING = "warning"  # Potential problem, processing continues
    ERROR = "error"  # Error occurred, file skipped
    CRITICAL = "critical"  # Severe error, may need to stop processing


@dataclass
class ErrorInfo:
    """
    Detailed error information with context.

    Captures all relevant information about an error for logging,
    reporting, and recovery decisions.
    """

    context: ErrorContext
    message: str
    filepath: Optional[str] = None
    exception: Optional[Exception] = None
    recoverable: Optional[bool] = None  # None = use context default
    severity: Optional[ErrorSeverity] = None  # None = use context default
    timestamp: datetime = field(default_factory=datetime.now)
    additional_info: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None

    def __post_init__(self):
        """Set defaults and extract stack trace."""
        # Set default recoverability from context
        if self.recoverable is None:
            self.recoverable = self.context.is_recoverable_by_default

        # Set default severity from context
        if self.severity is None:
            severity_str = self.context.severity
            self.severity = ErrorSeverity(severity_str)

        # Extract stack trace from exception if present
        if self.exception and not self.stack_trace:
            self.stack_trace = "".join(
                traceback.format_exception(
                    type(self.exception), self.exception, self.exception.__traceback__
                )
            )

    @classmethod
    def from_exception(
        cls,
        context: ErrorContext,
        exception: Exception,
        filepath: Optional[str] = None,
        message: Optional[str] = None,
        **kwargs,
    ) -> "ErrorInfo":
        """
        Create ErrorInfo from an exception.

        Args:
            context: Error context
            exception: The exception that occurred
            filepath: Optional file path where error occurred
            message: Optional custom message (uses exception message if not provided)
            **kwargs: Additional ErrorInfo fields

        Returns:
            Populated ErrorInfo instance

        Examples:
            >>> try:
            ...     font = TTFont("bad.ttf")
            ... except Exception as e:
            ...     error = ErrorInfo.from_exception(
            ...         ErrorContext.LOADING,
            ...         e,
            ...         filepath="bad.ttf"
            ...     )
        """
        if message is None:
            message = str(exception)

        return cls(
            context=context,
            message=message,
            filepath=filepath,
            exception=exception,
            **kwargs,
        )

    @property
    def filename(self) -> Optional[str]:
        """Get just the filename from filepath."""
        if self.filepath:
            return Path(self.filepath).name
        return None

    @property
    def exception_type(self) -> Optional[str]:
        """Get exception type name."""
        if self.exception:
            return type(self.exception).__name__
        return None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation suitable for JSON/logging
        """
        return {
            "context": self.context.value,
            "message": self.message,
            "filepath": self.filepath,
            "filename": self.filename,
            "exception_type": self.exception_type,
            "exception_message": str(self.exception) if self.exception else None,
            "recoverable": self.recoverable,
            "severity": self.severity.value if self.severity else None,
            "timestamp": self.timestamp.isoformat(),
            "additional_info": self.additional_info,
        }

    def to_user_message(self) -> str:
        """
        Format as user-friendly message (no stack traces).

        Returns:
            Brief error message suitable for console display
        """
        parts = [f"[{self.context.value.upper()}]"]

        if self.filename:
            parts.append(self.filename)

        parts.append(self.message)

        if self.exception and str(self.exception) != self.message:
            parts.append(f"({str(self.exception)})")

        return " ".join(parts)

    def to_log_message(self) -> str:
        """
        Format as detailed log message (with exception info).

        Returns:
            Detailed error message suitable for log files
        """
        parts = [
            f"Context: {self.context.value}",
            f"Severity: {self.severity.value if self.severity else 'unknown'}",
            f"Message: {self.message}",
        ]

        if self.filepath:
            parts.append(f"File: {self.filepath}")

        if self.exception:
            parts.append(f"Exception: {self.exception_type}: {str(self.exception)}")

        if self.additional_info:
            parts.append(f"Additional Info: {self.additional_info}")

        if not self.recoverable:
            parts.append("Recoverable: NO")

        return " | ".join(parts)


class ErrorTracker:
    """
    Track and aggregate errors during batch processing.

    Provides centralized error collection, categorization, and reporting
    for batch font processing operations.
    """

    def __init__(self):
        """Initialize empty error tracker."""
        self.errors: List[ErrorInfo] = []
        self._errors_by_context: Dict[ErrorContext, List[ErrorInfo]] = {}
        self._errors_by_file: Dict[str, List[ErrorInfo]] = {}

    def add_error(self, error: ErrorInfo) -> None:
        """
        Add an error to the tracker.

        Args:
            error: ErrorInfo instance to track
        """
        self.errors.append(error)

        # Index by context
        if error.context not in self._errors_by_context:
            self._errors_by_context[error.context] = []
        self._errors_by_context[error.context].append(error)

        # Index by file
        if error.filepath:
            if error.filepath not in self._errors_by_file:
                self._errors_by_file[error.filepath] = []
            self._errors_by_file[error.filepath].append(error)

        # Log based on severity
        log_message = error.to_log_message()
        if error.severity == ErrorSeverity.CRITICAL:
            logger.error(log_message)
            if error.stack_trace:
                logger.debug(f"Stack trace:\n{error.stack_trace}")
        elif error.severity == ErrorSeverity.ERROR:
            logger.error(log_message)
        elif error.severity == ErrorSeverity.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)

    def add_from_exception(
        self,
        context: ErrorContext,
        exception: Exception,
        filepath: Optional[str] = None,
        message: Optional[str] = None,
        **kwargs,
    ) -> ErrorInfo:
        """
        Create and add error from exception.

        Args:
            context: Error context
            exception: The exception that occurred
            filepath: Optional file path
            message: Optional custom message
            **kwargs: Additional ErrorInfo fields

        Returns:
            The created ErrorInfo instance
        """
        error = ErrorInfo.from_exception(
            context, exception, filepath, message, **kwargs
        )
        self.add_error(error)
        return error

    def get_summary(self) -> Dict[str, Any]:
        """
        Get error summary statistics.

        Returns:
            Dictionary with error counts and breakdowns
        """
        return {
            "total_errors": len(self.errors),
            "recoverable_errors": sum(1 for e in self.errors if e.recoverable),
            "non_recoverable_errors": sum(1 for e in self.errors if not e.recoverable),
            "by_context": {
                ctx.value: len(errs) for ctx, errs in self._errors_by_context.items()
            },
            "by_severity": {
                sev.value: sum(1 for e in self.errors if e.severity == sev)
                for sev in ErrorSeverity
            },
            "files_with_errors": len(self._errors_by_file),
        }

    def get_errors_for_file(self, filepath: str) -> List[ErrorInfo]:
        """
        Get all errors for a specific file.

        Args:
            filepath: File path to query

        Returns:
            List of ErrorInfo instances for that file
        """
        return self._errors_by_file.get(filepath, [])

    def get_errors_by_context(self, context: ErrorContext) -> List[ErrorInfo]:
        """
        Get all errors for a specific context.

        Args:
            context: Context to query

        Returns:
            List of ErrorInfo instances for that context
        """
        return self._errors_by_context.get(context, [])

    def has_critical_errors(self) -> bool:
        """Check if any critical errors were encountered."""
        return any(e.severity == ErrorSeverity.CRITICAL for e in self.errors)

    def has_non_recoverable_errors(self) -> bool:
        """Check if any non-recoverable errors were encountered."""
        return any(not e.recoverable for e in self.errors)

    def print_summary(self, console=None) -> None:
        """
        Print error summary to console.

        Args:
            console: Optional Rich console instance
        """
        from FontCore.core_console_styles import (
            emit,
            fmt_header,
            fmt_count,
            ERROR_LABEL,
        )

        summary = self.get_summary()

        if summary["total_errors"] == 0:
            return

        emit("", console=console)
        fmt_header("ERROR SUMMARY", console=console)
        emit("", console=console)

        emit(
            f"{ERROR_LABEL} Total errors: {fmt_count(summary['total_errors'])}",
            console=console,
        )

        if summary["non_recoverable_errors"] > 0:
            emit(
                f"{ERROR_LABEL} Non-recoverable: {fmt_count(summary['non_recoverable_errors'])}",
                console=console,
            )

        emit("", console=console)
        emit("  Errors by context:", console=console)
        for context, count in sorted(summary["by_context"].items()):
            emit(f"    {context:20} : {fmt_count(count)}", console=console)

        emit("", console=console)
        emit("  Errors by severity:", console=console)
        for severity, count in sorted(summary["by_severity"].items()):
            if count > 0:
                emit(f"    {severity:20} : {fmt_count(count)}", console=console)

    def clear(self) -> None:
        """Clear all tracked errors."""
        self.errors.clear()
        self._errors_by_context.clear()
        self._errors_by_file.clear()


# Global error tracker for convenience
_global_tracker = ErrorTracker()


def get_global_tracker() -> ErrorTracker:
    """Get the global error tracker instance."""
    return _global_tracker


__all__ = [
    "ErrorContext",
    "ErrorSeverity",
    "ErrorInfo",
    "ErrorTracker",
    "get_global_tracker",
]
