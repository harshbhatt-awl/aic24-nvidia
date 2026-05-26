class ConfigError(ValueError):
    """Raised when configuration is invalid or missing fields."""


class ValidationError(Exception):
    """Raised when a validation gate (frame count, det count, etc.) fails."""


class StageError(RuntimeError):
    """Raised when a stage subprocess fails or its outputs are malformed."""

    def __init__(self, stage: str, returncode: int, log_path: str):
        self.stage = stage
        self.returncode = returncode
        self.log_path = log_path
        super().__init__(
            f"stage '{stage}' failed (exit {returncode}); see {log_path}"
        )
