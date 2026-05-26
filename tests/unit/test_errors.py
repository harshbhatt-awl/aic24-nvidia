import pytest
from aic24_nvidia.errors import StageError, ConfigError, ValidationError


def test_stage_error_carries_context():
    err = StageError(stage="detect", returncode=1, log_path="/tmp/log.txt")
    assert err.stage == "detect"
    assert err.returncode == 1
    assert err.log_path == "/tmp/log.txt"
    assert "detect" in str(err)
    assert "/tmp/log.txt" in str(err)


def test_config_error_is_a_value_error():
    with pytest.raises(ConfigError):
        raise ConfigError("missing field")


def test_validation_error_is_distinct():
    assert not issubclass(ValidationError, ConfigError)
    assert not issubclass(ConfigError, ValidationError)
