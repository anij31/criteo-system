import pytest
from pydantic import ValidationError

from criteo_system.config import SplitConfig


def test_split_config_accepts_valid_fractions():
    config = SplitConfig(
        strategy="chronological",
        train_fraction=0.7,
        validation_fraction=0.2,
        test_fraction=0.1,
    )

    assert config.train_fraction == 0.7


def test_split_config_rejects_invalid_sum():
    with pytest.raises(ValidationError) as exc_info:
        SplitConfig(
            strategy="chronological",
            train_fraction=0.5,
            validation_fraction=0.5,
            test_fraction=0.5,
        )

    assert "Split fractions must sum to 1.0" in str(exc_info.value)
