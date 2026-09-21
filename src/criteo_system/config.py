from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DatasetSourceConfig(BaseModel):
    name: str


class SamplingConfig(BaseModel):
    strategy: Literal["uniform_reservoir"]
    target_rows: int = Field(gt=0)
    random_seed: int = Field(ge=0)
    preserve_source_order: bool


class FormatConfig(BaseModel):
    output: Literal["parquet"]


class SplitConfig(BaseModel):
    strategy: Literal["chronological"]
    train_fraction: float = Field(gt=0, lt=1)
    validation_fraction: float = Field(gt=0, lt=1)
    test_fraction: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def validate_fractions(self) -> "SplitConfig":
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Split fractions must sum to 1.0")
        return self


class DatasetConfig(BaseModel):
    dataset: DatasetSourceConfig
    sampling: SamplingConfig
    format: FormatConfig
    split: SplitConfig
