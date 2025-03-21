from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class CustomerDatasetSize(Enum):
    LARGE = "LARGE"
    MEDIUM = "MEDIUM"
    SMALL = "SMALL"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    SOLAR_GEN_DIR: Path = Path(__file__).parent

    # Hopsworks
    HOPSWORKS_API_KEY: SecretStr | None = None

    # Training
    TWO_TOWER_MODEL_EMBEDDING_SIZE: int = 16
    TWO_TOWER_MODEL_BATCH_SIZE: int = 2048
    TWO_TOWER_NUM_EPOCHS: int = 5
    TWO_TOWER_WEIGHT_DECAY: float = 0.001
    TWO_TOWER_LEARNING_RATE: float = 0.001 #0.01
    TWO_TOWER_DATASET_VALIDATON_SPLIT_SIZE: float = 0.1
    TWO_TOWER_DATASET_TEST_SPLIT_SIZE: float = 0.1

    RANKING_DATASET_VALIDATON_SPLIT_SIZE: float = 0.1
    RANKING_LEARNING_RATE: float = 0.2
    RANKING_ITERATIONS: int = 100
    RANKING_SCALE_POS_WEIGHT: int = 10
    RANKING_EARLY_STOPPING_ROUNDS: int = 5



settings = Settings()