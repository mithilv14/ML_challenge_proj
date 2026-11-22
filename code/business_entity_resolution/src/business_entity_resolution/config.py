from dataclasses import dataclass
from pathlib import Path


def default_project_root() -> Path:
    # config.py -> package -> src -> code/business_entity_resolution -> code -> project root
    return Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path = default_project_root()
    max_train_per_country: int = 10000
    max_validation_per_country: int = 2000
    train_hash_range: tuple = (10, 70)
    validation_hash_range: tuple = (0, 10)
    hash_buckets: int = 1000
    negative_samples_per_entity: int = 20
    negative_to_positive_ratio: int = 5
    seed: int = 1729
    max_token_posting: int = 5000
    max_candidates_per_entity: int = 50
    thresholds: tuple = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)

    @property
    def data_dir(self) -> Path:
        return self.project_root / "dataset"

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "code" / "business_entity_resolution" / "artifacts"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "output"
