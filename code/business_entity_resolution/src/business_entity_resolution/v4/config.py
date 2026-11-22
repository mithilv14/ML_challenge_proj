from dataclasses import dataclass
from pathlib import Path


V3_CANDIDATE_CAP = 50
SUPPORTED_CANDIDATE_CAPS = (75, 100)
THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
POSITIVE_SAMPLE_WEIGHT = 2.0

# Deliberately identical to V3's XGBClassifier setup; V4 owns this copy so it
# does not import, patch, or write beside any V3 module.
XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 1729,
    "n_jobs": -1,
    "tree_method": "hist",
}

# Reference supplied by the user for comparison; the local V3 model/report is not required.
V3_REPORTED_REFERENCE = {
    "precision": 0.9565,
    "recall": 0.7402,
    "blocking_recall": 0.8250,
    "f0_5": 0.8499,
    "f0_5_definition": "User-supplied V3 F0.5; macro/micro form was not specified.",
}


@dataclass(frozen=True)
class V4Config:
    project_root: Path
    max_train_per_country: int = 10000
    max_validation_per_country: int = 2000
    train_hash_range: tuple = (10, 70)
    validation_hash_range: tuple = (0, 10)
    hash_buckets: int = 1000
    negative_samples_per_entity: int = 20
    negative_to_positive_ratio: int = 5
    seed: int = 1729
    max_token_posting: int = 5000
    max_candidates_per_entity: int = 100
    thresholds: tuple = THRESHOLDS

    def __post_init__(self):
        if self.max_candidates_per_entity not in SUPPORTED_CANDIDATE_CAPS:
            raise ValueError(
                "V4 candidate cap must be 75 or 100; "
                f"got {self.max_candidates_per_entity}"
            )

    @property
    def data_dir(self) -> Path:
        return Path(self.project_root) / "dataset"

    @property
    def base_artifacts_dir(self) -> Path:
        return Path(self.project_root) / "code" / "business_entity_resolution" / "artifacts"

    @property
    def artifacts_dir(self) -> Path:
        return self.base_artifacts_dir / "v4" / f"cap_{self.max_candidates_per_entity}"
