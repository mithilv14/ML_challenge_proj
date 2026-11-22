THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)

# Starting values from the request; these are experiment settings, not claimed optima.
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

# V2 used positive_weight=2.0 during SGD updates; pass the same class emphasis to XGBoost.
POSITIVE_SAMPLE_WEIGHT = 2.0

# Reference values from the completed V2 experiment. The script prefers its saved report/cache.
V2_REFERENCE = {
    "overall": {
        "macro_f0_5": 0.77404035659903,
        "precision_micro": 0.8954493087557603,
        "recall_micro": 0.6687939194034132,
        "blocking_recall": 0.8250394378316364,
        "candidate_pairs": 127397,
        "predicted_pairs": 10416,
    },
    "countries": {
        "india": {
            "macro_f0_5": 0.7405186532079511,
            "precision_micro": 0.84625468164794,
            "recall_micro": 0.6522806004618937,
            "blocking_recall": 0.8276558891454965,
            "candidate_pairs": 75559,
            "predicted_pairs": 5340,
            "best_threshold": 0.75,
        },
        "us": {
            "macro_f0_5": 0.807562059990109,
            "precision_micro": 0.9472025216706068,
            "recall_micro": 0.6850954687945283,
            "blocking_recall": 0.8224565403248789,
            "candidate_pairs": 51838,
            "predicted_pairs": 5076,
            "best_threshold": 0.75,
        },
    },
}
