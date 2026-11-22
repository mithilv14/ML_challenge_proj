from dataclasses import dataclass


def f_beta(precision, recall, beta=0.5):
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta2 = beta * beta
    denom = beta2 * precision + recall
    return (1.0 + beta2) * precision * recall / denom if denom else 0.0


@dataclass
class _ThresholdCounts:
    macro_sum: float = 0.0
    true_positive: int = 0
    predicted: int = 0
    true: int = 0
    singleton_correct: int = 0


class ValidationAccumulator:
    def __init__(self, thresholds):
        self.thresholds = tuple(thresholds)
        self.counts = {threshold: _ThresholdCounts() for threshold in self.thresholds}
        self.entities = 0
        self.singleton_total = 0
        self.blocked_true = 0
        self.true_total = 0
        self.candidate_total = 0
        self.candidate_max = 0

    def add(self, truth_ids, candidate_ids, scores):
        truth = set(truth_ids)
        candidates = set(candidate_ids)
        self.entities += 1
        self.candidate_total += len(candidate_ids)
        self.candidate_max = max(self.candidate_max, len(candidate_ids))
        self.blocked_true += len(truth & candidates)
        self.true_total += len(truth)
        if not truth:
            self.singleton_total += 1

        for threshold in self.thresholds:
            predicted = {entity_id for entity_id, score in zip(candidate_ids, scores) if score >= threshold}
            tp = len(predicted & truth)
            if not truth:
                score = 1.0 if not predicted else 0.0
                if not predicted:
                    self.counts[threshold].singleton_correct += 1
            else:
                precision = tp / len(predicted) if predicted else 0.0
                recall = tp / len(truth)
                score = f_beta(precision, recall)
            stats = self.counts[threshold]
            stats.macro_sum += score
            stats.true_positive += tp
            stats.predicted += len(predicted)
            stats.true += len(truth)

    def report(self, target_count=0):
        by_threshold = {}
        for threshold in self.thresholds:
            stats = self.counts[threshold]
            precision = stats.true_positive / stats.predicted if stats.predicted else 0.0
            recall = stats.true_positive / stats.true if stats.true else 0.0
            by_threshold[str(threshold)] = {
                "precision_micro": precision,
                "recall_micro": recall,
                "macro_f0_5": stats.macro_sum / self.entities if self.entities else 0.0,
                "singleton_accuracy": stats.singleton_correct / self.singleton_total if self.singleton_total else 0.0,
                "predicted_pairs": stats.predicted,
            }
        best_threshold = max(
            self.thresholds,
            key=lambda value: (self.counts[value].macro_sum / self.entities if self.entities else 0.0, value),
        )
        chosen = by_threshold[str(best_threshold)]
        return {
            "entities": self.entities,
            "singleton_entities": self.singleton_total,
            "candidate_pairs": self.candidate_total,
            "average_candidates_per_entity": self.candidate_total / self.entities if self.entities else 0.0,
            "maximum_candidates_per_entity": self.candidate_max,
            "blocking_recall": self.blocked_true / self.true_total if self.true_total else 0.0,
            "blocked_true_matches": self.blocked_true,
            "true_matches": self.true_total,
            "candidate_reduction_ratio": 1.0 - self.candidate_total / (self.entities * target_count) if self.entities and target_count else 0.0,
            "best_threshold": best_threshold,
            "precision_micro": chosen["precision_micro"],
            "recall_micro": chosen["recall_micro"],
            "macro_f0_5": chosen["macro_f0_5"],
            "singleton_accuracy": chosen["singleton_accuracy"],
            "true_positive": self.counts[best_threshold].true_positive,
            "predicted_pairs": self.counts[best_threshold].predicted,
            "singleton_correct": self.counts[best_threshold].singleton_correct,
            "thresholds": by_threshold,
        }
