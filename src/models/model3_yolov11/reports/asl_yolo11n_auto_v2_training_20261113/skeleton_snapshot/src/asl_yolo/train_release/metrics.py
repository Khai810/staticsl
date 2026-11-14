def score_predictions(samples: list[dict], predictions: dict, classes: list[str], supported: list[str], threshold: float) -> dict:
    """Chấm positive và background riêng, giữ NO_DETECTION."""
    # TODO: supported macro, weighted, coverage, Wilson, confusion, FP.
    raise NotImplementedError


def is_better(candidate: dict, previous: dict | None) -> bool:
    """Chọn macro F1 rồi Accuracy rồi epoch sớm hơn."""
    # TODO: deterministic tie break.
    raise NotImplementedError
