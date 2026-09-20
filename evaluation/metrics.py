def _entity_set(entities):
    return {
        (int(entity["start"]), int(entity["end"]), str(entity["type"]).upper())
        for entity in entities
        if {"start", "end", "type"} <= set(entity)
    }


def calculate_metrics(predictions):
    """Strict exact-span + exact-type entity evaluation."""
    tp = fp = fn = valid_json_count = 0

    for item in predictions:
        gold = _entity_set(item["gold_entities"])

        if item.get("valid_json", False):
            valid_json_count += 1
            predicted = _entity_set(item["predicted_entities"])
        else:
            predicted = set()

        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    beta = 2.0
    f2 = (
        (1 + beta**2) * precision * recall
        / ((beta**2 * precision) + recall)
        if ((beta**2 * precision) + recall)
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "pii_leakage_rate": fn / (tp + fn) if (tp + fn) else 0.0,
        "over_redaction_rate": fp / (tp + fp) if (tp + fp) else 0.0,
        "json_validity": (
            valid_json_count / len(predictions) if predictions else 0.0
        ),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }
