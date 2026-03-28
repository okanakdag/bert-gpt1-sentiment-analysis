# Evaluation helpers for both BERT and GPT experiments.


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_classification_metrics(true_labels, predicted_labels):
    return {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels, zero_division=0),
        "recall": recall_score(true_labels, predicted_labels, zero_division=0),
        "f1": f1_score(true_labels, predicted_labels, zero_division=0),
    }


def save_metrics(metrics, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                file.write(f"{metric_name}: {metric_value:.4f}\n")
            else:
                file.write(f"{metric_name}: {metric_value}\n")
