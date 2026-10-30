from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from data_loader import load_dataset as load_original
from data_loader_b_png import load_dataset as load_srgan

tf.keras.utils.set_random_seed(42)

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

OUTPUT_DIR = Path("results/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def evaluate(name, checkpoint, csv_path, loader):
    print(f"\nEvaluating {name}...", flush=True)

    df = pd.read_csv(csv_path)
    y_true = df["label"].to_numpy(dtype=np.int32)

    ds = loader(
        csv_path,
        batch_size=32,
        training=False
    )

    model = tf.keras.models.load_model(
        checkpoint,
        compile=False
    )

    probabilities = model.predict(ds, verbose=1).reshape(-1)
    predictions = (probabilities >= 0.5).astype(np.int32)

    assert len(probabilities) == len(y_true), (
        f"Prediction count mismatch for {name}"
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions)),
        "recall": float(recall_score(y_true, predictions)),
        "f1_score": float(f1_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities))
    }

    cm = confusion_matrix(y_true, predictions, labels=[0, 1])

    pd.DataFrame({
        "filepath": df["filepath"],
        "true_label": y_true,
        "predicted_label": predictions,
        "probability_dog": probabilities
    }).to_csv(
        OUTPUT_DIR / f"{name}_predictions.csv",
        index=False
    )

    pd.DataFrame(
        cm,
        index=["actual_cat", "actual_dog"],
        columns=["predicted_cat", "predicted_dog"]
    ).to_csv(OUTPUT_DIR / f"{name}_confusion_matrix.csv")

    print(f"\n{name} metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(classification_report(
        y_true, predictions,
        target_names=["cat", "dog"]
    ))

    del model
    tf.keras.backend.clear_session()

    return metrics

results = {}

results["model_a"] = evaluate(
    "model_a",
    "checkpoints/model_a/best_finetuned.keras",
    "data/splits/test.csv",
    load_original
)

results["model_b"] = evaluate(
    "model_b",
    "checkpoints/model_b/best_finetuned.keras",
    "data/splits/test_srgan.csv",
    load_srgan
)

with open(OUTPUT_DIR / "comparison.json", "w") as f:
    json.dump(results, f, indent=2)

comparison = pd.DataFrame(results).T
comparison.to_csv(OUTPUT_DIR / "comparison.csv")

print("\nFINAL COMPARISON")
print(comparison.to_string())

print("\nResults saved to:", OUTPUT_DIR)
