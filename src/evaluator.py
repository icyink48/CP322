import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


class ModelEvaluator:
    def evaluate_models(self, trained_models, X_train, X_test, y_train, y_test):
        """
        Evaluate all trained models on both training and test data.
        """
        results = []
        confusion_matrices = {}

        for name, model in trained_models.items():
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            result = {
                "model": name,
                "train_accuracy": accuracy_score(y_train, y_train_pred),
                "test_accuracy": accuracy_score(y_test, y_test_pred),
                "precision_weighted": precision_score(y_test, y_test_pred, average="weighted", zero_division=0),
                "recall_weighted": recall_score(y_test, y_test_pred, average="weighted", zero_division=0),
                "f1_weighted": f1_score(y_test, y_test_pred, average="weighted", zero_division=0),
            }
            results.append(result)

            cm = confusion_matrix(y_test, y_test_pred)
            confusion_matrices[name] = pd.DataFrame(
                cm,
                index=["Actual <=50K", "Actual >50K"],
                columns=["Predicted <=50K", "Predicted >50K"]
            )

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by="test_accuracy", ascending=False).reset_index(drop=True)

        return results_df, confusion_matrices

    def save_results(self, results_df, output_path="results/metrics/model_results.csv"):
        """
        Save evaluation results to CSV.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)

    def print_results(self, results_df):
        """
        Print a clean summary of results.
        """
        print("\nModel Comparison Results:")
        print(results_df.to_string(index=False))
