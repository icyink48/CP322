import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    ConfusionMatrixDisplay
)


class ModelEvaluator:
    def evaluate_models(self, trained_models, X_train, X_test, y_train, y_test):
        """
        Evaluate all trained models on both training and test data.
        """
        results = []
        confusion_matrices = {}
        classification_reports = {}

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

            # ROC-AUC where possible
            roc_auc = None
            try:
                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(X_test)[:, 1]
                    y_true_binary = (y_test == ">50K").astype(int)
                    roc_auc = roc_auc_score(y_true_binary, y_score)
                elif hasattr(model, "decision_function"):
                    y_score = model.decision_function(X_test)
                    y_true_binary = (y_test == ">50K").astype(int)
                    roc_auc = roc_auc_score(y_true_binary, y_score)
            except Exception:
                roc_auc = None

            result["roc_auc"] = roc_auc
            results.append(result)

            cm = confusion_matrix(y_test, y_test_pred)
            confusion_matrices[name] = cm

            report = classification_report(y_test, y_test_pred, zero_division=0)
            classification_reports[name] = report

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by="test_accuracy", ascending=False).reset_index(drop=True)

        return results_df, confusion_matrices, classification_reports

    def save_results(self, results_df, output_path="results/metrics/model_results.csv"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)

    def save_classification_reports(self, classification_reports, output_dir="results/reports"):
        os.makedirs(output_dir, exist_ok=True)

        for model_name, report in classification_reports.items():
            file_path = os.path.join(output_dir, f"{model_name}_classification_report.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(report)

    def save_confusion_matrix_plots(self, confusion_matrices, output_dir="results/plots"):
        os.makedirs(output_dir, exist_ok=True)

        for model_name, cm in confusion_matrices.items():
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["<=50K", ">50K"]
            )
            disp.plot()
            plt.title(f"Confusion Matrix - {model_name}")
            plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"))
            plt.close()

    def print_results(self, results_df):
        print("\nModel Comparison Results:")
        print(results_df.to_string(index=False))
