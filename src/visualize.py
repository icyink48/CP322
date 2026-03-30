import os
import matplotlib.pyplot as plt


class Visualizer:
    def save_target_distribution(self, y, output_path="results/plots/target_distribution.png"):
        """
        Save a bar chart of the target class distribution.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        y.value_counts().plot(kind="bar")
        plt.title("Target Class Distribution")
        plt.xlabel("Income")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def save_model_accuracy_plot(self, results_df, output_path="results/plots/model_accuracy_comparison.png"):
        """
        Save a bar chart comparing test accuracy across models.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        sorted_df = results_df.sort_values(by="test_accuracy", ascending=False)

        plt.figure(figsize=(10, 6))
        plt.bar(sorted_df["model"], sorted_df["test_accuracy"])
        plt.title("Model Test Accuracy Comparison")
        plt.xlabel("Model")
        plt.ylabel("Test Accuracy")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
