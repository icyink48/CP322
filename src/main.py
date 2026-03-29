from data_loader import DataLoader
from preprocessor import DataPreprocessor
from train_models import ModelTrainer
from evaluator import ModelEvaluator


def main():
    # Step 1: Load and clean data
    loader = DataLoader("data/raw/adult.csv")
    df = loader.load_data()
    df = loader.clean_data(df)
    X, y = loader.get_features_and_target(df, target_column="income")

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    # Step 2: Split data
    preprocessor_builder = DataPreprocessor(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = preprocessor_builder.split_data(X, y)

    print("Training set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)

    # Step 3: Build preprocessor
    preprocessor, numeric_features, categorical_features = preprocessor_builder.build_preprocessor(X_train)

    print("\nNumeric features:")
    print(numeric_features)

    print("\nCategorical features:")
    print(categorical_features)

    # Step 4: Train base models
    trainer = ModelTrainer(random_state=42)
    models = trainer.get_models(preprocessor)
    trained_models = trainer.train_models(models, X_train, y_train)

    # Step 5: Tune Random Forest
    print("\nTuning RandomForest...")
    tuned_rf = trainer.get_tuned_random_forest(preprocessor)
    tuned_rf.fit(X_train, y_train)
    trained_models["RandomForest_Tuned"] = tuned_rf

    print("Best RandomForest parameters:")
    print(tuned_rf.best_params_)

    # Step 6: Evaluate models
    evaluator = ModelEvaluator()
    results_df, confusion_matrices, classification_reports = evaluator.evaluate_models(
        trained_models, X_train, X_test, y_train, y_test
    )

    evaluator.print_results(results_df)
    evaluator.save_results(results_df)
    evaluator.save_classification_reports(classification_reports)
    evaluator.save_confusion_matrix_plots(confusion_matrices)

    # Step 7: Print best model
    best_model_name = results_df.iloc[0]["model"]
    print(f"\nBest model based on test accuracy: {best_model_name}")


if __name__ == "__main__":
    main()
