from data_loader import DataLoader
from preprocessor import DataPreprocessor


def main():
    loader = DataLoader("data/raw/adult.csv")
    preprocessor = DataPreprocessor(test_size=0.2, random_state=42)

    # Load and clean data
    df = loader.load_data()
    df = loader.clean_data(df)
    X, y = loader.get_features_and_target(df, target_column="income")

    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

    # Build preprocessing pipeline
    transformer, numeric_features, categorical_features = preprocessor.build_preprocessor(X_train)

    # Fit on training data and transform both sets
    X_train_processed = transformer.fit_transform(X_train)
    X_test_processed = transformer.transform(X_test)

    print("Dataset loaded and split successfully.")
    print(f"Full dataset shape: {df.shape}")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    print(f"Processed training shape: {X_train_processed.shape}")
    print(f"Processed testing shape: {X_test_processed.shape}")

    print("\nNumeric features:")
    print(numeric_features)

    print("\nCategorical features:")
    print(categorical_features)

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())


if __name__ == "__main__":
    main()
