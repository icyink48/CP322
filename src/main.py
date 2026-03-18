from data_loader import DataLoader


def main():
    loader = DataLoader("data/raw/adult.csv")

    df = loader.load_data()
    df = loader.clean_data(df)
    X, y = loader.get_features_and_target(df, target_column="income")

    print("Dataset loaded successfully.")
    print("Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nTarget distribution:")
    print(y.value_counts())


if __name__ == "__main__":
    main()
