import pandas as pd


class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from a CSV file.
        """
        return pd.read_csv(self.file_path)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by:
        - stripping whitespace from string columns
        - converting '?' values to missing values
        - dropping duplicate rows
        """
        df = df.copy()

        # Strip whitespace from all object columns
        object_cols = df.select_dtypes(include=["object"]).columns
        for col in object_cols:
            df[col] = df[col].astype(str).str.strip()

        # Replace '?' and empty strings with missing values
        df.replace(r"^\?$", pd.NA, regex=True, inplace=True)
        df.replace(r"^\s*$", pd.NA, regex=True, inplace=True)

        # Drop duplicate rows
        df.drop_duplicates(inplace=True)

        return df

    def get_features_and_target(
        self,
        df: pd.DataFrame,
        target_column: str = "income"
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Split the dataframe into features (X) and target (y).
        """
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )

        X = df.drop(columns=[target_column])
        y = df[target_column]

        return X, y
