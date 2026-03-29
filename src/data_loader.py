import pandas as pd


class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.file_path)
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Strip whitespace first
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()

        # Replace ? with missing values
        df.replace("?", pd.NA, inplace=True)

        # Drop duplicates
        df.drop_duplicates(inplace=True)

        return df

    def get_features_and_target(self, df: pd.DataFrame, target_column: str = "income"):
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y
