import pandas as pd


class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from CSV.
        """
        df = pd.read_csv(self.file_path)
        return df

def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset:
        - replace '?' with NaN
        - strip whitespace from string columns
        - drop duplicate rows
        """
        df = df.copy()

        # Replace ? with missing values
        df.replace("?", pd.NA, inplace=True)

        # Strip whitespace from object/string columns
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()

        # Drop duplicates if any
        df.drop_duplicates(inplace=True)

        return df
