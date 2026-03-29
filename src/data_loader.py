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
