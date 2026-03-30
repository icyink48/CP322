import pandas as pd


class FeatureBuilder:
    def add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for the Adult Income dataset.
        """
        df = df.copy()

        # Net capital feature
        if "capital_gain" in df.columns and "capital_loss" in df.columns:
            df["net_capital"] = df["capital_gain"] - df["capital_loss"]

        # Hours category feature
        if "hours_per_week" in df.columns:
            df["hours_per_week_group"] = pd.cut(
                df["hours_per_week"],
                bins=[0, 25, 40, 60, 100],
                labels=["part_time", "full_time", "over_time", "extreme_time"],
                include_lowest=True
            )

        # Age group feature
        if "age" in df.columns:
            df["age_group"] = pd.cut(
                df["age"],
                bins=[0, 25, 45, 65, 100],
                labels=["young", "adult", "middle_aged", "senior"],
                include_lowest=True
            )

        return df

    def summarize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a simple summary of columns, data types, and missing values.
        """
        summary = pd.DataFrame({
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "missing_values": df.isna().sum().values,
            "unique_values": df.nunique(dropna=False).values
        })
        return summary
