import polars as pl

import numpy as np


class RiskModel:

    """Creating volatility target column for allocation step"""

    def __init__(self, df_ref: pl.DataFrame, vol_window: int):
        self.df_ref = self._compute_volatility(df_ref=df_ref, window=vol_window)

    def _compute_volatility(self, df_ref: pl.DataFrame, window: int) -> pl.DataFrame:
        return (
            df_ref.with_columns(
                [
                    (
                        pl.col("px_last")
                        .pct_change(1)
                        .forward_fill()
                        .rolling_std(window)
                        * np.sqrt(window)
                    )
                    .over("asset_name")
                    .alias("annualized_vol")
                ]
            )
            .select(["date", "asset_name", "annualized_vol"])
            .rename({"asset_name": "asset_id"})
        )

    def compute_simple_vol_target(
        self, df_prediction: pl.DataFrame, vol_target: float
    ) -> pl.DataFrame:
        return df_prediction.join(
            self.df_ref, on=["date", "asset_id"], how="left"
        ).with_columns(
            pl.min(
                1,
                vol_target / pl.col("annualized_vol"),
            ).alias("vol_target")
        )
