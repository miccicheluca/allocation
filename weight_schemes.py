import polars as pl
from tqdm import tqdm
from typing import Union, List, Optional


def rank_weighted(
    df: pl.DataFrame, prediction_column: str, date_column: str, power: float = 1.0
) -> pl.DataFrame:
    """Rank weighted and Equal weighted in case of NaN values."""
    return (
        df.with_columns(
            pl.col(prediction_column).rank().over(date_column).alias("portfolio_rank")
        )
        .with_columns((pl.col("portfolio_rank") ** power).alias("relative_weight"))
        .with_columns(
            pl.col("relative_weight")
            .fill_nan(None)
            .fill_null(pl.lit(1))
            .alias("relative_weight")
        )
    )


def vol_weighted(
    df: pl.DataFrame, volatility_column: str, vol_target: float = 0.20
) -> pl.DataFrame:
    """Vol weighted and Equal weighted in case of NaN values."""
    return df.with_columns(
        (vol_target / pl.col(volatility_column)).alias("relative_weight")
    ).with_columns(
        pl.col("relative_weight")
        .fill_nan(None)
        .fill_null(pl.lit(1))
        .alias("relative_weight")
    )


def rank_vol_weighted(
    df: pl.DataFrame,
    prediction_column: str,
    date_column: str,
    volatility_column: str,
    power: float = 1.0,
    vol_target: float = 0.20,
) -> pl.DataFrame:
    """Rank weighted in combination with vol weighted and Equal weighted in case of NaN values."""
    return (
        df.with_columns(
            pl.col(prediction_column)
            .rank(method="ordinal")
            .over(date_column)
            .alias("portfolio_rank")
        )
        .with_columns((pl.col("portfolio_rank") ** power).alias("relative_weight"))
        .with_columns(
            (pl.col("relative_weight") * vol_target / pl.col(volatility_column)).alias(
                "relative_weight"
            )
        )
        .with_columns(
            pl.col("relative_weight")
            .fill_nan(None)
            .fill_null(pl.lit(1))
            .alias("relative_weight")
        )
    )


def equal_weighted(
    df: pl.DataFrame,
    over: Union[str, List],
    normalize: bool = False,
    weight_column_name: str = "weight",
) -> pl.DataFrame:
    """Equal weighted."""
    df = df.with_columns(pl.lit(1.0).alias(weight_column_name))
    if normalize:
        df = df.pipe(weight_normalization, over, weight_column_name)
    return df


def normalize_weight(
    df: pl.DataFrame, weight_column: str, date_column: str, portfolio_column: str
) -> pl.DataFrame:
    """Normalize weighs for each portfolio each date."""
    return df.with_columns(
        pl.col(weight_column)
        / (pl.col(weight_column).sum().over([date_column, portfolio_column])).alias(
            weight_column
        )
    )


def weight_normalization(
    df: pl.DataFrame, over: Union[str, List], weight_column_name: str
) -> pl.DataFrame:
    """Normalize weight so the sum is 1."""
    return df.with_columns(
        (
            (pl.col(weight_column_name) / (pl.col(weight_column_name).sum().over(over)))
        ).alias(weight_column_name)
    )


def volatility_weighted(
    df: pl.DataFrame,
    volatility_column: str,
    over: Union[str, List],
    volatility_target: float = 0.20,
    min_leverage: float = 0,
    max_leverage: float = 6,
    normalize: bool = False,
    max_portfolio_buget: Optional[float] = 1.0,
    weight_column_name: Optional[str] = "relative_weight",
) -> pl.DataFrame:
    """Vol weighted and Equal weighted in case of NaN values."""
    df = df.with_columns(
        (
            (volatility_target / pl.col(volatility_column)).clip(min_leverage, max_leverage)
        ).alias(weight_column_name)
    ).with_columns(
        pl.col(weight_column_name)
        .fill_nan(None)
        .fill_null(pl.lit(1))
        .alias(weight_column_name)
    )
    if normalize:
        df = df.pipe(weight_normalization, over, weight_column_name)
    else:
        df = (
            df.pipe(equal_weighted, over, True, "weight_ew")
            .with_columns((pl.col("weight_ew") * pl.col(weight_column_name)).alias(weight_column_name))
            .drop("weight_ew")
            .pipe(check_portfolio_exposure, weight_column_name, over, max_portfolio_buget)
        )
    return df


def check_portfolio_exposure(
    df: pl.DataFrame,
    weight_column: str,
    over: Union[str, List],
    constraint_max_risk_buget: float = 1.0,
    date_column: str = "date"
) -> pl.DataFrame:
    """ """
    # 1) current risk budget for each date
    sum_risk_budget = df.groupby(over).agg(
        pl.col(weight_column).sum().alias("risk_budget")
    ).select([date_column, "risk_budget"])
    max_w = df[weight_column].max()
    i = 0
    # 2) max risk budget over all dates
    sum_risk_budget_max = sum_risk_budget["risk_budget"].max()
    # 3) adjust max found risk budget until respect constraint_max_risk_buget
    while sum_risk_budget_max > constraint_max_risk_buget:
        max_w = max_w - (max_w * 0.05)
        df = df.join(sum_risk_budget, how="left", on=date_column).with_columns(
            pl.when(pl.col("risk_budget") > constraint_max_risk_buget)
            .then(pl.col(weight_column).clip_max(max_w))
            .otherwise(pl.col(weight_column))
        ).drop("risk_budget")
        sum_risk_budget = df.groupby(over).agg(
            pl.col(weight_column).sum().alias("risk_budget")
        ).select([date_column, "risk_budget"])
        sum_risk_budget_max = sum_risk_budget["risk_budget"].max()
        i += 1
        if i > 1000:
            raise ValueError(
                "Probably in a endless while loop. Check weight constraints."
            )
    return df


# def adjust_max_weight(
#     df: pl.DataFrame,
#     weight_column: str,
#     portfolio_column: str,
#     date_column: str,
#     max_weight: float = 0.1,
# ) -> pl.DataFrame:
#     """Verify max weight constraint for each of the portfolio for each date and adjust if necessary."""
#     # get max weight
#     max_w = df.select(pl.col(weight_column)).max()[0, 0]
#     # adjust until max weight is equal or below threshold
#     while max_w >= (max_weight + max_weight * 0.001):
#         df = df.with_columns(
#             pl.col(weight_column)
#             .clip_max(max_weight)
#             .over([portfolio_column, date_column])
#             .alias(weight_column)
#         ).pipe(normalize_weight, weight_column, date_column, portfolio_column)
#         max_w = df.select(pl.col(weight_column)).max()[0, 0]
#     return df


def adjust_max_weight(
    df: pl.DataFrame,
    weight_column: str,
    portfolio_column: str,
    date_column: str,
    max_weight: float = 0.1,
) -> pl.DataFrame:
    """Verify max weight constraint for each of the portfolio for each date and adjust if necessary."""
    # get max weight
    dates = df.select(date_column).to_series().unique().sort()
    portfolios = df.select(portfolio_column).to_series().unique().sort()
    cols = df.columns
    out = []
    for d in tqdm(dates):
        for p in portfolios:
            temp_df = df.filter(
                (pl.col(date_column) == d) & (pl.col(portfolio_column) == p)
            )
            max_w = temp_df.select(pl.col(weight_column)).max()[0, 0]
            # adjust until max weight is equal or below threshold
            # i = 1
            while max_w >= (max_weight + max_weight * 0.001):
                temp_df = (
                    temp_df.with_columns(
                        [
                            pl.when(pl.col(weight_column) == max_w)
                            .then(0)
                            .when(pl.col(weight_column) == max_weight)
                            .then(0)
                            .otherwise(
                                (
                                    pl.col(weight_column)
                                    .cast(pl.Int64)
                                    .rank(method="ordinal")
                                )
                            )
                            .alias("w_surplus"),
                        ]
                    )
                    # .with_columns(
                    #     [
                    #         pl.when(pl.col("w_surplus") == 0)
                    #         .then(0)
                    #         .otherwise(
                    #             1
                    #             / (
                    #                 1
                    #                 + (
                    #                     -(1 / (pl.col("w_surplus").max() / 10))
                    #                     * (
                    #                         pl.col("w_surplus")
                    #                         - (pl.col("w_surplus").max() / 2)
                    #                     )
                    #                 ).exp()
                    #             )
                    #         )
                    #         .alias("w_surplus")
                    #     ]
                    # )
                    .with_columns([(pl.col("w_surplus") / pl.col("w_surplus").sum())])
                    .with_columns(
                        [
                            pl.when(pl.col(weight_column) == max_w)
                            .then(max_weight)
                            .when(pl.col(weight_column) == max_w)
                            .then(max_w)
                            .otherwise(
                                pl.col("relative_weight")
                                + pl.col("w_surplus") * (max_w - max_weight)
                            )
                            .alias("relative_weight")
                        ]
                    )
                    .sort("relative_weight")
                )
                max_w = temp_df.select(pl.col(weight_column)).max()[0, 0]
                # i += 1
            out.append(
                temp_df.with_columns(pl.col(weight_column).cast(pl.Float64)).select(
                    cols
                )
            )
    return pl.concat(out)
