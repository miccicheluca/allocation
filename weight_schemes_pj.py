import polars as pl
from typing import Union, List, Optional


def weight_normalization(
    df: pl.DataFrame, over: Union[str, List], weight_column_name: str
) -> pl.DataFrame:
    """Normalize weight so the sum is 1."""
    return df.with_columns(
        (
            (pl.col(weight_column_name) / (pl.col(weight_column_name).sum().over(over)))
        ).alias(weight_column_name)
    )


def rank_weighted(
    df: pl.DataFrame,
    prediction_column: str,
    over: Union[str, List],
    power: float = 1.0,
    normalize: bool = True,
    weight_column_name: Optional[str] = "weight",
) -> pl.DataFrame:
    """Rank weighted and Equal weighted in case of NaN values."""
    df = (
        df.with_columns(
            pl.col(prediction_column).rank().over(over).alias(weight_column_name)
        )
        .with_columns((pl.col(weight_column_name) ** power).alias(weight_column_name))
        .with_columns(
            pl.col(weight_column_name)
            .fill_nan(None)
            .fill_null(pl.col(prediction_column).count().over(over) / 2)
            .alias(weight_column_name)
        )
    )
    if normalize:
        df = df.pipe(weight_normalization, over, weight_column_name)
    return df


def volatility_weighted(
    df: pl.DataFrame,
    volatility_column: str,
    over: Union[str, List],
    volatility_target: float = 0.20,
    min_leverage: float = 0,
    max_leverage: float = 6,
    normalize: bool = False,
    max_portfolio_buget: Optional[float] = 1.0,
    weight_column_name: Optional[str] = "weight",
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
            .with_columns((pl.col("weight_ew") * pl.col("weight")).alias("weight"))
            .drop("weight_ew")
            .pipe(check_portfolio_exposure, "weight", over, max_portfolio_buget)
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


def rank_volatility_weighted(
    df: pl.DataFrame,
    prediction_column: str,
    volatility_column: str,
    over: Union[str, List],
    power: float = 1.0,
    volatility_target: float = 0.20,
    min_leverage: float = 0.0,
    max_leverage: float = 3,
    normalize: bool = False,
    weight_column_name: str = "weight",
) -> pl.DataFrame:
    """Rank weighted in combination with vol weighted and Equal weighted in case of NaN values."""
    df = (
        df.pipe(rank_weighted, prediction_column, over, power, True, "weight_rank")
        .pipe(
            volatility_weighted,
            volatility_column=volatility_column,
            volatility_target=volatility_target,
            normalize=True,
            over=over,
            weight_column_name="weight_vol",
        )
        .with_columns(
            (pl.col("weight_rank") * pl.col("weight_vol")).alias(weight_column_name)
        )
    ).drop(["weight_rank", "weight_vol"])
    if normalize:
        df = df.pipe(weight_normalization, over, weight_column_name)
    return df


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

