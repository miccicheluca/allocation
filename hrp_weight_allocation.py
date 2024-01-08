import polars as pl
import datetime as dt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, ward
import matplotlib.pyplot as plt
from typing import List, Optional
import time as tm
from tqdm import tqdm

import top_down as td
import bottom_up as bu


def compute_corr(
    df_ref: pl.DataFrame,
    ptf_assets: pl.Series,
    n_month_delay: int,
    rebal_date: dt.datetime,
) -> pd.DataFrame:
    return (
        df_ref.filter(
            (pl.col("date") >= rebal_date - dt.timedelta(weeks=4 * n_month_delay))
            & (pl.col("date") <= rebal_date)
            & (pl.col("asset_name").is_in(ptf_assets))
        )
        .with_columns(
            pl.col("px_last").pct_change(1).over("asset_name").alias("daily_rets")
        )
        .select(["date", "asset_name", "daily_rets"])
        .sort("date")
        .pivot(index="date", columns="asset_name", values="daily_rets")
        .fill_nan(None)
        .fill_null(strategy="forward")
        .to_pandas()
        .corr()
    )


def pariwise_exp_cov(X: pl.Series, Y: pl.Series, exp_half_life: float) -> float:
    covariation = (X - X.mean()) * (Y - Y.mean())
    # return covariation.ewm(halflife=exp_half_life, ignore_na=True).mean().iloc[-1]
    return covariation.ewm_mean(alpha=exp_half_life)[-1]


def compute_rets(
    df_ref: pl.DataFrame,
    rebal_date: dt.datetime,
    n_month_delay: int,
    ptf_assets: pl.Series,
    rolling_window: int,
) -> pl.DataFrame:
    return (
        df_ref.filter(
            (pl.col("date") >= rebal_date - dt.timedelta(weeks=4 * n_month_delay))
            & (pl.col("date") <= rebal_date)
            & (pl.col("asset_name").is_in(ptf_assets))
        )
        .with_columns(
            [
                (
                    pl.col("px_last")
                    .pct_change(1)
                    .over("asset_name")
                    .alias("daily_rets")
                ),
                (
                    pl.col("px_last")
                    .pct_change(1)
                    .rolling_mean(rolling_window)
                    .over("asset_name")
                    .alias("rolling_mean_daily_rets")
                ),
                (
                    (1 + pl.col("px_last").pct_change(1))
                    .log()
                    .over("asset_name")
                    .alias("log_daily_rets")
                ),
                (
                    (1 + pl.col("px_last").pct_change(1))
                    .log()
                    .rolling_mean(rolling_window)
                    .over("asset_name")
                    .alias("rolling_mean_log_daily_rets")
                ),
            ]
        )
        .sort("date")
    )


def _set_up_for_cov_computation(
    df: pl.DataFrame, log_rets: bool, rolling_window: int
) -> pl.DataFrame:
    res = pl.DataFrame
    if (log_rets is False) & (rolling_window <= 1):
        res = (
            df.select(["date", "asset_name", "daily_rets"])
            .pivot(index="date", columns="asset_name", values="daily_rets")
            .fill_nan(None)
            .fill_null(strategy="forward")
        )

    elif (log_rets is False) & (rolling_window > 1):
        res = (
            df.select(["date", "asset_name", "rolling_mean_daily_rets"])
            .pivot(index="date", columns="asset_name", values="rolling_mean_daily_rets")
            .fill_nan(None)
            .fill_null(strategy="forward")
        )

    elif (log_rets is True) & (rolling_window <= 1):
        res = (
            df.select(["date", "asset_name", "log_daily_rets"])
            .pivot(index="date", columns="asset_name", values="log_daily_rets")
            .fill_nan(None)
            .fill_null(strategy="forward")
        )

    elif (log_rets is True) & (rolling_window > 1):
        res = (
            df.select(["date", "asset_name", "rolling_mean_log_daily_rets"])
            .pivot(
                index="date", columns="asset_name", values="rolling_mean_log_daily_rets"
            )
            .fill_nan(None)
            .fill_null(strategy="forward")
        )
    return res


def _compute_cov_matrix(
    df: pl.DataFrame,
    exp_weighted: bool,
    exp_half_life: float,
    ptf_assets: Optional[pl.Series] = None,
) -> pd.DataFrame:
    df_res = pd.DataFrame
    if exp_weighted:
        # Loop over matrix, filling entries with the pairwise exp cov
        nb_assets = ptf_assets.shape[0]
        res = np.zeros((nb_assets, nb_assets))
        assets = df.columns[1:]
        for i in range(nb_assets):
            for j in range(i, nb_assets):
                res[i, j] = res[j, i] = pariwise_exp_cov(
                    X=df.select([assets[i]]).to_series(),
                    Y=df.select([assets[j]]).to_series(),
                    exp_half_life=exp_half_life,
                )
        df_res = pd.DataFrame(res, columns=assets, index=assets)

    else:
        df_res = df.to_pandas().cov()
    return df_res


def compute_cov(
    df_ref: pl.DataFrame,
    ptf_assets: pl.Series,
    rebal_date: dt.datetime,
    n_month_delay: dt.datetime,
    log_rets: bool = False,
    rolling_window: int = 1,
    exp_weighted: bool = False,
    exp_half_life: float = 0,
) -> pd.DataFrame:
    df_temp = compute_rets(
        df_ref=df_ref,
        rebal_date=rebal_date,
        n_month_delay=n_month_delay,
        rolling_window=rolling_window,
        ptf_assets=ptf_assets,
    )

    df_temp = _set_up_for_cov_computation(
        df=df_temp, rolling_window=rolling_window, log_rets=log_rets
    )

    df_res = _compute_cov_matrix(
        df=df_temp,
        exp_weighted=exp_weighted,
        exp_half_life=exp_half_life,
        ptf_assets=ptf_assets,
    )

    return df_res * 252


def compute_dist(df_corr: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "ang_dist":
        return ((1 - df_corr) / 2) ** 0.5
    elif method == "abs_ang_dist":
        return ((1 - df_corr.abs()) / 2) ** 0.5
    elif method == "square_ang_dist":
        return ((1 - df_corr**2) / 2) ** 0.5
    else:
        print("Method should be in {'abs_dis', 'abs_ang_dist', 'square_ang_dist'}")


def compute_cluster(df_dist_corr: pd.DataFrame, method: str) -> np.ndarray:
    return linkage(df_dist_corr, method)


def compute_quasi_diag(link: np.ndarray) -> List[int]:
    # Sort clustered items by distance
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]  # number of original items
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
        df0 = sortIx[sortIx >= numItems]  # find clusters
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0]  # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = sortIx.append(df0)  # item 2
        sortIx = sortIx.sort_index()  # re-sort
        sortIx.index = range(sortIx.shape[0])  # re-index
    return sortIx.tolist()


def reorder_matrix(df: pd.DataFrame, df_ref_stock_id: pl.DataFrame) -> pd.DataFrame:
    new_index = df_ref_stock_id.select("asset").to_series().to_list()
    return df.reindex(new_index)[new_index]


def compute_weight(
    method: str,
    arr_linkage: np.array,
    df_sort_stocks_id: pl.DataFrame,
    weight_max: float,
    args_cov: Optional[dict],
    args_rets: Optional[dict],
    risk_aversion: Optional[float],
    args_vol_pred: Optional[dict],
):
    weights = pl.DataFrame
    if method == "top_down":
        weights = td.compute_top_down_w(
            arr_linkage=arr_linkage,
            df_sort_stocks_id=df_sort_stocks_id,
            args_cov=args_cov,
            args_rets=args_rets,
            risk_aversion=risk_aversion,
            args_vol_pred=args_vol_pred,
            weight_max=weight_max,
        )
    # elif method == "bottom_up":
    #     weights = bu.compute_bottom_up_w(
    #         linkage=arr_linkage, cov=quasi_diag_cov, sort_stocks_idx=sort_stocks_id
    #     )
    return weights


def compute_hrp(
    df_prediction: pl.DataFrame,
    df_reference: pl.DataFrame,
    distance_method: str,
    n_month_delay_corr: int,
    exp_weighted: bool,
    exp_half_life: float,
    linkage_method: str,
    weights_method: str,
    log_rets: bool = False,
    rolling_window: int = 1,
    ret_rolling_window: Optional[int] = None,
    n_month_delay_cov: Optional[int] = None,
    n_month_delay_ret: Optional[int] = None,
    risk_aversion: Optional[float] = None,
    df_vol_pred: Optional[pl.DataFrame] = None,
    pred_aversion: Optional[float] = None,
):
    dates = df_prediction.select("date").unique().to_series().sort()
    out = []
    args_rets = None
    args_cov_matrix = None
    for d in tqdm(dates[:1]):
        df_temp = df_prediction.filter(pl.col("date") == d)
        rebal_assets = df_temp.select("asset_id").to_series().unique()

        df_corr = compute_corr(
            df_ref=df_reference,
            ptf_assets=rebal_assets,
            n_month_delay=n_month_delay_corr,
            rebal_date=d,
        )

        args_cov_matrix = None
        if n_month_delay_cov is not None:
            args_cov_matrix = {
                "df_ref": df_reference,
                "ptf_assets": rebal_assets,
                "rebal_date": d,
                "n_month_delay": n_month_delay_cov,
                "log_rets": log_rets,
                "rolling_window": rolling_window,
                "exp_weighted": exp_weighted,
                "exp_half_life": exp_half_life,
            }

        args_rets = None
        if n_month_delay_ret is not None:
            args_rets = {
                "df_ref": df_reference,
                "ptf_assets": rebal_assets,
                "rebal_date": d,
                "n_month_delay": n_month_delay_ret,
                "rolling_window": ret_rolling_window,
            }

        assets_idx = pl.DataFrame(
            {
                "id": list(range(df_temp.shape[0])),
                "asset_id": df_corr.columns.to_list(),
            }
        )

        df_dist_corr = compute_dist(df_corr=df_corr, method=distance_method)
        arr_linkage = compute_cluster(df_dist_corr, linkage_method)
        print(dendrogram(arr_linkage))
        print(arr_linkage)

        sort_stocks_id = compute_quasi_diag(arr_linkage)

        df_sort_stock_id = pl.DataFrame({"id": sort_stocks_id}).join(
            pl.DataFrame(
                {
                    "id": list(range(df_corr.shape[0])),
                    "asset": df_corr.columns.to_list(),
                }
            ),
            on="id",
            how="left",
        )

        args_vol_pred = None
        if df_vol_pred is not None:
            args_vol_pred = {
                "df_pred": df_vol_pred,
                "pred_aversion": pred_aversion,
                "rebal_date": d,
                "prediction_aversion": pred_aversion,
            }

        weights = compute_weight(
            method=weights_method,
            arr_linkage=arr_linkage,
            df_sort_stocks_id=df_sort_stock_id,
            args_cov=args_cov_matrix,
            args_rets=args_rets,
            risk_aversion=risk_aversion,
            args_vol_pred=args_vol_pred,
            weight_max=self.weight_max,
        )

        assets_idx = assets_idx.join(weights, on="id", how="left")

        # display(assets_idx.sort("relative_weight"))

        out.append(
            df_temp.join(
                assets_idx.select(["asset_id", "relative_weight"]),
                on=["asset_id"],
                how="left",
            )
        )

    return pl.concat(out)
