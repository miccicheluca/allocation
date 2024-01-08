from typing import List, Optional, Tuple

import polars as pl

import datetime as dt

import numpy as np

import pandas as pd


from sklearn.covariance import LedoitWolf

from sklearn.covariance import OAS

import time as tm

from dataclasses import dataclass


from allocation_dataclass import (
    hc_args,
    hrp_cov_args,
    hrp_rets_args,
    hrp_vol_pred_args,
    current_date_infos,
)


@dataclass
class data:
    """
    Class containing elements for splitting inverse variance portfolio
    with performance/prediction according to a criteria in [0,1]
    """

    df: pl.DataFrame
    criteria: float


def init_clusters(
    arr_linkage: np.array,
    sort_stocks_id: List[int],
) -> dict:
    """Compute which stocks are in each clusters"""
    clusters = {elem: [elem] for elem in sort_stocks_id}

    new_asset_id = len(sort_stocks_id) - 1

    for cluster in arr_linkage:
        left_elem = int(cluster[0])

        right_elem = int(cluster[1])

        new_asset_id += 1

        clusters[new_asset_id] = clusters[left_elem] + clusters[right_elem]

    return clusters


def compute_vol_tilda_weight(cov: pd.DataFrame, elems: List[int]) -> np.array:
    """Compute inverse variance weighting"""
    tilda_weights = 1 / np.diag(cov.loc[elems, elems])

    return tilda_weights / np.sum(tilda_weights)


def compute_rets_tilda_weight(rets: pd.DataFrame):
    """Compute rets weighting"""
    tilda_weights = np.ones(rets.shape[0]) / rets.shape[0]

    return tilda_weights / np.sum(tilda_weights)


def compute_rets_cluster(
    df_rets: pl.DataFrame, df_sort_stocks_id: pl.DataFrame, elems: List[int]
) -> float:
    return (
        df_rets.filter(pl.col("id").is_in(elems))
        .select("weights")
        .to_series()
        .mean()
    )


def compute_vol(tilda_weight: pd.DataFrame, df_cov: pd.DataFrame) -> float:
    """Compute portfolio variance"""
    return np.dot(
        np.dot(
            tilda_weight,
            df_cov,
        ),
        tilda_weight.transpose(),
    )


def compute_weight_vol_pred(
    df_vol_pred: pl.DataFrame, elem: List[int]
) -> float:
    return (
        df_vol_pred.filter(pl.col("id").is_in(elem))
        .select("weights")
        .to_series()
        .mean()
    )


def compute_vol_risk_measure(
    df_cov: pd.DataFrame,
    clusters: dict,
    left_elem: List[int],
    right_elem: List[int],
) -> float:
    """Compute alpha left"""
    left_tilda_weights = compute_vol_tilda_weight(df_cov, clusters[left_elem])

    right_tilda_weights = compute_vol_tilda_weight(df_cov, clusters[right_elem])

    left_risk_measure = compute_vol(
        tilda_weight=left_tilda_weights,
        df_cov=df_cov.loc[clusters[left_elem], clusters[left_elem]],
    )

    right_risk_measure = compute_vol(
        tilda_weight=right_tilda_weights,
        df_cov=df_cov.loc[clusters[right_elem], clusters[right_elem]],
    )

    return right_risk_measure / (right_risk_measure + left_risk_measure)


def compute_vol_pred_risk_measure(
    df_vol_pred: pl.DataFrame,
    clusters: dict,
    left_elem: List[int],
    right_elem: List[int],
) -> float:
    """Compute logistic rank sacaled on the ranked volatility prediction and the left branch weight"""
    df_vol_pred = (
        df_vol_pred.filter(
            pl.col("id").is_in(clusters[left_elem] + clusters[right_elem])
        )
        .with_columns(
            pl.col("y_test_pred")
            .rank(method="ordinal")
            .cast(pl.Int64)
            .alias("rank")
        )
        .with_columns(
            (
                1
                / (
                    1
                    + (
                        -(1 / (pl.col("rank").max() / 5))
                        * (pl.col("rank") - (pl.col("rank").max() * (1 / 2)))
                    ).exp()
                )
            ).alias("weights")
        )
    )

    left_vol_pred_weight = compute_weight_vol_pred(
        df_vol_pred=df_vol_pred, elem=clusters[left_elem]
    )

    right_vol_pred_weight = compute_weight_vol_pred(
        df_vol_pred=df_vol_pred, elem=clusters[right_elem]
    )

    return right_vol_pred_weight / (
        left_vol_pred_weight + right_vol_pred_weight
    )


def compute_rets_measure(
    df_rets: pl.DataFrame,
    df_sort_stocks_id: pl.DataFrame,
    clusters: dict,
    left_elem: List[int],
    right_elem: List[int],
) -> float:
    left_rets = compute_rets_cluster(
        df_rets=df_rets,
        df_sort_stocks_id=df_sort_stocks_id,
        elems=clusters[left_elem],
    )

    right_rets = compute_rets_cluster(
        df_rets=df_rets,
        df_sort_stocks_id=df_sort_stocks_id,
        elems=clusters[right_elem],
    )

    return left_rets / (left_rets + right_rets)


def compute_right_left_alpha(
    vol_risk_measure: float,
    vol_pred_risk_measure: float,
    rets_measure: float,
    risk_aversion: float,
    prediction_aversion: float,
):
    alpha_left, alpha_right = 0.5, 0.5

    if vol_risk_measure is not None:
        alpha_left, alpha_right = vol_risk_measure, (1 - vol_risk_measure)

    elif rets_measure is not None:
        alpha_left, alpha_right = rets_measure, (1 - rets_measure)

    elif vol_pred_risk_measure is not None:
        alpha_left, alpha_right = vol_pred_risk_measure, (
            1 - vol_pred_risk_measure
        )

    if prediction_aversion is not None:
        alpha_left = (
            prediction_aversion * vol_risk_measure
            + (1 - prediction_aversion) * vol_pred_risk_measure
        )

        alpha_right = prediction_aversion * (1 - vol_risk_measure) + (
            1 - prediction_aversion
        ) * (1 - vol_pred_risk_measure)

    if risk_aversion is not None:
        alpha_left = (
            1 - risk_aversion
        ) * vol_risk_measure + risk_aversion * rets_measure

        alpha_right = (1 - risk_aversion) * (
            1 - vol_risk_measure
        ) + risk_aversion * (1 - rets_measure)

    return alpha_left, alpha_right


def compute_alpha(
    clusters: dict,
    left_elem: int,
    right_elem: int,
    df_sort_stocks_id: pl.DataFrame,
    df_cov: pd.DataFrame,
    rets_data: data,
    vol_pred_data: data,
) -> float:
    vol_risk_measure = None

    if df_cov is not None:
        vol_risk_measure = compute_vol_risk_measure(
            df_cov=df_cov,
            clusters=clusters,
            left_elem=left_elem,
            right_elem=right_elem,
        )

    vol_pred_risk_measure = None

    if vol_pred_data.df is not None:
        vol_pred_risk_measure = compute_vol_pred_risk_measure(
            df_vol_pred=vol_pred_data.df,
            clusters=clusters,
            left_elem=left_elem,
            right_elem=right_elem,
        )

    rets_measure = None

    if rets_data.df is not None:
        rets_measure = compute_rets_measure(
            df_rets=rets_data.df,
            df_sort_stocks_id=df_sort_stocks_id,
            clusters=clusters,
            left_elem=left_elem,
            right_elem=right_elem,
        )

    alpha_left, alpha_right = compute_right_left_alpha(
        vol_risk_measure=vol_risk_measure,
        vol_pred_risk_measure=vol_pred_risk_measure,
        rets_measure=rets_measure,
        risk_aversion=rets_data.criteria,
        prediction_aversion=vol_pred_data.criteria,
    )

    return alpha_left, alpha_right


def update_weight(
    weights: pl.DataFrame,
    alpha_left: float,
    alpha_right: float,
    left_clust_elem: int,
    right_clust_elem: int,
) -> pl.DataFrame:
    return weights.with_columns(
        [
            pl.when(pl.col("id").is_in(left_clust_elem))
            .then(pl.col("relative_weight") * alpha_left)
            .when(pl.col("id").is_in(right_clust_elem))
            .then(pl.col("relative_weight") * alpha_right)
            .otherwise(pl.col("relative_weight"))
            .alias("relative_weight")
        ]
    )


def _pariwise_exp_cov(X: pl.Series, Y: pl.Series, alpha: float) -> float:
    covariation = (X - X.mean()) * (Y - Y.mean())

    return covariation.ewm_mean(alpha=alpha)[-1]


def _compute_cov(
    args_cov: hrp_cov_args,
    rebal_infos: current_date_infos,
) -> pd.DataFrame:
    df_res = pd.DataFrame

    df_ref_temp = (
        rebal_infos.df_reference.filter(
            pl.col("date").is_between(
                rebal_infos.rebal_date
                - dt.timedelta(weeks=int(4 * args_cov.n_month_delay)),
                rebal_infos.rebal_date,
            )
        )
        .sort("date")
        .select(rebal_infos.rebal_assets)
    )

    if args_cov.exp_alpha > 0:
        # Loop over matrix, filling entries with the pairwise exp cov

        nb_assets = len(rebal_infos.rebal_assets)

        res = np.zeros((nb_assets, nb_assets))

        for i in range(nb_assets):
            for j in range(i, nb_assets):
                res[i, j] = res[j, i] = _pariwise_exp_cov(
                    X=df_ref_temp.select(
                        [rebal_infos.rebal_assets[i]]
                    ).to_series(),
                    Y=df_ref_temp.select(
                        [rebal_infos.rebal_assets[j]]
                    ).to_series(),
                    alpha=args_cov.exp_alpha,
                )

        df_res = pd.DataFrame(
            res,
            columns=rebal_infos.rebal_assets,
            index=rebal_infos.rebal_assets,
        )

    elif args_cov.shrinkage_method == "Ledoit_Wolf":
        res = LedoitWolf().fit(df_ref_temp.to_pandas().dropna()).covariance_

        df_res = pd.DataFrame(
            res,
            columns=rebal_infos.rebal_assets,
            index=rebal_infos.rebal_assets,
        )

    elif args_cov.shrinkage_method == "OAS":
        res = OAS().fit(df_ref_temp.to_pandas().dropna()).covariance_

        df_res = pd.DataFrame(
            res,
            columns=rebal_infos.rebal_assets,
            index=rebal_infos.rebal_assets,
        )

    else:
        df_res = df_ref_temp.to_pandas().cov()

    return df_res * 252


def _reorder_matrix(
    df: pd.DataFrame, df_ref_stock_id: pl.DataFrame
) -> pd.DataFrame:
    new_index = df_ref_stock_id.select("asset_id").to_series().to_list()

    return df.reindex(new_index)[new_index]


def get_bounds(
    n_elem_right: int,
    n_elem_left: int,
    bound_weight: float,
    actual_node_weight: float,
) -> Tuple[float]:
    return ((n_elem_left * bound_weight) / actual_node_weight), (
        (n_elem_right * bound_weight) / actual_node_weight
    )


def bound_alpha(
    alpha_left: float,
    alpha_right: float,
    left_elems: List[int],
    right_elems: List[int],
    weights: pl.DataFrame,
    weight_max: float,
    weight_min: float,
) -> Tuple[float]:
    actual_node_weight = (
        weights.filter(pl.col("id") == left_elems[0])
        .select("relative_weight")
        .to_series()[0]
    )

    upper_bound_left, upper_bound_right = get_bounds(
        n_elem_left=len(left_elems),
        n_elem_right=len(right_elems),
        bound_weight=weight_max,
        actual_node_weight=actual_node_weight,
    )

    lower_bound_left, lower_bound_right = get_bounds(
        n_elem_left=len(left_elems),
        n_elem_right=len(right_elems),
        bound_weight=weight_min,
        actual_node_weight=actual_node_weight,
    )

    if alpha_left > upper_bound_left:
        alpha_right += alpha_left - upper_bound_left

        alpha_left = upper_bound_left

    elif alpha_right > upper_bound_right:
        alpha_left += alpha_right - upper_bound_right

        alpha_right = upper_bound_right

    elif alpha_left < lower_bound_left:
        alpha_right -= lower_bound_left - alpha_left

        alpha_left = lower_bound_left

    elif alpha_right < lower_bound_right:
        alpha_left -= lower_bound_right - alpha_right

        alpha_right = lower_bound_right

    return alpha_left, alpha_right


def _compute_rets_matrix(
    rebal_infos: current_date_infos,
    args_rets: hrp_rets_args,
    df_sort_stocks_id: pl.DataFrame,
) -> pl.DataFrame:
    """Compute logistic rescale on ranked returns to compute the rets weight (in HARP method)"""
    df_rets = None

    risk_aversion = None

    if args_rets is not None:
        df_rets = (
            rebal_infos.df_reference.filter(
                (
                    pl.col("date")
                    >= rebal_infos.rebal_date
                    - dt.timedelta(weeks=4 * args_rets.n_month_delay)
                )
                & (pl.col("date") <= rebal_infos.rebal_date)
            )
            .select(rebal_infos.rebal_assets)
            .mean()
            .transpose(include_header=True, column_names=["daily_rets"])
            .with_columns(
                pl.col("daily_rets")
                .rank(method="ordinal")
                .cast(pl.Int64)
                .alias("daily_rets")
            )
            .with_columns(
                (
                    1
                    / (
                        1
                        + (
                            -(1 / (pl.col("daily_rets").max() / 6))
                            * (
                                pl.col("daily_rets")
                                - (pl.col("daily_rets").max() / 2)
                            )
                        ).exp()
                    )
                ).alias("daily_rets")
            )
        )

        df_rets.columns = ["asset_name", "weights"]

        df_rets = df_rets.join(
            df_sort_stocks_id.rename({"asset_id": "asset_name"}),
            on="asset_name",
            how="left",
        )

        risk_aversion = args_rets.risk_aversion

    return data(df=df_rets, criteria=risk_aversion)


def _compute_vol_pred_matrix(
    rebal_infos: current_date_infos,
    args_vol_pred: hrp_vol_pred_args,
    df_sort_stocks_id: pl.DataFrame,
) -> pl.DataFrame:
    df_vol_pred = None

    pred_aversion = None

    if args_vol_pred is not None:
        df_vol_pred = args_vol_pred.df_vol_pred

        df_vol_pred = df_vol_pred.filter(
            (pl.col("date") == rebal_infos.rebal_date)
            & (
                pl.col("asset_id").is_in(
                    df_sort_stocks_id.select("asset_id").to_series()
                )
            )
        ).join(
            df_sort_stocks_id.rename({"asset_id": "asset_id"}),
            on="asset_id",
            how="left",
        )

        pred_aversion = args_vol_pred.pred_aversion

    return data(df=df_vol_pred, criteria=pred_aversion)


def compute_top_down_w(
    arr_linkage: np.array,
    df_sort_stocks_id: pl.DataFrame,
    rebal_infos: current_date_infos,
    args_hc: Optional[hc_args] = None,
    args_cov: Optional[hrp_cov_args] = None,
    args_rets: Optional[hrp_rets_args] = None,
    args_vol_pred: Optional[hrp_vol_pred_args] = None,
):
    sort_stocks_id = df_sort_stocks_id.select("id").to_series().to_list()

    weights = pl.DataFrame(
        {
            "id": sort_stocks_id,
            "relative_weight": np.ones(df_sort_stocks_id.shape[0]),
        }
    )

    clusters = init_clusters(
        arr_linkage=arr_linkage, sort_stocks_id=sort_stocks_id
    )

    df_cov_qd = None

    if args_cov is not None:
        df_cov = _compute_cov(rebal_infos=rebal_infos, args_cov=args_cov)

        df_cov_qd = _reorder_matrix(df_cov, df_sort_stocks_id)

        df_cov_qd.columns = sort_stocks_id

        df_cov_qd.index = sort_stocks_id

    rets_data = _compute_rets_matrix(
        rebal_infos=rebal_infos,
        df_sort_stocks_id=df_sort_stocks_id,
        args_rets=args_rets,
    )

    vol_pred_data = _compute_vol_pred_matrix(
        rebal_infos=rebal_infos,
        args_vol_pred=args_vol_pred,
        df_sort_stocks_id=df_sort_stocks_id,
    )

    for cluster in arr_linkage[::-1]:
        left_elem = int(cluster[0])

        right_elem = int(cluster[1])

        alpha_left, alpha_right = compute_alpha(
            clusters=clusters,
            left_elem=left_elem,
            right_elem=right_elem,
            df_sort_stocks_id=df_sort_stocks_id,
            df_cov=df_cov_qd,
            rets_data=rets_data,
            vol_pred_data=vol_pred_data,
        )

        alpha_left, alpha_right = bound_alpha(
            alpha_left=alpha_left,
            alpha_right=alpha_right,
            left_elems=clusters[left_elem],
            right_elems=clusters[right_elem],
            weights=weights,
            weight_max=args_hc.weight_max,
            weight_min=args_hc.weight_min,
        )

        weights = update_weight(
            weights=weights,
            alpha_left=alpha_left,
            alpha_right=alpha_right,
            left_clust_elem=clusters[left_elem],
            right_clust_elem=clusters[right_elem],
        )

    return weights
