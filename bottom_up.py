import polars as pl
import datetime as dt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, ward
import matplotlib.pyplot as plt
from typing import List
import time as tm


def compute_inv_variance(
    cov: pd.DataFrame,
    left_elem: int,
    right_elem: int,
) -> float:
    var_left = cov.iloc[left_elem, left_elem]
    var_right = cov.iloc[right_elem, right_elem]
    return var_right / (var_right + var_left)


def update_weight(
    weights: pl.DataFrame,
    alpha: float,
    left_clust_elem: int,
    right_clust_elem: int,
) -> pl.DataFrame:
    return weights.with_columns(
        [
            pl.when(pl.col("id").is_in(left_clust_elem))
            .then(pl.col("relative_weight") * alpha)
            .when(pl.col("id").is_in(right_clust_elem))
            .then(pl.col("relative_weight") * (1 - alpha))
            .otherwise(pl.col("relative_weight"))
            .alias("relative_weight")
        ]
    )


def update_clusters(
    clusters: dict, left_elem: int, right_elem: int, new_asset_id: int
) -> dict:
    clusters[new_asset_id] = clusters[left_elem] + clusters[right_elem]
    return clusters


def update_cov(
    cov: pd.DataFrame,
    alpha: float,
    left_elem: int,
    right_elem: int,
    id_left_elem: int,
    id_right_elem: int,
    n_cluster: int,
    new_asset_id: int,
    n_assets: int,
) -> pd.DataFrame:
    A = np.identity(n_assets - n_cluster + 1)
    A[id_left_elem][id_left_elem] = alpha
    A[id_left_elem][id_right_elem] = 1 - alpha
    A = np.delete(A, id_right_elem, 0)
    # print("matrix A: ", A)

    new_cov = pd.DataFrame(np.dot(np.dot(A, cov.to_numpy()), A.transpose()))
    prev_sorted_assets = cov.columns.to_list()
    # print("PREV SORTED ASSETS", prev_sorted_assets)
    if left_elem == cov.columns[0]:
        new_sorted_assets = [new_asset_id] + prev_sorted_assets[id_right_elem + 1 :]
        # print("AHHHHHHHHHHHHHH",new_sorted_assets)
    elif right_elem == cov.columns[-1]:
        # print(prev_sorted_assets[:1])
        new_sorted_assets = prev_sorted_assets[:id_left_elem] + [new_asset_id]
    else:
        new_sorted_assets = (
            prev_sorted_assets[:id_left_elem]
            + [new_asset_id]
            + prev_sorted_assets[id_right_elem + 1 :]
        )

    new_cov.columns = new_sorted_assets
    new_cov["id"] = new_sorted_assets

    # print(new_cov.set_index("assets"))
    return new_cov.set_index("id"), new_sorted_assets


def compute_bottom_up_w(linkage: np.array, cov: pd.DataFrame, sort_stocks_idx: List):
    weights = pl.DataFrame(
        {
            "id": sort_stocks_idx,
            "relative_weight": [1 for _ in range(len(sort_stocks_idx))],
        }
    )
    cov.columns = sort_stocks_idx
    cov.index = sort_stocks_idx
    clusters = {elem: [elem] for elem in sort_stocks_idx}
    n_assets = len(clusters)
    new_asset_id = n_assets - 1
    # print(assets_id)
    # print(clusters)
    for i, cluster in enumerate(linkage):
        # print("ITTERATION: ", i)
        left_elem = int(cluster[0])
        right_elem = int(cluster[1])
        id_left_elem = sort_stocks_idx.index(left_elem)
        id_right_elem = sort_stocks_idx.index(right_elem)
        new_asset_id += 1
        alpha = compute_inv_variance(cov, id_left_elem, id_right_elem)
        # print("ALPHA: ", alpha)
        weights = update_weight(
            weights, alpha, clusters[left_elem], clusters[right_elem]
        )
        # print("WEIGHTS: ", weights)
        clusters = update_clusters(clusters, left_elem, right_elem, new_asset_id)
        # print("CLUSTERS: ", clusters)
        cov, sort_stocks_idx = update_cov(
            cov=cov,
            alpha=alpha,
            left_elem=left_elem,
            right_elem=right_elem,
            id_left_elem=id_left_elem,
            id_right_elem=id_right_elem,
            n_cluster=i + 1,
            new_asset_id=new_asset_id,
            n_assets=n_assets,
        )
        # print("COV", cov)
    return weights


def compute_top_down_w():
    return None
