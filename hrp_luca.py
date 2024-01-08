from typing import List, Optional

import polars as pl

import datetime as dt

import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression


from tqdm import tqdm

import fastcluster as fc

from scipy.cluster.hierarchy import (
    linkage,
    dendrogram,
    ward,
    optimal_leaf_ordering,
    leaves_list,
)


import top_down as td

from allocation_dataclass import (
    hc_args,
    hrp_residuals_rets_args,
    hrp_corr_args,
    hrp_cov_args,
    hrp_rets_args,
    hrp_vol_pred_args,
    current_date_infos,
)


class HRP:
    def __init__(
        self,
        df_ref: pl.DataFrame,
    ):
        """
        df_ref: raw data set

        The function creates here a TxN matrix of stock returns which will be used to compute correlation and covariance matrix in the process.
        T is the number of date, N is the number of stock in the universe.
        """
        self.df_reference = self._pivot_data(
            df=self._compute_rets(df_ref=df_ref),
        )

    def _compute_rets(
        self,
        df_ref: pl.DataFrame,
    ) -> pl.DataFrame:
        return df_ref.with_columns(
            [
                (
                    pl.col("px_last")
                    .fill_nan(None)
                    .fill_null(strategy="forward")
                    .pct_change(1)
                    .clip(lower_bound=-0.25, upper_bound=0.25)
                    .over("asset_name")
                    .alias("daily_rets")
                ),
            ]
        ).sort("date")

    def _pivot_data(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.select(["date", "asset_name", "daily_rets"]).pivot(
            index="date", columns="asset_name", values="daily_rets"
        )

    def _pariwise_exp_cov(
        self, X: pl.Series, Y: pl.Series, alpha: float
    ) -> float:
        """Compute alpha exponentially weighted covariance matrix between to series X and Y"""
        covariation = (X - X.mean()) * (Y - Y.mean())

        return covariation.ewm_mean(alpha=alpha)[-1]

    def _compute_corr(
        self,
        infos: current_date_infos,
        args_corr: hrp_corr_args,
    ) -> pd.DataFrame:
        """ "
        infos: corresponds to the actual rebal infos (date, assets in portfolio, )
        args_corr: corresponds to the correlation matrix computation infos (lag, type and method)

        Compute the correlation matrix which will be used to create the hierarchi through our univers
        """
        df_ref_temp = (
            infos.df_reference.filter(
                (
                    pl.col("date")
                    >= infos.rebal_date
                    - dt.timedelta(weeks=4 * args_corr.n_month_delay)
                )
                & (pl.col("date") <= infos.rebal_date)
            )
            .sort("date")
            .select(infos.rebal_assets)
        )

        if args_corr.exp_alpha > 0:
            # Loop over matrix, filling entries with the pairwise exp cov

            nb_assets = len(infos.rebal_assets)

            res = np.zeros((nb_assets, nb_assets))

            for i in range(nb_assets):
                for j in range(i, nb_assets):
                    res[i, j] = res[j, i] = self._pariwise_exp_cov(
                        X=df_ref_temp.select(
                            [infos.rebal_assets[i]]
                        ).to_series(),
                        Y=df_ref_temp.select(
                            [infos.rebal_assets[j]]
                        ).to_series(),
                        alpha=args_corr.exp_alpha,
                    )

            df_res = self.cov2corr(
                pd.DataFrame(
                    res,
                    columns=infos.rebal_assets,
                    index=infos.rebal_assets,
                )
            )

        else:
            df_res = df_ref_temp.to_pandas().corr(method=args_corr.corr_method)

        return df_res

    def cov2corr(self, df_cov: pd.DataFrame) -> pd.DataFrame:
        """Derive the correlation matrix from a covariance matrix"""

        std = np.sqrt(np.diag(df_cov))

        corr = df_cov / np.outer(std, std)

        corr[corr < -1], corr[corr > 1] = -1, 1  # numerical error

        return corr

    def _compute_cluster(
        self, df_dist_corr: pd.DataFrame, method: str
    ) -> np.ndarray:
        return fc.linkage(df_dist_corr, method)

    def _compute_dist(self, df_corr: pd.DataFrame, method: str) -> pd.DataFrame:
        if method == "ang_dist":
            return ((1 - df_corr) / 2) ** 0.5

        if method == "abs_ang_dist":
            return ((1 - df_corr.abs()) / 2) ** 0.5

        if method == "square_ang_dist":
            return ((1 - df_corr**2) / 2) ** 0.5

        else:
            print(
                "Method should be in {'abs_dis', 'abs_ang_dist', 'square_ang_dist'}"
            )

    def compute_quasi_diag(
        self, arr_linkage: np.ndarray, df_dist: pd.DataFrame
    ) -> List[int]:
        """rearrangement according to clusters. This list is used to creat the quasi-diagonal matrix"""
        return leaves_list(optimal_leaf_ordering(arr_linkage, df_dist))

    def _compute_weight(
        self,
        method: str,
        arr_linkage: np.array,
        rebal_infos: current_date_infos,
        df_sort_stocks_id: pl.DataFrame,
        args_hc: Optional[hc_args] = None,
        args_cov: Optional[hrp_cov_args] = None,
        args_rets: Optional[hrp_rets_args] = None,
        args_vol_pred: Optional[hrp_vol_pred_args] = None,
    ):
        """
        method: top-down or bottom-up apporach (bottum up need to be implemented)
        arr_linkage: represents cluster hierarchi given by scipy
        rebal_infos: represents the current rebal infos (date, assets in ptf etc..)
        df_sort_stock_id: corresponds to the identification of stocks and their id in the hierarchi
        args_hc: infos about the computation of the hierarchi
        args_cov: infos about the computation of covariance matrix for top-down approach
        args_rets: infos about the computation of returns if you want to use HARP approach
        args_vol_pred: infos if you want to use a "future aversion" method

        if you want to use equaly weighted hrp: give only args hc
        if you want to use simple hrp: give args_hc & args_cov
        if you want to use harp: give args_hc, args_cov, args_rets
        """
        weights = pl.DataFrame

        if method == "top_down":
            weights = td.compute_top_down_w(
                arr_linkage=arr_linkage,
                rebal_infos=rebal_infos,
                df_sort_stocks_id=df_sort_stocks_id,
                args_hc=args_hc,
                args_cov=args_cov,
                args_rets=args_rets,
                args_vol_pred=args_vol_pred,
            )

        return weights

    def _compute_pca_residuals(
        self,
        df_cov: pd.DataFrame,
        past_rets: pd.DataFrame,
        nb_eigen_vectors: int,
    ) -> pd.DataFrame:
        eigen_vals, eigen_vecs = np.linalg.eig(df_cov)

        permutations = np.argsort(-eigen_vals)

        eigen_vals = eigen_vals[permutations]

        eigen_vecs = eigen_vecs[:, permutations]

        zeroed_returns = past_rets.replace(np.nan, 0)

        factors = zeroed_returns @ eigen_vecs.real

        factors_5 = factors.iloc[:, :nb_eigen_vectors]

        reg = LinearRegression(fit_intercept=True).fit(
            X=factors_5, y=past_rets.fillna(0)
        )

        pca_rets = reg.predict(factors_5)

        return past_rets - pca_rets

    def _compute_df_ref_temp(
        self,
        rebal_date: dt.datetime,
        rebal_assets: List[str],
        args_rets_residual: hrp_residuals_rets_args,
    ) -> pl.DataFrame:
        df_ref_temp = self.df_reference.select(pl.col(["date"] + rebal_assets))

        if args_rets_residual is not None:
            past_rets = (
                df_ref_temp.filter(
                    pl.col("date").is_between(
                        rebal_date
                        - dt.timedelta(
                            weeks=4 * args_rets_residual.n_month_delay
                        ),
                        rebal_date,
                    )
                )
                .select(rebal_assets)
                .to_pandas()
            )

            dates = df_ref_temp.filter(
                pl.col("date").is_between(
                    rebal_date
                    - dt.timedelta(weeks=4 * args_rets_residual.n_month_delay),
                    rebal_date,
                )
            ).select("date")

            df_cov = past_rets.cov()

            df_residuals = self._compute_pca_residuals(
                df_cov=df_cov,
                past_rets=past_rets,
                nb_eigen_vectors=args_rets_residual.nb_eigen_vectors,
            )

            df_ref_temp = pl.concat(
                [
                    dates,
                    pl.DataFrame(df_residuals),
                ],
                how="horizontal",
            )

        return df_ref_temp

    def _compute_current_date_infos(
        self,
        df_prediction: pl.DataFrame,
        rebal_date: dt.datetime,
        args_rets_residual: hrp_residuals_rets_args,
    ) -> current_date_infos:
        df_pred_temp = df_prediction.filter(pl.col("date") == rebal_date)

        rebal_assets = df_pred_temp.select("asset_id").to_series()

        df_ref_temp = self._compute_df_ref_temp(
            rebal_assets=rebal_assets,
            rebal_date=rebal_date,
            args_rets_residual=args_rets_residual,
        )

        return current_date_infos(
            df_reference=df_ref_temp,
            df_pred=df_pred_temp,
            rebal_date=rebal_date,
            rebal_assets=rebal_assets,
        )

    def compute_hrp(
        self,
        df_prediction: pl.DataFrame,
        args_corr: hrp_corr_args,
        args_cov: Optional[hrp_cov_args] = None,
        args_rets_residual: Optional[hrp_residuals_rets_args] = None,
        args_rets: Optional[hrp_rets_args] = None,
        args_vol_pred: Optional[hrp_vol_pred_args] = None,
        args_hc: Optional[hc_args] = None,
    ) -> pl.DataFrame:
        out = []

        dates = df_prediction.select("date").to_series().unique().sort()

        for d in tqdm(dates):
            rebal_infos = self._compute_current_date_infos(
                df_prediction=df_prediction,
                rebal_date=d,
                args_rets_residual=args_rets_residual,
            )

            df_corr = self._compute_corr(infos=rebal_infos, args_corr=args_corr)

            df_dist_corr = self._compute_dist(
                df_corr=df_corr, method=args_hc.distance_method
            )

            arr_linkage = self._compute_cluster(
                df_dist_corr=df_dist_corr, method=args_hc.linkage_method
            )

            sort_stocks_id = self.compute_quasi_diag(
                arr_linkage, df_dist_corr
            ).astype(np.int64)

            assets_id = pl.DataFrame(
                {
                    "id": list(range(df_corr.shape[0])),
                    "asset_id": df_corr.columns.to_list(),
                }
            )

            df_sort_stock_id = pl.DataFrame({"id": sort_stocks_id}).join(
                assets_id, on="id", how="left"
            )

            weights = self._compute_weight(
                method=args_hc.weights_method,
                arr_linkage=arr_linkage,
                rebal_infos=rebal_infos,
                df_sort_stocks_id=df_sort_stock_id,
                args_cov=args_cov,
                args_rets=args_rets,
                args_vol_pred=args_vol_pred,
                args_hc=args_hc,
            )

            assets_id = assets_id.join(weights, on="id", how="left")

            out.append(
                rebal_infos.df_pred.join(
                    assets_id.select(["asset_id", "relative_weight"]),
                    on=["asset_id"],
                    how="left",
                )
            )

        return pl.concat(out)
