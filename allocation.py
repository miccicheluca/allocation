from tqdm import tqdm

from typing import Optional, Dict, Tuple

import polars as pl

import datetime as dt

import weight_schemes as ws


class PortfolioAllocation:
    def __init__(
        self,
        predictions: pl.DataFrame,
        weight_recipe: Dict,
        amount_to_invest: float,
        price_column: str,
        asset_id_column: str,
        date_column: str,
        portfolio_column: str,
        meta_vol_column: str,
        weight_constraints: Optional[Tuple] = None,
        vol_target_recipe: Optional[dict] = None,
        start_alloc: Optional[dt.datetime] = None,
        end_alloc: Optional[dt.datetime] = None,
        subsample: Optional[str] = None,
        sample_time: Optional[str] = None,
        rebal_frequency: int = 1,
    ):
        self.allocation_history = None

        self.predictions = predictions

        self.weight_recipe = weight_recipe

        self.weight_constraints = weight_constraints

        self.vol_target_recipe = vol_target_recipe

        if start_alloc is not None and end_alloc is not None:
            self.predictions = self.predictions.filter(
                pl.col(date_column).is_between(start_alloc, end_alloc)
            )

        self.subsample = subsample

        self.sample_time = sample_time

        self.rebal_frequency = rebal_frequency

        self.amount_to_invest = amount_to_invest

        self.price_column = price_column

        self.asset_id_column = asset_id_column

        self.date_column = date_column

        self.portfolio_column = portfolio_column

        # subsample

        if (self.subsample is not None) & (self.sample_time is not None):
            self._subsample()

        self.dates = sorted(self.predictions[self.date_column].unique())

        # compute nr of constituents per portfolio

        self._compute_nr_constituents()

        # compute relative weights & volatility target constraints

        self._compute_relative_weights()

        self._compute_vol_target()

        if self.weight_constraints is not None:
            func = self.weight_constraints[0]

            args = self.weight_constraints[1]

            self.predictions = self.predictions.pipe(func, **args)

    def get_allocation_history(self) -> pl.DataFrame:
        """

        Returns the allocation history.

        """

        return self.allocation_history

    def perform_allocation_backtest(self) -> None:
        """

        Perform the historical backtest.

        """

        store_allocation = []

        first_date = self.dates[0]

        rest_dates = self.dates[1:]

        df_q0 = self.predictions.filter(pl.col(self.date_column) == first_date)

        out_alloc = self.first_allocation(df_q0)

        store_allocation.append(out_alloc)

        for d in tqdm(rest_dates):
            df_new_q = self.predictions.filter(pl.col(self.date_column) == d)

            out_alloc = self.update_allocation(df_new_q, out_alloc)

            store_allocation.append(out_alloc)

            out_alloc = out_alloc.filter(
                (pl.col("status") == "new_portfolio")
                | (pl.col("status") == "amount_changes")
                | (pl.col("status") == "out_universe")
            )

        self.allocation_history = pl.concat(store_allocation)

    def _subsample(self) -> None:
        """

        Subsample dataframe to weekly/monthly (last or first day).

        """

        df = (
            self.predictions.with_columns(
                [
                    pl.col(self.date_column).dt.year().alias("year"),
                    pl.col(self.date_column).dt.month().alias("month"),
                    pl.col(self.date_column).dt.week().alias("week"),
                    pl.col(self.date_column).dt.day().alias("day"),
                ]
            )
            .with_columns(
                [(pl.col(self.subsample) % self.rebal_frequency).alias("rebal")]
            )
            .filter(pl.col("rebal") == 0)
        )

        if self.subsample in ["week", "month"]:
            if self.sample_time == "first":
                dates = (
                    df.groupby(["year", self.subsample])
                    .first()
                    .select(self.date_column)
                    .to_series()
                )

            elif self.sample_time == "last":
                dates = (
                    df.groupby(["year", self.subsample])
                    .last()
                    .select(self.date_column)
                    .to_series()
                )

            else:
                raise ValueError("sample_time should be first or last")

        elif self.subsample == "day":
            dates = df.select(self.date_column).to_series().unique()

        else:
            raise ValueError("subsample should be week, month or day")

        self.predictions = self.predictions.filter(
            pl.col(self.date_column).is_in(dates)
        )

    def first_allocation(self, df: pl.DataFrame) -> pl.DataFrame:
        """ """

        return (
            df.pipe(self._first_allocation_helper)
            .with_columns(
                [
                    pl.lit(None).alias(f"{self.date_column}_t_m_1"),
                    pl.lit(None).alias(f"{self.price_column}_t_m_1"),
                    pl.lit(None).alias(f"{self.portfolio_column}_t_m_1"),
                    pl.lit(None).alias("nr_constituents_t_m_1"),
                    pl.col(f"{self.portfolio_column}").alias(
                        f"{self.portfolio_column}_t_p_1"
                    ),
                ]
            )
            .pipe(self._order_columns)
            .pipe(self._format_dtypes_alloc)
        )

    def _first_allocation_helper(self, df: pl.DataFrame) -> pl.DataFrame:
        """

        Performs the iteration of the portfolio allocation process

        First allocation the current is always 0

        Target amount = amount to invest / # constituents / asset price

        Amount to buy sell = Target amount - current amount

        """

        return (
            df.with_columns(pl.lit(0.0).alias("current_amount"))
            .pipe(self._compute_target_amount)
            .with_columns(
                [
                    (pl.col("target_amount") - pl.col("current_amount"))
                    .round(0)
                    .alias("amount_buy_sell"),
                    (pl.lit("new_portfolio").alias("status")),
                ]
            )
        )

    def update_allocation(
        self, df_new_q: pl.DataFrame, df_prev_alloc: pl.DataFrame
    ) -> pl.DataFrame:
        """ """

        df_changes = self._find_asset_changes(df_new_q, df_prev_alloc)

        df_out = self._find_assets_out(df_new_q, df_prev_alloc)

        df_new = self._find_new_assets_in(df_new_q, df_prev_alloc)

        return pl.concat([df_changes, df_out, df_new])

    def _compute_nr_constituents(self) -> None:
        """

        Computes the number of consituents per portfolio per date.

        """

        self.predictions = self.predictions.with_columns(
            pl.col(self.date_column)
            .count()
            .over([self.date_column, self.portfolio_column])
            .alias("nr_constituents")
        )

    def _find_asset_changes(
        self, df_new_q: pl.DataFrame, df_prev_alloc: pl.DataFrame
    ) -> pl.DataFrame:
        """ """

        # 0) format dtypes new query

        df_new_q = df_new_q.pipe(self._format_dtypes_query)

        # 1) update names old portfolio

        df_prev_alloc = df_prev_alloc.select(
            [
                self.date_column,
                self.asset_id_column,
                self.portfolio_column,
                self.price_column,
                "nr_constituents",
                "target_amount",
                "relative_weight",
                "vol_target",
            ]
        )

        # 2) inner join and add previous portfolio column

        df_new_port = (
            df_new_q.select(
                [
                    self.date_column,
                    self.asset_id_column,
                    self.portfolio_column,
                    self.price_column,
                    "nr_constituents",
                    "relative_weight",
                    "vol_target",
                ]
            )
            .join(
                df_prev_alloc,
                on=[self.asset_id_column, self.portfolio_column],
                suffix="_t_m_1",
                how="inner",
            )
            .with_columns(
                [
                    pl.col(self.portfolio_column).alias(
                        f"{self.portfolio_column}_t_m_1"
                    ),
                    pl.lit(None).alias(f"{self.portfolio_column}_t_p_1"),
                ]
            )
        )

        # 3) commpute target amount & how much to buy sell

        return (
            df_new_port.with_columns(
                pl.col("target_amount").round(0).alias("current_amount")
            )
            .pipe(self._compute_target_amount)
            .with_columns(
                [
                    (pl.col("target_amount") - pl.col("current_amount"))
                    .round(0)
                    .alias("amount_buy_sell"),
                    pl.lit("amount_changes").alias("status"),
                ]
            )
            .pipe(self._order_columns)
            .pipe(self._format_dtypes_alloc)
        )

    def _find_assets_out(
        self, df_new_q: pl.DataFrame, df_prev_alloc: pl.DataFrame
    ) -> pl.DataFrame:
        """

        Find assets that change portfolio or left the universe.

        """

        # 0) format dtypes new query

        df_new_q = df_new_q.pipe(self._format_dtypes_query)

        # 1) get all rows with no match

        df_out = df_prev_alloc.join(
            df_new_q,
            on=[self.asset_id_column, self.portfolio_column],
            suffix="_out",
            how="anti",
        ).select(
            [
                pl.col(self.asset_id_column),
                pl.col(
                    [
                        self.date_column,
                        self.price_column,
                        self.portfolio_column,
                        "nr_constituents",
                    ]
                ).suffix("_t_m_1"),
                pl.col("target_amount"),
                pl.col("relative_weight"),
                pl.col("vol_target"),
            ]
        )

        # 2.1) changed portfolio and add info to fill nans (inner)

        # for which to find info

        df_out_port = df_out.join(
            df_new_q.select(
                [
                    self.date_column,
                    self.asset_id_column,
                    self.price_column,
                    self.portfolio_column,
                    "nr_constituents",
                    "relative_weight",
                    "vol_target",
                ]
            ),
            on=[self.asset_id_column],
            how="inner",
        ).with_columns(
            [
                pl.col("target_amount").alias("current_amount"),
                pl.lit("out_portfolio").alias("status"),
                (pl.col(self.portfolio_column).alias(f"{self.portfolio_column}_t_p_1")),
            ]
        )

        df_out_port = df_out_port.with_columns(pl.lit(0.0).alias("target_amount"))

        df_out_port = (
            df_out_port.with_columns(
                [
                    (pl.col("target_amount") - pl.col("current_amount"))
                    .round(0)
                    .alias("amount_buy_sell"),
                    (
                        pl.col(f"{self.portfolio_column}_t_m_1").alias(
                            self.portfolio_column
                        )
                    ),
                ]
            )
            .pipe(self._format_dtypes_alloc)
            .pipe(self._order_columns)
        )

        # 2.2)  out of universe

        # todays date

        date_new_q = (
            df_new_q.filter(pl.col(self.date_column) != None)
            .select(self.date_column)
            .to_series()[0]
        )

        df_out_uni = (
            df_out.join(
                df_new_q.select(
                    [
                        self.date_column,
                        self.asset_id_column,
                        self.price_column,
                        self.portfolio_column,
                        "nr_constituents",
                        "relative_weight",
                        "vol_target",
                    ]
                ),
                on=[self.asset_id_column],
                how="anti",
            )
            .with_columns(
                [
                    pl.lit(date_new_q).alias(self.date_column),
                    pl.col(f"{self.price_column}_t_m_1").alias(self.price_column),
                    pl.col(f"{self.portfolio_column}_t_m_1").alias(
                        self.portfolio_column
                    ),
                    pl.col("nr_constituents_t_m_1").alias("nr_constituents"),
                    pl.col("target_amount").alias("current_amount"),
                    pl.lit("out_universe").alias("status"),
                ]
            )
            .filter(pl.col("current_amount") > 0)
        )

        df_out_uni = df_out_uni.with_columns(pl.lit(0.0).alias("target_amount"))

        df_out_uni = (
            df_out_uni.with_columns(
                [
                    (pl.col("target_amount") - pl.col("current_amount"))
                    .round(0)
                    .alias("amount_buy_sell"),
                    (
                        pl.col(f"{self.portfolio_column}_t_m_1").alias(
                            self.portfolio_column
                        )
                    ),
                    (pl.lit(None).alias(f"{self.portfolio_column}_t_p_1")),
                ]
            )
            .pipe(self._format_dtypes_alloc)
            .pipe(self._order_columns)
        )

        return pl.concat([df_out_port, df_out_uni]).pipe(self._order_columns)

    def _find_new_assets_in(
        self, df_new_q: pl.DataFrame, df_prev_alloc: pl.DataFrame
    ) -> pl.DataFrame:
        """

        Compute per query

        """

        # 0) format dtypes new query

        df_new_q = df_new_q.pipe(self._format_dtypes_query)

        # 1) update names old portfolio

        df_prev_alloc = df_prev_alloc.with_columns(
            [pl.col(self.portfolio_column).alias("portfolio_t_m_1")]
        )

        # 2) left join and get all nans = new assets in

        df_new_in_port = (
            df_new_q.join(
                df_prev_alloc,
                on=[self.asset_id_column, self.portfolio_column],
                suffix="_new",
                how="left",
            )
            .filter(pl.col(f"{self.date_column}_new").is_null())
            .select(
                [
                    pl.col(
                        [
                            self.asset_id_column,
                            self.date_column,
                            self.price_column,
                            self.portfolio_column,
                            "nr_constituents",
                            "relative_weight",
                            "vol_target",
                        ]
                    )
                ]
            )
        )

        # 3) fill the nan columns

        df_prev_alloc_sub = df_prev_alloc.select(
            [
                pl.col(self.asset_id_column),
                pl.col(
                    [
                        self.date_column,
                        self.price_column,
                        self.portfolio_column,
                        "nr_constituents",
                    ]
                ).suffix("_t_m_1"),
            ]
        )

        df_new_in_port = df_new_in_port.join(
            df_prev_alloc_sub, on=[self.asset_id_column], how="left"
        ).with_columns(
            pl.col(self.portfolio_column).alias(f"{self.portfolio_column}_t_p_1")
        )

        # 4) compute how much to buy: you can treat them as first allocation

        return (
            self._first_allocation_helper(df_new_in_port)
            .pipe(self._order_columns)
            .pipe(self._format_dtypes_alloc)
        )

    def _normalize_weights(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize the weighting so its to 1."""

        return df.with_columns(
            (pl.col("relative_weight") / pl.col("relative_weight").sum())
            .over(self.date_column)
            .alias("relative_weight")
        )

    def _compute_relative_weights(self) -> None:
        """Computes weight between (0 and 1) for every asset. Sum is 1."""

        store = []

        for k, v in self.weight_recipe.items():
            func = v[0]

            args = v[1]

            out = (
                self.predictions.filter(pl.col(self.portfolio_column) == k)
                .pipe(func, **args)
                .pipe(self._normalize_weights)
            )

            store.append(out)

        self.predictions = pl.concat(store)

    def _compute_vol_target(self) -> None:
        """Computes weight between (0 and 1) for every asset. Sum is 1."""

        # self.predictions = self.predictions.with_columns(

        #     pl.when((vol_target / pl.col(meta_vol_column) > 1.0))

        #     .then(vol_target / pl.col(meta_vol_column))

        #     .otherwise(vol_target / pl.col(meta_vol_column))

        #     .alias("vol_target")

        # ).with_columns(

        #     pl.col("vol_target").fill_nan(None).fill_null(pl.lit(1)).alias("vol_target")

        # )

        self.predictions = self.predictions.with_columns(pl.lit(1).alias("vol_target"))

        if self.vol_target_recipe is not None:
            func = self.vol_target_recipe[0]

            args = self.vol_target_recipe[1]

            self.predictions = self.predictions.pipe(func, **args)

        # store = []

        # for k, v in self.vol_target_recipe.items():

        #     func = v[0]

        #     args = v[1]

        #     out = self.predictions.filter(pl.col(self.portfolio_column) == k).pipe(

        #         func, **args

        #     )

        #     store.append(out)

        # self.predictions = pl.concat(store)

    def _compute_target_amount(self, df: pl.DataFrame) -> pl.DataFrame:
        """ """

        if self.vol_target_recipe is not None:
            return df.with_columns(
                (
                    (
                        pl.min(pl.col("vol_target"), 1)
                        * pl.col("relative_weight")
                        * self.amount_to_invest
                        / pl.col(self.price_column)
                    ).round(0)
                ).alias("target_amount")
            )

        else:
            return df.with_columns(
                (
                    (
                        pl.col("relative_weight")
                        * self.amount_to_invest
                        / pl.col(self.price_column)
                    ).round(0)
                ).alias("target_amount")
            )

    def _order_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """ """

        return df.select(
            [
                self.asset_id_column,
                f"{self.date_column}_t_m_1",
                f"{self.price_column}_t_m_1",
                f"{self.portfolio_column}_t_m_1",
                f"{self.portfolio_column}_t_p_1",
                "nr_constituents_t_m_1",
                self.date_column,
                self.price_column,
                self.portfolio_column,
                "nr_constituents",
                "relative_weight",
                "vol_target",
                "current_amount",
                "target_amount",
                "amount_buy_sell",
                "status",
            ]
        )

    def _format_dtypes_alloc(self, df: pl.DataFrame) -> pl.DataFrame:
        """ """

        return df.with_columns(
            [
                pl.col([f"{self.date_column}_t_m_1", self.date_column]).cast(
                    pl.Datetime, strict=False
                ),
                pl.col([self.asset_id_column, "status"]).cast(pl.Utf8, strict=False),
                pl.col(
                    [
                        f"{self.price_column}_t_m_1",
                        self.price_column,
                        f"{self.portfolio_column}_t_m_1",
                        f"{self.portfolio_column}_t_p_1",
                        "nr_constituents_t_m_1",
                        self.portfolio_column,
                        "nr_constituents",
                        "relative_weight",
                        "current_amount",
                        "target_amount",
                        "amount_buy_sell",
                    ]
                ).cast(pl.Float32, strict=False),
            ]
        )

    def _format_dtypes_query(self, df: pl.DataFrame) -> pl.DataFrame:
        """ """

        return df.with_columns(
            [
                pl.col(self.date_column).cast(pl.Datetime, strict=False),
                pl.col(self.asset_id_column).cast(pl.Utf8, strict=False),
                pl.col([self.price_column, self.portfolio_column]).cast(
                    pl.Float32, strict=False
                ),
            ]
        )
