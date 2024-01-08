import polars as pl
import datetime as dt
from typing import List
from dataclasses import dataclass


@dataclass
class hc_args:
    """Class containing args for the hierarchical clustering part"""

    distance_method: str
    linkage_method: str
    weights_method: str
    weight_max: float = 1.0
    weight_min: float = 0.0


@dataclass
class hrp_corr_args:
    """Class containing correlation matrix args"""

    n_month_delay: int
    exp_alpha: float = 0
    corr_method: str = "pearson"


@dataclass
class hrp_cov_args:
    """Class containing covariance matrix args"""

    n_month_delay: int
    exp_alpha: float = 0
    shrinkage_method: str = None


@dataclass
class hrp_rets_args:
    """Class containing returns arguments for HRPP method"""

    n_month_delay: int
    ret_rolling_window: int
    risk_aversion: float


@dataclass
class hrp_vol_pred_args:
    """Class containing args for the  HRPF method"""

    df_vol_pred: pl.DataFrame
    pred_aversion: float


@dataclass
class current_date_infos:
    """Class containing args for the  HRPF method"""

    df_reference: pl.DataFrame
    df_pred: pl.DataFrame
    rebal_date: dt.datetime
    rebal_assets: List[str]
