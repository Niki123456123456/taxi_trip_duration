from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.cluster import MiniBatchKMeans
from tqdm.auto import tqdm

from features.distance import haversine, osrm_distance_km


def _check_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
  missing = [c for c in cols if c not in df.columns]
  if missing:
    raise KeyError(f"Missing column(s) {missing}")


# ------------------------------------------------------------------ #
def add_us_holiday_flag(
    df: pd.DataFrame,
    dt_col: str,
    flag_col: str = "is_holiday",
) -> pd.DataFrame:
  """
  Add boolean *flag_col* that is **True** on U.S. federal holidays.

  Parameters
  ----------
  df : DataFrame
      Input taxi records.
  dt_col : str
      Name of the datetime column (must be timezone-aware or in local NYC time).
  flag_col : str, default ``"is_holiday"``
      Name of the new column.

  Returns
  -------
  DataFrame
      Copy of *df* with one additional boolean column.
  """
  _check_columns(df, [dt_col])
  df = df.copy()

  df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")

  cal = USFederalHolidayCalendar()
  holidays = cal.holidays(
      start=df[dt_col].min().normalize(),
      end=df[dt_col].max().normalize(),
  )
  df[flag_col] = df[dt_col].dt.normalize().isin(holidays).astype("int8")
  return df


def add_time_features(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
  """Derive hour, weekday, month, hour-of-year and weekend flag."""
  _check_columns(df, [dt_col])
  df = df.copy()
  ts = df[dt_col].dt

  df["pickup_hour"] = ts.hour.astype("int8")
  df["pickup_weekday"] = ts.dayofweek.astype("int8")
  df["pickup_month"] = ts.month.astype("int8")
  df["hour_of_year"] = ((ts.dayofyear - 1) * 24 + ts.hour).astype("int16")
  df["is_weekend"] = (ts.dayofweek >= 5).astype("int8")

  df["is_early_morning"] = df["pickup_hour"].between(3, 5, 'both').astype(
      "int8")

  df["is_rush_am"] = df["pickup_hour"].between(6, 8, 'both').astype("int8")

  df["is_rush_pm"] = df["pickup_hour"].between(16, 18, 'both').astype("int8")

  df["is_night"] = (
      (df["pickup_hour"] < 3) |
      (df["pickup_hour"] >= 22)
  ).astype("int8")
  return df


def add_trip_duration_features(df: pd.DataFrame) -> pd.DataFrame:
  """Add duration in minutes and log-seconds (← skew reduction)."""
  _check_columns(df, ["trip_duration"])
  df = df.copy()
  df["trip_duration_min"] = (df["trip_duration"] / 60.0).astype("float32")
  df["trip_duration_log"] = np.log1p(df["trip_duration"]).astype("float32")
  return df


def add_haversine(df: pd.DataFrame) -> pd.DataFrame:
  """Add haversine distance in km + log1p(km)."""
  needed = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
  ]
  _check_columns(df, needed)
  df = df.copy()
  df["hav_dist_km"] = haversine(
      df["pickup_latitude"],
      df["pickup_longitude"],
      df["dropoff_latitude"],
      df["dropoff_longitude"],
  ).astype("float32")
  df["hav_dist_km_log"] = np.log1p(df["hav_dist_km"]).astype("float32")
  return df


def add_store_and_fwd_flag(
    df: pd.DataFrame,
    src_col: str = "store_and_fwd_flag",
    dest_col: str = "store_and_fwd_flag_bin",
) -> pd.DataFrame:
  """
  Convert ``"Y"``/``"N"`` string flag to 1/0 tiny integer.
  Unknown values become NaN (Int8 can hold that).
  """
  _check_columns(df, [src_col])
  df = df.copy()
  mapping = {"Y": 1, "N": 0}
  df[dest_col] = df[src_col].map(mapping).astype("Int8")
  return df


def create_geo_clusters(df, feature_cols, prefix, n_clusters, random_state,
    batch_size):
  coords = df[feature_cols]
  kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                           random_state=random_state,
                           batch_size=batch_size)
  cluster_labels = kmeans.fit_predict(coords)
  df[f'{prefix}_cluster'] = pd.Series(cluster_labels, index=df.index).astype(
      np.int32)
  return df


def create_route_distance(df):
  tqdm.pandas()
  out = "../data/derived/with_route_dist.parquet"
  if os.path.exists(out):
    done = pd.read_parquet(out)
    start_ix = done.shape[0]
    df_remaining = df.iloc[start_ix:].copy()
  else:
    start_ix = 0
    df_remaining = df.copy()
    done = pd.DataFrame(columns=df.columns.tolist() + ["route_distance_km"])
  tqdm.pandas()
  chunk = 50_000
  for i in range(0, len(df_remaining), chunk):
    sub = df_remaining.iloc[i:i + chunk]
    sub["route_distance_km"] = sub.progress_apply(
        lambda row: osrm_distance_km(
            row.pickup_longitude, row.pickup_latitude,
            row.dropoff_longitude, row.dropoff_latitude
        ), axis=1
    ).astype("float32")

    done = pd.concat([done, sub], axis=0)
    done.to_parquet(out, index=False)
  df = done
  return df


def get_lgua(df):
  global lon, lat
  # Convert columns to NumPy arrays for pickups
  lon = df["pickup_longitude"].to_numpy()
  lat = df["pickup_latitude"].to_numpy()
  # Define La Guardia boundaries (approximate)
  LA_LON = (-73.894, -73.861)
  LA_LAT = (40.774, 40.765)
  # Vectorized flag for La Guardia pickups
  df["is_laguardia_pick"] = (
      (LA_LON[0] <= lon) & (lon <= LA_LON[1]) &
      (LA_LAT[0] <= lat) & (lat <= LA_LAT[1])
  ).astype("int8")
  # Repeat for drop-offs (update lon/lat arrays)
  lon = df["dropoff_longitude"].to_numpy()
  lat = df["dropoff_latitude"].to_numpy()
  df["is_laguardia_drop"] = (
      (LA_LON[0] <= lon) & (lon <= LA_LON[1]) &
      (LA_LAT[0] <= lat) & (lat <= LA_LAT[1])
  ).astype("int8")
  return df


def get_jfk_flag(df):
  global lon, lat
  # Convert columns to NumPy arrays
  lon = df["pickup_longitude"].to_numpy()
  lat = df["pickup_latitude"].to_numpy()
  # Define JFK boundaries
  JFK_LON = (-73.837, -73.745)  # widen by 0.01° east-west
  JFK_LAT = (40.622, 40.675)  # widen by 0.01° south-north
  # Vectorized flag for JFK pickups
  df["is_jfk_pick"] = (
      (JFK_LON[0] <= lon) & (lon <= JFK_LON[1]) &
      (JFK_LAT[0] <= lat) & (lat <= JFK_LAT[1])
  ).astype("int8")
  # Repeat for drop-offs (update lon/lat arrays)
  lon = df["dropoff_longitude"].to_numpy()
  lat = df["dropoff_latitude"].to_numpy()
  df["is_jfk_drop"] = (
      (JFK_LON[0] <= lon) & (lon <= JFK_LON[1]) &
      (JFK_LAT[0] <= lat) & (lat <= JFK_LAT[1])
  ).astype("int8")
  return df
