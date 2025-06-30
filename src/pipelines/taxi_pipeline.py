from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import constants.taxi_c as taxi_constants
from constants.modelling_c import RANDOM_STATE
from constants.taxi_c import (PassengerLimits as pl, GeoBounds as gb,
                              RAW_TIME_COL,
                              TAXI_PROCESSED_CSV, n_pickup_clusters, batch_size,
                              n_dropoff_clusters)
from data_io import load_taxi_data
from features.taxi import create_geo_clusters, create_route_distance, \
  get_jfk_flag, get_lgua
from features import taxi as feat_taxi
from features import utils as feat_utils


def build_taxi_dataset(save_csv: bool = False) -> pd.DataFrame:
  """
  Build the cleaned+ feature‑enriched NYC‑taxi DataFrame.

  Parameters
  ----------
  save_csv : bool, default ``False``
      When *True*, the function writes the resulting dataset to the path
      defined in ``constants.taxi_const.TAXI_PROCESSED_CSV`` (overwriting any
      existing file).

  Returns
  -------
  pandas.DataFrame
      A copy of the raw taxi data after basic cleaning, outlier flagging,
      feature engineering (time‑features, Haversine distance, holiday flag,
      etc.) and ready for modelling.

  Notes
  -----
  * The transformation is **pure** except for the optional CSV write side
    effect triggered by ``save_csv=True``.
  * The function does **not** mutate the cached raw DataFrame returned by
    :pyfunc:`data_io.load_taxi_data`; all operations are performed on a copy.
  """

  # 1 Raw Ingest  ----------------------------------------

  df = load_taxi_data()

  # 2 ─ Timestamps  ----------------------------------------
  df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors="coerce"
                                         )
  df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'],
                                          errors="coerce"
                                          )

  # 3 Outlier-Flags / Clipping  ----------------------------------------
  outlier = (
    ("passenger_count", "passenger_count_invalid", pl.min_passengers,
     pl.max_passengers),
    ("pickup_longitude", "pickup_longitude_invalid", gb.min_lon, gb.max_lon),
    ("pickup_latitude", "pickup_latitude_invalid", gb.min_lat, gb.max_lat),
    ("dropoff_longitude", "dropoff_longitude_invalid", gb.min_lon, gb.max_lon),
    ("dropoff_latitude", "dropoff_latitude_invalid", gb.min_lat, gb.max_lat),
    ("trip_duration", "trip_duration_outlier",
     taxi_constants.TripDurationLimits.min_sec,
     taxi_constants.TripDurationLimits.max_sec),
  )

  df["is_group_trip"] = (df["passenger_count"] >= 2).astype("int8")

  for src, flag, lo, hi in outlier:
    df = feat_utils.flag_and_clip(df, src, flag, lo, hi)

  # 4 feature engineering ----------------------------------------
  df = feat_taxi.add_store_and_fwd_flag(df)
  df = feat_taxi.add_us_holiday_flag(df, 'pickup_datetime')
  df = feat_taxi.add_trip_duration_features(df)
  df = feat_taxi.add_haversine(df)

  df = create_route_distance(df)

  df = feat_taxi.add_time_features(df, RAW_TIME_COL)

  df['route_distance_log_km'] = np.log1p(df['route_distance_km'])

  df = create_geo_clusters(df, ['pickup_longitude', 'pickup_latitude'],
                           'pickup', n_pickup_clusters, RANDOM_STATE,
                           batch_size)
  df = create_geo_clusters(df, ['dropoff_longitude', 'dropoff_latitude'],
                           'dropoff', n_dropoff_clusters, RANDOM_STATE,
                           batch_size)
  df = get_jfk_flag(df)
  df = get_lgua(df)

  # 5 write CSV ---------------------------------------------------
  if save_csv:
    Path(TAXI_PROCESSED_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAXI_PROCESSED_CSV, index=False)

  return df
