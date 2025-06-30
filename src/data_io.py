import logging
import pickle
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from kaggle import KaggleApi

from constants import taxi_c, weather_c, core_c

log = logging.getLogger(__name__)


def download_kaggle_competition(api, competition_name, path):
  try:
    api.authenticate()
  except Exception as e:
    raise RuntimeError("Kaggle API authentication failed. "
                       "Ensure that ~/.kaggle/kaggle.json exists and is properly configured.") from e
  api.competition_download_files(competition_name, path=path)


def download_kaggle_dataset(api, dataset_slug, path):
  try:
    api.authenticate()
  except Exception as e:
    raise RuntimeError("Kaggle API authentication failed. "
                       "Ensure that ~/.kaggle/kaggle.json exists and is properly configured.") from e
  api.dataset_download_files(dataset_slug, path=str(path), unzip=False)


def extract_inner_zips(zip_path, data_dir, required_inner_zips):
  existing_inner_zips = {z.name for z in data_dir.glob("*.zip")}
  missing_inner_zips = required_inner_zips - existing_inner_zips

  if missing_inner_zips:
    with ZipFile(zip_path, 'r') as outer_zip:
      outer_zip.extractall(data_dir)


def extract_csv_from_zip(zip_path: Path, csv_name: str, target_dir: Path):
  target_dir.mkdir(parents=True, exist_ok=True)

  with ZipFile(zip_path) as zf:
    names = zf.namelist()

    if csv_name in names:
      member = csv_name
    else:
      matches = [n for n in names if Path(n).name == csv_name]
      if not matches:
        raise KeyError(
            f"{csv_name!r} nicht im Archiv. Enthalten: {names[:5]} …"
        )
      member = matches[0]

    zf.extract(member, path=target_dir)

    extracted = target_dir / member
    extracted.rename(target_dir / csv_name)


def extract_csv_from_inner_zips(data_dir, extracted_dir):
  extracted_dir.mkdir(parents=True, exist_ok=True)
  for inner_zip in data_dir.glob("*.zip"):
    with ZipFile(inner_zip, 'r') as zip_ref:
      for member in zip_ref.namelist():
        if member.endswith(".csv"):
          target_file = extracted_dir / Path(member).name
          if not target_file.is_file():
            with zip_ref.open(member) as src, open(target_file, "wb") as dst:
              dst.write(src.read())


def load_taxi_data():
  log.info("Loading NYC taxi raw data …")

  packed_file_path = taxi_c.TAXI_RAW_ZIP
  csv_path = taxi_c.TAXI_RAW_CSV
  pkl_path = taxi_c.TAXI_CACHE_PICKLE

  if pkl_path.is_file():
    with open(pkl_path, "rb") as f:
      return pickle.load(f)

  if not packed_file_path.is_file():
    core_c.ZIP_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    download_kaggle_competition(api, "nyc-taxi-trip-duration",
                                core_c.ZIP_DIR)

  inner_zip_names = {"train.zip", "test.zip", "sample_submission.zip"}
  extract_inner_zips(packed_file_path, core_c.ZIP_DIR, inner_zip_names)

  extract_csv_from_inner_zips(core_c.ZIP_DIR, core_c.RAW_DIR)

  if not csv_path.is_file():
    raise FileNotFoundError(
        f"'{csv_path}' was not found - unpacking failed.")

  df = pd.read_csv(csv_path)

  if not pkl_path.is_file():
    core_c.CACHE_DIR.mkdir(parents=True, exist_ok=True)
  with open(pkl_path, "wb") as f:
    pickle.dump(df, f)

  return df


def load_weather_data() -> pd.DataFrame:
  log.info("Loading NYC weather raw data …")

  dataset_slug = "pschale/nyc-taxi-wunderground-weather"
  packed_file_path = weather_c.WEATHER_RAW_ZIP
  csv_path = weather_c.WEATHER_RAW_CSV1
  csv_name = csv_path.name

  pkl_path = weather_c.WEATHER_CACHE_PICKLE

  if pkl_path.is_file():
    with open(pkl_path, "rb") as f:
      return pickle.load(f)

  if not packed_file_path.is_file():
    core_c.ZIP_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    download_kaggle_dataset(api, dataset_slug, core_c.ZIP_DIR)
    log.info("Weather ZIP downloaded.")

  if not csv_path.is_file():
    extract_csv_from_zip(packed_file_path, csv_name, core_c.RAW_DIR)
    log.info("Weather CSV extracted.")

  if not csv_path.is_file():
    raise FileNotFoundError(f"{csv_path} not found after extraction.")

  df = pd.read_csv(csv_path)

  if not pkl_path.is_file():
    core_c.CACHE_DIR.mkdir(parents=True, exist_ok=True)
  with open(pkl_path, "wb") as f:
    pickle.dump(df, f)

  return df
