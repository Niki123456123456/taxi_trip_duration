import numpy as np
import requests
from numpy.typing import ArrayLike

EARTH_RADIUS_KM: float = 6_378.137
__all__ = ["EARTH_RADIUS_KM", "haversine"]


def haversine(
    lat1: ArrayLike,
    lon1: ArrayLike,
    lat2: ArrayLike,
    lon2: ArrayLike,
    radius: float = EARTH_RADIUS_KM
) -> np.ndarray:
  """
  Vectorised great‑circle distance between two points using the
  haversine formula.
  Parameters
  ----------
  lat1, lon1, lat2, lon2
      Coordinate pairs in **decimal degrees**.  Can be floats, NumPy arrays,
      or pandas Series – they are converted with ``np.asarray`` and support
      broadcasting.
  radius
      Sphere radius in kilometres.  Defaults to the WGS‑84 equatorial value
      (6378.137km).
      Returns
  -------
  np.ndarray
      Distance(s) in kilometres with the same broadcasted shape as the
      inputs.
  """
  # Ensure NumPy arrays for broadcasting then convert to radians
  lat1, lon1, lat2, lon2 = map(
      np.radians, map(np.asarray, (lat1, lon1, lat2, lon2))
  )
  dlat = lat2 - lat1
  dlon = lon2 - lon1
  a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(
      dlon / 2.0
  ) ** 2
  return 2.0 * radius * np.arcsin(np.sqrt(a))


OSRM_URL = "http://localhost:5001/route/v1/driving/"


def osrm_distance_km(pick_lon, pick_lat, drop_lon, drop_lat):
#   url = (f"{OSRM_URL}"
#          f"{pick_lon},{pick_lat};{drop_lon},{drop_lat}"
#          "?overview=false")
#   r = requests.get(url, timeout=2)
#   if r.ok and r.json()["code"] == "Ok":
#     return r.json()["routes"][0]["distance"] / 1000.0  # metres → km
#   return np.nan
    return haversine(pick_lat, pick_lon, drop_lat, drop_lon)  # metres → km
