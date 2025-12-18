from influxdb import InfluxDBClient
from datetime import datetime, timedelta, timezone
import numpy as np

INFLUX_HOST = "localhost"
INFLUX_PORT = 8086
INFLUX_DB = "doniczka"

FIELD = "wartosc"  # u Ciebie w Influx jest dokładnie "wartosc"

client = InfluxDBClient(
    host=INFLUX_HOST,
    port=INFLUX_PORT,
    database=INFLUX_DB
)


def _to_influx_time(dt: datetime) -> str:
    """
    InfluxDB v1 InfluxQL lubi timestampy w ISO z 'Z' (UTC).
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def fetch_series(measurement: str, since_dt: datetime) -> list[float]:
    since = _to_influx_time(since_dt)
    q = f'SELECT "{FIELD}" FROM "{measurement}" WHERE time > \'{since}\''
    result = client.query(q)
    points = list(result.get_points())
    return [float(p[FIELD]) for p in points if FIELD in p]


def build_sensor_input() -> dict:
    now = datetime.now(timezone.utc)

    # --- zakresy czasowe ---
    since_6h = now - timedelta(hours=6)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # --- dane ---
    light_6h = fetch_series("light", since_6h)
    soil_6h = fetch_series("soil_capacitance", since_6h)
    temp_6h = fetch_series("temperature", since_6h)
    light_today = fetch_series("light", today_start)

    if len(light_6h) == 0 or len(soil_6h) == 0 or len(temp_6h) == 0:
        raise ValueError("Brak danych z InfluxDB (light/soil/temperature) w ostatnich 6h")

    # --- feature engineering ---
    light_now = light_6h[-1]
    rolling_avg_light_6h = float(np.mean(light_6h))

    # ile godzin dziś było jasno (liczone z próbek co 5 min)
    BRIGHT_THRESHOLD = 1000.0
    bright_samples = sum(l >= BRIGHT_THRESHOLD for l in light_today)
    light_hours_today = bright_samples * 5 / 60.0

    delta_soil = float(soil_6h[-1] - soil_6h[0])

    # uproszczona logika "ile godzin od podlania" po skoku wilgotności
    WATER_SPIKE = 8.0
    hours_since_last_watering = 999.0
    for i in range(len(soil_6h) - 1, 0, -1):
        if (soil_6h[i] - soil_6h[i - 1]) >= WATER_SPIKE:
            hours_since_last_watering = (len(soil_6h) - 1 - i) * 5 / 60.0
            break

    return {
        "light": float(light_now),
        "temperature": float(np.mean(temp_6h)),
        "soil_capacitance": float(soil_6h[-1]),
        "hours_since_last_watering": float(hours_since_last_watering),
        "delta_soil": float(delta_soil),
        "rolling_avg_light_6h": float(rolling_avg_light_6h),
        "light_hours_today": float(light_hours_today),
        "hour_of_day": int(now.hour),
    }
