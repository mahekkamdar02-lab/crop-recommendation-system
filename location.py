"""Location -> features. Fetches climate (NASA POWER) and soil pH (ISRIC SoilGrids) for a lat/lon.

NOTE: written from API documentation and not tested against the live services in the
authoring environment. If a call fails, the app falls back to manual input.
"""
import calendar
import requests

POWER_URL = "https://power.larc.nasa.gov/api/temporal/climatology/point"
SOIL_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
MONTHS = [m.upper() for m in calendar.month_abbr[1:]]  # JAN..DEC
_DAYS = [calendar.monthrange(2023, i)[1] for i in range(1, 13)]  # non-leap month lengths


def parse_power(payload, months):
    """Seasonal mean temperature (C), relative humidity (%) and mean MONTHLY rainfall (mm)."""
    p = payload["properties"]["parameter"]
    idx = [MONTHS.index(m) for m in months]
    temp = [p["T2M"][MONTHS[i]] for i in idx]
    rh = [p["RH2M"][MONTHS[i]] for i in idx]
    rain = [p["PRECTOTCORR"][MONTHS[i]] * _DAYS[i] for i in idx]  # mm/day -> mm/month
    if any(v is None or v <= -990 for v in temp + rh + rain):
        raise ValueError("NASA POWER returned missing values for this location/season")
    n = len(idx)
    return {
        "temperature": round(sum(temp) / n, 1),
        "humidity": round(sum(rh) / n, 1),
        "rainfall": round(sum(rain) / n, 1),
    }


def get_climate(lat, lon, months):
    params = {
        "parameters": "T2M,PRECTOTCORR,RH2M",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "format": "JSON",
    }
    r = requests.get(POWER_URL, params=params, timeout=40)
    r.raise_for_status()
    return parse_power(r.json(), months)


def parse_soil_ph(payload):
    """Mean pH over the returned depth layers (SoilGrids stores pH x 10; d_factor handles it)."""
    for layer in payload["properties"]["layers"]:
        if layer["name"] == "phh2o":
            d = layer["unit_measure"].get("d_factor", 10) or 10
            vals = [x["values"]["mean"] for x in layer["depths"] if x["values"].get("mean") is not None]
            if not vals:
                raise ValueError("No soil pH value for this location (water body or no data)")
            return round(sum(vals) / len(vals) / d, 2)
    raise ValueError("phh2o layer missing in SoilGrids response")


def get_soil_ph(lat, lon):
    params = [
        ("lon", lon), ("lat", lat), ("property", "phh2o"),
        ("depth", "0-5cm"), ("depth", "5-15cm"), ("value", "mean"),
    ]
    r = requests.get(SOIL_URL, params=params, timeout=60)
    r.raise_for_status()
    return parse_soil_ph(r.json())


def get_features(lat, lon, months):
    """Best-effort fetch. Returns (features dict, list of warnings)."""
    feats, warnings = {}, []
    try:
        feats.update(get_climate(lat, lon, months))
    except Exception as e:  # network, API change, missing data
        warnings.append(f"Climate lookup failed: {e}")
    try:
        feats["ph"] = get_soil_ph(lat, lon)
    except Exception as e:
        warnings.append(f"Soil pH lookup failed: {e}")
    return feats, warnings
