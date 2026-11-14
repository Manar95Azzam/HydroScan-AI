"""
app.py -- HydroScan: two tools in one app, chosen with the router at the top of the sidebar.

Run with:   python -m streamlit run app.py
Install:    pip install streamlit streamlit-folium folium rasterio scipy scikit-learn shapely earthengine-api requests

  Irrigation Detection     draw a box -> real Sentinel-2 detection (Earth Engine) -> map overlay, details,
                           optional VV/VH/NDVI + trained-model analysis, NASA context cards.
                           (All detection / GEE / map / metric code is unchanged; it now lives in
                           irrigation_detection_view().)
  Crop Rotation Strategy   reads the box and date chosen in the first view, pulls NASA GPM / SMAP / POWER
                           (live when reachable, otherwise a deterministic climatology), and recommends ONE
                           rotation strategy with an explanation. No random numbers anywhere.
"""

import datetime as dt
import math
import os
import tempfile
import zlib

import folium
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from branca.element import MacroElement, Template
from folium.plugins import Draw
from rasterio.features import shapes
from rasterio.transform import from_bounds
from scipy import ndimage
from streamlit_folium import st_folium

from model_training import NODATA, make_features

try:  # optional: smooth polygons. Without it the app falls back to a raster overlay.
    from shapely.geometry import mapping, shape
    HAVE_SHAPELY = True
except ImportError:  # pragma: no cover
    HAVE_SHAPELY = False

ACRE_M2 = 4046.8564224
DEFAULT_BBOX = (30.40, 30.50, 30.45, 30.55)  # west, south, east, north
MAX_POLYGONS = 1500
BASIS_PEAK = "Peak greenness over the period (best for irrigation)"
BASIS_MEDIAN = "Median (a typical snapshot)"

st.set_page_config(page_title="Farm Monitor", page_icon="💧", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; max-width: 1400px;}
      div[data-testid="stMetric"] {background:#fff; border:1px solid #e3ebf7; border-radius:14px;
             padding:.8rem 1rem; box-shadow:0 4px 12px rgba(20,60,140,.08);}
      div[data-testid="stMetricValue"] {color:#0b3d91;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Keep sidebar widget values alive while the other tool is shown (Streamlit drops widgets that are not rendered).
for _k in ("west", "south", "east", "north", "project", "start", "days", "basis", "ndvi_thr", "excl_water",
           "min_acres", "sat_start", "model_path"):
    if _k in st.session_state:
        st.session_state[_k] = st.session_state[_k]

# Step 2: router at the very top of the sidebar
view = st.sidebar.radio("Choose Tool:", ["Irrigation Detection", "Crop Rotation Strategy"])

# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def metres_per_degree(south, north):
    return 111_320 * math.cos(math.radians((south + north) / 2)), 110_540


def bbox_from_drawings(drawings):
    """Last drawn rectangle from st_folium's all_drawings -> (W, S, E, N) or None."""
    for d in reversed(drawings or []):
        geom = (d or {}).get("geometry") or {}
        if geom.get("type") == "Polygon":
            ring = geom["coordinates"][0]
            lons, lats = [p[0] for p in ring], [p[1] for p in ring]
            return (round(min(lons), 6), round(min(lats), 6), round(max(lons), 6), round(max(lats), 6))
    return None


# --------------------------------------------------------------------------- #
# REAL detection: Sentinel-2 NDVI/NDWI for the box (Earth Engine) -> irrigated polygons
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=6 * 3600)
def fetch_indices(project, bbox, start_iso, days, peak):
    """Download a cloud-masked Sentinel-2 NDVI (+ NDWI) composite for bbox."""
    import gee_pipeline as gp

    ee = gp.ee
    ee.Initialize(project=project)
    region = gp.make_region(list(bbox))
    s, e = gp.date_range(start_iso, days)
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region).filterDate(s, e)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40)))
    n = col.size().getInfo()
    if n == 0:
        raise RuntimeError("No cloud-free Sentinel-2 scenes in this period. Try a longer period or other dates.")

    def indices(img):
        r = img.select(["B3", "B4", "B8"]).divide(10000)
        return (r.normalizedDifference(["B8", "B4"]).rename("NDVI")
                .addBands(r.normalizedDifference(["B3", "B8"]).rename("NDWI")))

    per_image = col.map(gp.mask_s2_clouds).map(indices)
    if peak:   # 90th-percentile greenness: robust to clouds, finds crops green at ANY time in the period
        ndvi = per_image.select("NDVI").reduce(ee.Reducer.percentile([90])).rename("NDVI")
    else:
        ndvi = per_image.select("NDVI").median()
    ndwi = per_image.select("NDWI").median()
    out = ndvi.addBands(ndwi).clip(region).toFloat().unmask(NODATA)

    mx, my = metres_per_degree(bbox[1], bbox[3])
    side_m = max((bbox[2] - bbox[0]) * mx, (bbox[3] - bbox[1]) * my)
    scale = max(10, math.ceil(side_m / 2000))  # keeps the download under Earth Engine's size limit

    path = os.path.join(tempfile.gettempdir(), "hydroscan_box.tif")
    gp.download_geotiff(out, region, list(bbox), path, scale=scale, n_bands=2)
    with rasterio.open(path) as ds:
        arr = ds.read().astype("float32")
        bounds = tuple(ds.bounds)
    arr[arr == NODATA] = np.nan
    return {"ndvi": arr[0], "ndwi": arr[1], "bounds": bounds, "scale": scale, "scenes": int(n),
            "period": (s, e), "peak": bool(peak)}


def classify(ind, thr, excl_water, min_acres):
    """NDVI/NDWI arrays -> cleaned irrigated mask, polygons (GeoJSON), patch table and stats."""
    ndvi, ndwi = ind["ndvi"], ind["ndwi"]
    H, W = ndvi.shape
    west, south, east, north = ind["bounds"]
    mx, my = metres_per_degree(south, north)
    px_area = ((east - west) / W * mx) * ((north - south) / H * my)

    valid = np.isfinite(ndvi)
    mask = valid & (np.nan_to_num(ndvi, nan=-1) >= thr)
    if excl_water:
        mask &= ~(np.nan_to_num(ndwi, nan=-1) > 0.1)
    mask = ndimage.binary_closing(mask, np.ones((3, 3), bool))   # fill pinholes
    mask = ndimage.binary_opening(mask, np.ones((3, 3), bool))   # remove speckle
    lab, n = ndimage.label(mask)
    if n:
        sizes = ndimage.sum(mask, lab, np.arange(1, n + 1))
        keep = np.where(sizes * px_area / ACRE_M2 >= min_acres)[0] + 1
        mask = np.isin(lab, keep)
        lab, n = ndimage.label(mask)

    # patch table (works with or without shapely)
    patches = []
    if n:
        ids = np.arange(1, n + 1)
        cnt = ndimage.sum(mask, lab, ids)
        cy, cx = zip(*ndimage.center_of_mass(mask, lab, ids))
        for i in np.argsort(cnt)[::-1][:MAX_POLYGONS]:
            patches.append({"patch": len(patches) + 1, "acres": round(float(cnt[i] * px_area / ACRE_M2), 1),
                            "mean NDVI": round(float(np.nanmean(ndvi[lab == ids[i]])), 2),
                            "lat": round(north - (cy[i] + 0.5) / H * (north - south), 5),
                            "lon": round(west + (cx[i] + 0.5) / W * (east - west), 5)})

    features = []
    if HAVE_SHAPELY and mask.any():
        tol = 0.6 * (east - west) / W
        polys = []
        for geom, val in shapes(mask.astype("uint8"), mask=mask, transform=from_bounds(west, south, east, north, W, H)):
            if val != 1:
                continue
            g = shape(geom).simplify(tol, preserve_topology=True)
            if not g.is_empty:
                polys.append((g.area * mx * my / ACRE_M2, g))
        polys.sort(key=lambda t: t[0], reverse=True)
        for i, (acres, g) in enumerate(polys[:MAX_POLYGONS]):
            features.append({"type": "Feature", "properties": {"field": i + 1, "acres": round(acres, 1)},
                             "geometry": mapping(g)})

    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[mask] = (30, 110, 255, 150)
    nd_rgba = (matplotlib.colormaps["RdYlGn"](np.clip((np.nan_to_num(ndvi, nan=-0.1) + 0.1) / 1.0, 0, 1)) * 255).astype(np.uint8)
    nd_rgba[..., 3] = np.where(valid, 170, 0)
    hist, edges = np.histogram(ndvi[valid], bins=np.round(np.arange(-0.2, 1.01, 0.05), 2)) if valid.any() else ([], [])
    return {
        "geojson": {"type": "FeatureCollection", "features": features} if features else None,
        "rgba": rgba, "ndvi_rgba": nd_rgba, "bounds": ind["bounds"], "ndvi": ndvi, "mask": mask,
        "hist": (hist, edges), "patches": patches,
        "irrigated_acres": float(mask.sum() * px_area / ACRE_M2),
        "total_acres": float(valid.sum() * px_area / ACRE_M2),
        "n_fields": int(n),
        "ndvi_in": float(np.nanmean(ndvi[mask])) if mask.any() else float("nan"),
        "ndvi_all": float(np.nanmean(ndvi)) if valid.any() else float("nan"),
    }


# --------------------------------------------------------------------------- #
# DEMO fallback (random polygons) when no Earth Engine project is entered
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def demo_area(bbox):
    west, south, east, north = bbox
    rng = np.random.default_rng(zlib.crc32(repr(bbox).encode()))
    w, h = east - west, north - south
    mx, my = metres_per_degree(south, north)
    feats, irr = [], 0.0
    for i in range(int(rng.integers(5, 10))):
        cx, cy = west + w * rng.uniform(0.12, 0.88), south + h * rng.uniform(0.12, 0.88)
        r = min(w, h) * rng.uniform(0.07, 0.18)
        ang = np.sort(rng.uniform(0, 2 * np.pi, int(rng.integers(12, 20))))
        rad = r * rng.uniform(0.65, 1.0, len(ang))
        xs, ys = np.clip(cx + 1.3 * rad * np.cos(ang), west, east), np.clip(cy + rad * np.sin(ang), south, north)
        X, Y = (xs - west) * mx, (ys - south) * my
        a = 0.5 * abs(np.sum(X * np.roll(Y, -1) - np.roll(X, -1) * Y))
        irr += a
        ring = [[float(x), float(y)] for x, y in zip(xs, ys)]
        feats.append({"type": "Feature", "properties": {"field": i + 1, "acres": round(a / ACRE_M2, 1)},
                      "geometry": {"type": "Polygon", "coordinates": [ring + [ring[0]]]}})
    return {"geojson": {"type": "FeatureCollection", "features": feats}, "rgba": None, "ndvi_rgba": None,
            "bounds": bbox, "irrigated_acres": irr / ACRE_M2, "total_acres": w * mx * h * my / ACRE_M2,
            "n_fields": len(feats), "patches": [], "ndvi": None, "mask": None, "hist": ([], []),
            "ndvi_in": float("nan"), "ndvi_all": float("nan")}


# --------------------------------------------------------------------------- #
# Map: satellite by default, ONE rectangle at a time, no pop-up alerts.
# The BASE map never changes between runs, so the user's drawing and view are kept.
# --------------------------------------------------------------------------- #
class SingleRectangle(MacroElement):
    """Keeps only the newest drawn rectangle and removes the default click-to-alert."""

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        {{ this._parent.get_name() }}.on('draw:created', function(e) {
            var group = drawnItems_{{ this.draw_name }};
            e.layer.off('click');
            group.eachLayer(function(l) { if (l !== e.layer) { group.removeLayer(l); } });
        });
        {% endmacro %}
        """
    )

    def __init__(self, draw):
        super().__init__()
        self._name = "SingleRectangle"
        self.draw_name = draw.get_name()


def base_map():
    cx, cy = (DEFAULT_BBOX[0] + DEFAULT_BBOX[2]) / 2, (DEFAULT_BBOX[1] + DEFAULT_BBOX[3]) / 2
    m = folium.Map(location=[cy, cx], zoom_start=12, tiles=None, control_scale=True)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite photo (older, for reference)", max_zoom=19).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street map", show=False).add_to(m)
    draw = Draw(
        export=False, position="topleft",
        draw_options={
            "rectangle": {"shapeOptions": {"color": "#19c3e6", "weight": 2, "fillOpacity": 0.05}},
            "polyline": False, "polygon": False, "circle": False, "marker": False, "circlemarker": False,
        },
        edit_options={"edit": True, "remove": True},
    )
    draw.add_to(m)
    SingleRectangle(draw).add_to(m)
    return m


@st.cache_resource
def load_model(path):
    import joblib
    return joblib.load(path)


def show_map(ax, data, title, cmap, vmin=None, vmax=None):
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")

def nasa_labels(n):
    rain_lbl = "Very low" if n["rain"] <= 5 else "Moderate" if n["rain"] <= 15 else "High"
    temp_lbl = "Very hot" if n["temp"] >= 35 else "Hot" if n["temp"] >= 30 else "Warm" if n["temp"] >= 25 else "Mild"
    smap_lbl = "Very dry" if n["smap"] < 20 else "Moderate" if n["smap"] < 30 else "Moist"
    return rain_lbl, temp_lbl, smap_lbl


def recommend_rotations(n):
    """Deterministic rule-based engine -> (strategies, scores, ranking). Same inputs, same answer."""
    rain, temp, smap = n["rain"], n["temp"], n["smap"]
    rain_lbl, temp_lbl, smap_lbl = nasa_labels(n)
    clamp = lambda x: max(0.0, min(1.0, x))
    dryness = clamp((35 - smap) / 20)                  # 1 = very dry soil
    heat = clamp((temp - 25) / 15)                     # 1 = 40 C
    no_rain = clamp(1 - rain / 10)                     # 1 = no rain
    water_stress = 0.45 * dryness + 0.35 * heat + 0.20 * no_rain
    moist_mid = clamp(1 - abs(smap - 27) / 15)         # legumes establish best at moderate moisture
    temp_fit = clamp(1 - max(0.0, temp - 30) / 10)
    raw = (40 + 60 * water_stress,
           30 + 40 * moist_mid + 20 * (1 - heat) + 10 * clamp(rain / 10),
           20 + 55 * (1 - water_stress) ** 2 + 25 * temp_fit)
    scores = [int(round(min(100, s))) for s in raw]
    order = sorted(range(3), key=lambda i: (-raw[i], i))
    stress = "high" if water_stress > 0.6 else "moderate" if water_stress > 0.35 else "low"
    strategies = [
        {"name": "Water Conservation Strategy", "icon": "💧", "crops": ["Sorghum", "Barley", "Pearl millet"],
         "benefits": ["Lowest irrigation demand", "Tolerates heat and dry spells", "Keeps land productive when water is scarce"],
         "watch": "Lower market price and yield than wheat or maize.",
         "why": (f"Water stress is **{stress}**: soil moisture is {smap_lbl.lower()} ({smap:.0f}%), it is {temp_lbl.lower()} "
                 f"({temp:.1f} °C) and only {rain:.1f} mm of rain fell in 30 days. Drought-tolerant crops protect scarce water.")},
        {"name": "Soil Health Strategy", "icon": "🌱", "crops": ["Alfalfa", "Faba bean", "Chickpea", "Clover"],
         "benefits": ["Legumes fix nitrogen, so less fertiliser is needed", "Breaks cereal pest and disease cycles",
                      "Builds soil organic matter"],
         "watch": "Alfalfa is water-hungry; avoid it when water stress is high.",
         "why": (f"Soil moisture of {smap:.0f}% is "
                 f"{'in the range where legumes establish well' if moist_mid > 0.6 else 'not ideal for legume establishment, so irrigation must be managed'}, "
                 f"and {temp:.1f} °C is {'cool enough for most legumes' if temp < 32 else 'hot, so prefer cool-season legumes'}.")},
        {"name": "High Yield Strategy", "icon": "📈", "crops": ["Wheat", "Corn (maize)"],
         "benefits": ["Highest yield and income potential", "Strong, established markets", "Well-known farming practices"],
         "watch": "Highest water and fertiliser demand; heat at flowering cuts yield.",
         "why": (f"Water stress is **{stress}** and {temp:.1f} °C is {'inside' if temp <= 30 else 'above'} the comfortable range for "
                 f"wheat and corn, with {rain:.1f} mm of recent rain and {smap:.0f}% soil moisture supporting high-demand crops.")},
    ]
    return strategies, scores, order, water_stress


# --------------------------------------------------------------------------- #
# NASA data layer: GPM rainfall, SMAP soil moisture, POWER temperature.
#   * Live when reachable: NASA POWER REST API (temperature, rainfall) and, with an Earth Engine project,
#     NASA GPM IMERG (rainfall) and NASA SMAP L4 (soil moisture).
#   * Otherwise a DETERMINISTIC climatology estimate for the exact coordinates and dates.
# No random numbers are used anywhere. Each value carries its source so the UI can say what is real.
# --------------------------------------------------------------------------- #
import time

MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Approximate long-term monthly climatology (mean air temperature C, rainfall mm/month) from regional station
# normals. Add more regions here to improve the offline estimate.
REGION_PROFILES = [
    {"name": "Upper Egypt / Toshka (hyper-arid)", "lat": (22.0, 27.5), "lon": (29.0, 35.5),
     "temp": [16.5, 18.5, 22.5, 27.5, 31.5, 34.0, 34.0, 34.0, 32.0, 28.5, 22.5, 17.5],
     "rain": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.3, 0.3], "smap_base": 6.0},
    {"name": "Nile Valley & Delta (arid)", "lat": (27.5, 31.7), "lon": (25.0, 35.5),
     "temp": [14.0, 15.0, 17.5, 21.5, 25.0, 27.5, 28.5, 28.5, 26.5, 23.5, 19.5, 15.5],
     "rain": [5, 4, 4, 1, 0.5, 0, 0, 0, 0, 1, 3, 5], "smap_base": 12.0},
    {"name": "California Central Valley (Mediterranean)", "lat": (34.8, 40.5), "lon": (-123.0, -118.5),
     "temp": [8.5, 11.0, 14.0, 17.0, 21.5, 25.5, 28.0, 27.0, 24.0, 18.5, 12.0, 8.5],
     "rain": [55, 55, 45, 22, 10, 3, 0, 0, 5, 15, 30, 50], "smap_base": 16.0},
]
WINDOW_DAYS = 30


def climatology_profile(lat, lon, d0, ndays=WINDOW_DAYS):
    """Deterministic estimate for a 30-day window starting at d0 (same inputs -> same numbers)."""
    days = [d0 + dt.timedelta(days=i) for i in range(ndays)]
    prof = next((p for p in REGION_PROFILES
                 if p["lat"][0] <= lat <= p["lat"][1] and p["lon"][0] <= lon <= p["lon"][1]), None)
    if prof:
        temp = float(np.mean([prof["temp"][d.month - 1] for d in days]))
        rain = float(sum(prof["rain"][d.month - 1] / MONTH_DAYS[d.month - 1] for d in days))
        annual, base, label = sum(prof["rain"]), prof["smap_base"], prof["name"]
    else:  # coarse zonal model: latitude-dependent annual mean and seasonal swing
        peak = 200 if lat >= 0 else 15
        t0, amp = 27 - 0.0067 * lat ** 2, min(18.0, 0.23 * abs(lat))
        temp = float(np.mean([t0 + amp * math.cos(2 * math.pi * (d.timetuple().tm_yday - peak) / 365) for d in days]))
        a = abs(lat)
        monthly = 150 if a < 10 else 70 if a < 20 else 25 if a < 30 else 45 if a < 40 else 60 if a < 60 else 40
        rain, annual = monthly * ndays / 30.4, monthly * 12
        base = 7.0 if annual < 100 else 14.0 if annual < 400 else 22.0 if annual < 1000 else 30.0
        label = "coarse zonal climatology"
    smap = base + min(8.0, 0.15 * rain)
    return {"rain": round(rain, 1), "temp": round(temp, 1), "smap": round(smap, 0), "label": label}


@st.cache_data(show_spinner=False, ttl=3 * 3600)
def nasa_power_api(lat, lon, s_iso, e_iso):
    """NASA POWER daily point data: mean 2 m temperature and total corrected precipitation. Raises on failure."""
    import requests

    r = requests.get("https://power.larc.nasa.gov/api/temporal/daily/point", timeout=12, params={
        "parameters": "T2M,PRECTOTCORR", "community": "AG", "longitude": lon, "latitude": lat,
        "start": s_iso.replace("-", ""), "end": e_iso.replace("-", ""), "format": "JSON"})
    r.raise_for_status()
    par = r.json()["properties"]["parameter"]
    t = [v for v in par["T2M"].values() if v > -900]
    p = [v for v in par["PRECTOTCORR"].values() if v > -900]
    if not t:
        raise RuntimeError("no data for this date range (POWER lags a few days behind today)")
    return {"temp": float(np.mean(t)), "rain": float(np.sum(p)) if p else None, "days": len(t)}


@st.cache_data(show_spinner=False, ttl=3 * 3600)
def nasa_gee(project, lat, lon, s_iso, e_iso):
    """GPM IMERG rainfall (mm) and SMAP L4 surface soil moisture (%) through Earth Engine. Raises if nothing found."""
    import gee_pipeline as gp

    ee = gp.ee
    ee.Initialize(project=project)
    area = ee.Geometry.Point([lon, lat]).buffer(5000)
    ndays = (dt.date.fromisoformat(e_iso) - dt.date.fromisoformat(s_iso)).days
    errors = []

    def window_mean(candidates, scale):
        for cid, band in candidates:
            try:
                col = ee.ImageCollection(cid).filterDate(s_iso, e_iso).select(band)
                if col.size().getInfo() == 0:
                    continue
                v = col.mean().reduceRegion(ee.Reducer.mean(), area, scale).get(band).getInfo()
                if v is not None:
                    return float(v)
            except Exception as exc:  # try the next dataset version
                errors.append(f"{cid}: {str(exc)[:80]}")
        return None

    rate = window_mean([("NASA/GPM_L3/IMERG_V07", "precipitation"), ("NASA/GPM_L3/IMERG_V06", "precipitationCal")], 10000)
    sm = window_mean([("NASA/SMAP/SPL4SMGP/008", "sm_surface"), ("NASA/SMAP/SPL4SMGP/007", "sm_surface"),
                      ("NASA/SMAP/SPL4SMGP/006", "sm_surface")], 11000)
    if rate is None and sm is None:
        raise RuntimeError("; ".join(errors) or "no GPM / SMAP images for this date range")
    return {"rain": None if rate is None else rate * 24 * ndays, "smap": None if sm is None else sm * 100}


def get_nasa_profile(lat, lon, d0, project="", live=False):
    """Values for the 30 days starting at d0. live=False -> climatology only (instant, offline)."""
    lat, lon = round(float(lat), 2), round(float(lon), 2)
    base = climatology_profile(lat, lon, d0)
    est = f"climatology estimate ({base['label']})"
    out = {"rain": base["rain"], "temp": base["temp"], "smap": base["smap"],
           "src": {"rain": est, "temp": est, "smap": est},
           "live": {"rain": False, "temp": False, "smap": False}, "notes": []}
    if not live:
        return out
    s_iso = d0.isoformat()
    e_excl = d0 + dt.timedelta(days=WINDOW_DAYS)
    fails = st.session_state.setdefault("_nasa_fail", {})

    def blocked(key):  # do not hammer a failing service on every rerun
        return time.time() - fails.get(key, 0) < 300

    key = ("power", lat, lon, s_iso)
    if not blocked(key):
        try:
            p = nasa_power_api(lat, lon, s_iso, (e_excl - dt.timedelta(days=1)).isoformat())
            out["temp"], out["src"]["temp"], out["live"]["temp"] = round(p["temp"], 1), "NASA POWER API (live)", True
            if p["rain"] is not None:
                out["rain"], out["src"]["rain"], out["live"]["rain"] = round(p["rain"], 1), "NASA POWER precipitation (live)", True
        except Exception as exc:
            fails[key] = time.time()
            out["notes"].append(f"NASA POWER API unavailable: {str(exc)[:140]}")
    else:
        out["notes"].append("NASA POWER API unavailable (retrying in a few minutes).")

    if project:
        key = ("gee", project, lat, lon, s_iso)
        if not blocked(key):
            try:
                g = nasa_gee(project, lat, lon, s_iso, e_excl.isoformat())
                if g["rain"] is not None:
                    out["rain"], out["src"]["rain"], out["live"]["rain"] = round(g["rain"], 1), "NASA GPM IMERG via Earth Engine (live)", True
                if g["smap"] is not None:
                    out["smap"], out["src"]["smap"], out["live"]["smap"] = round(g["smap"], 0), "NASA SMAP L4 via Earth Engine (live)", True
            except Exception as exc:
                fails[key] = time.time()
                out["notes"].append(f"Earth Engine GPM/SMAP unavailable: {str(exc)[:140]}")
    else:
        out["notes"].append("Enter an Earth Engine project ID in the Irrigation Detection view to pull GPM and SMAP.")
    return out


def nasa_tag(n, key):
    return "live" if n["live"][key] else "estimate"


def nasa_for_irrigation(bbox, sat_start):
    """Instant values for the irrigation view: live ones if the rotation view already fetched them."""
    clat, clon = round((bbox[1] + bbox[3]) / 2, 2), round((bbox[0] + bbox[2]) / 2, 2)
    cached = st.session_state.get("_nasa_live", {}).get((clat, clon, sat_start.isoformat()))
    return cached or get_nasa_profile(clat, clon, sat_start, live=False)





# --------------------------------------------------------------------------- #
# VIEW 1: Irrigation Detection (all original UI + logic, unchanged, wrapped in one function)
# --------------------------------------------------------------------------- #
def irrigation_detection_view():
    st.title("💧 Farm Monitor")
    # --------------------------------------------------------------------------- #
    # State: the sidebar numbers are the single source of truth for the box.
    # --------------------------------------------------------------------------- #
    for k, v in zip(("west", "south", "east", "north"), DEFAULT_BBOX):
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("has_area", False)
    st.session_state.setdefault("drawn_bbox", None)

    pending = st.session_state.pop("pending_bbox", None)
    if pending:
        st.session_state.update(west=pending[0], south=pending[1], east=pending[2], north=pending[3])

    st.session_state.setdefault("start", dt.date.today() - dt.timedelta(days=365))
    st.session_state.setdefault("days", 365)
    st.session_state.setdefault("ndvi_thr", 0.30)
    st.session_state.setdefault("excl_water", True)
    st.session_state.setdefault("min_acres", 1.0)
    st.session_state.setdefault("sat_start", dt.date.today() - dt.timedelta(days=60))
    st.session_state.setdefault("model_path", "model.joblib")

    with st.sidebar:
        st.header("📍 Area")
        st.caption("Draw a rectangle on the map, or type coordinates:")
        c1, c2 = st.columns(2)
        c1.number_input("West", key="west", format="%.5f", step=0.001)
        c2.number_input("East", key="east", format="%.5f", step=0.001)
        c1.number_input("South", key="south", format="%.5f", step=0.001)
        c2.number_input("North", key="north", format="%.5f", step=0.001)
        b1, b2 = st.columns(2)
        if b1.button("Show overlay"):
            st.session_state["has_area"] = True
        if b2.button("Clear"):
            st.session_state["has_area"] = False
            st.session_state["drawn_bbox"] = None

        st.divider()
        st.header("🌱 Irrigated-land detection")
        project = st.text_input("Earth Engine project ID", key="project", help="Needed for real detection. "
                                "Leave empty to see demo polygons.")
        start = st.date_input("Period starts", key="start")
        days = st.slider("Period length (days)", 30, 730, step=5, key="days",
                         help="Longer = catches crops that are only green in some seasons.")
        basis = st.radio("Detection based on", [BASIS_PEAK, BASIS_MEDIAN], key="basis",
                         help="Peak greenness finds every field that gets green at any time in the period. "
                              "Median only finds fields that are green most of the time.")
        ndvi_thr = st.slider("Green-vegetation threshold (NDVI)", 0.15, 0.70, step=0.01, key="ndvi_thr",
                             help="Higher = only dense, healthy crops. Lower = includes sparse vegetation.")
        excl_water = st.checkbox("Exclude water", key="excl_water")
        min_acres = st.slider("Ignore patches smaller than (acres)", 0.2, 20.0, step=0.2, key="min_acres")

        st.divider()
        st.header("🛰️ Satellite analysis")
        sat_start = st.date_input("Radar/NDVI window starts (30 days)", key="sat_start")
        model_path = st.text_input("Model file (optional)", key="model_path")
        run = st.button("Run satellite analysis", type="primary")

    bbox = tuple(round(float(st.session_state[k]), 6) for k in ("west", "south", "east", "north"))
    # Remember the selection so the Crop Rotation view can read it (plain keys survive tool switches).
    st.session_state["_sel"] = {"bbox": bbox, "sat_start": sat_start, "start": start, "days": int(days),
                                "project": project.strip()}
    # --------------------------------------------------------------------------- #
    # Compute the result for the current box (cached per settings)
    # --------------------------------------------------------------------------- #
    result = None
    if st.session_state["has_area"]:
        peak = basis == BASIS_PEAK
        sig = (bbox, project.strip(), start.isoformat(), int(days), peak, round(ndvi_thr, 3), excl_water,
               round(min_acres, 2))
        if st.session_state.get("res_sig") == sig:
            result = st.session_state.get("res")
        else:
            try:
                if project.strip():
                    with st.spinner("Fetching Sentinel-2 for your box from Google Earth Engine (can take ~1 min)..."):
                        ind = fetch_indices(project.strip(), bbox, start.isoformat(), int(days), peak)
                    with st.spinner("Detecting irrigated land..."):
                        result = classify(ind, ndvi_thr, excl_water, min_acres)
                    result.update(mode="real", scale=ind["scale"], scenes=ind["scenes"], period=ind["period"], peak=peak)
                else:
                    result = dict(demo_area(bbox), mode="demo")
            except Exception as exc:
                st.error(f"Could not process this area: {exc}")
                result = None
            st.session_state["res_sig"], st.session_state["res"] = sig, result

    fg = folium.FeatureGroup(name="Irrigated land (detected)")
    fg_ndvi = folium.FeatureGroup(name="NDVI greenness (what the satellite saw)", show=False)
    if result:
        west, south, east, north = bbox
        folium.Rectangle([[south, west], [north, east]], color="#19c3e6", weight=2, dash_array="6",
                         fill=False).add_to(fg)
        if result["geojson"]:
            folium.GeoJson(
                result["geojson"],
                style_function=lambda _f: {"color": "#ffffff", "weight": 1, "fillColor": "#1e6eff", "fillOpacity": 0.55},
                highlight_function=lambda _f: {"weight": 2.5, "fillOpacity": 0.8},
                tooltip=folium.GeoJsonTooltip(fields=["acres"], aliases=["Irrigated patch (acres)"]),
            ).add_to(fg)
        elif result["rgba"] is not None:
            w_, s_, e_, n_ = result["bounds"]
            folium.raster_layers.ImageOverlay(result["rgba"], bounds=[[s_, w_], [n_, e_]], opacity=1).add_to(fg)
        if result.get("ndvi_rgba") is not None:
            w_, s_, e_, n_ = result["bounds"]
            folium.raster_layers.ImageOverlay(result["ndvi_rgba"], bounds=[[s_, w_], [n_, e_]], opacity=1).add_to(fg_ndvi)

    if st.session_state["has_area"]:
        st.caption("Drag the box's handles to adjust it, or draw a new one (it replaces the old one). Use the layer button "
                   "(top-right of the map) to switch layers, including **NDVI greenness**. The satellite photo is an "
                   "older reference image, so compare detection against the NDVI layer, not only the photo.")
    else:
        st.info("👆 Click the **▭ rectangle tool** at the top-left of the map and drag over your farm.")

    out = st_folium(base_map(), feature_group_to_add=[fg, fg_ndvi], layer_control=folium.LayerControl(collapsed=True),
                    height=620, use_container_width=True, returned_objects=["all_drawings"], key="draw_map")

    # --- capture the drawn box ---------------------------------------------------
    drawn = bbox_from_drawings((out or {}).get("all_drawings"))
    if drawn and drawn != st.session_state["drawn_bbox"]:
        st.session_state["drawn_bbox"] = drawn
        st.session_state["pending_bbox"] = drawn
        st.session_state["has_area"] = True
        st.rerun()
    if drawn is None and st.session_state["drawn_bbox"] is not None and out and out.get("all_drawings") == []:
        st.session_state["drawn_bbox"] = None  # user deleted the rectangle
        st.session_state["has_area"] = False
        st.rerun()


    # --- NASA context (same values the Crop Rotation view uses; live when that view has fetched them) ------
    nasa = nasa_for_irrigation(bbox, sat_start)
    st.markdown("#### 🛰️ NASA context")
    n1, n2, n3 = st.columns(3)
    n1.metric("🌧️ NASA GPM · Rainfall", f"{nasa['rain']:.1f} mm", f"30-day total · {nasa_tag(nasa, 'rain')}", delta_color="off")
    n2.metric("💧 NASA SMAP · Soil moisture", f"{nasa['smap']:.0f} %", f"surface · {nasa_tag(nasa, 'smap')}", delta_color="off")
    n3.metric("🌡️ NASA POWER · Temperature", f"{nasa['temp']:.1f} °C", f"mean · {nasa_tag(nasa, 'temp')}", delta_color="off")
    if nasa["rain"] <= 5:
        st.info("NASA environmental data confirms low recent rainfall, validating the AI detection of "
                "human-driven irrigation.")
    else:
        st.warning("NASA data shows notable recent rainfall, so some of the green area may be rain-fed rather than "
                   "irrigated. Treat the detection with more caution.")
    if not any(nasa["live"].values()):
        st.caption("NASA values are a deterministic climatology estimate for this location. Open **Crop Rotation "
                   "Strategy** to fetch live NASA data.")
    # --------------------------------------------------------------------------- #
    # Details (tabs under the map)
    # --------------------------------------------------------------------------- #
    if result:
        real = result["mode"] == "real"
        if not real:
            st.warning("DEMO MODE: these blue shapes are random. Enter your Earth Engine project ID in the "
                       "sidebar to detect real irrigated land for this box.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Selected area", f"{result['total_acres']:,.0f} acres")
        m2.metric("💧 Irrigated land", f"{result['irrigated_acres']:,.0f} acres")
        m3.metric("Irrigated share", f"{100 * result['irrigated_acres'] / max(result['total_acres'], 1e-9):.1f} %")
        m4.metric("Patches found", f"{result['n_fields']:,}")

        t_sum, t_patch, t_ndvi, t_method = st.tabs(["📊 Summary", "🌾 Patches", "🌿 NDVI view", "ℹ️ Method & data"])

        with t_sum:
            st.code(f"bbox (W, S, E, N) = {bbox}", language="text")
            if real:
                a, b, c = st.columns(3)
                a.metric("Mean NDVI, irrigated patches", f"{result['ndvi_in']:.2f}")
                b.metric("Mean NDVI, whole box", f"{result['ndvi_all']:.2f}")
                c.metric("Sentinel-2 scenes used", result["scenes"])
                hist, edges = result["hist"]
                if len(hist):
                    st.subheader("How green is the box? (NDVI distribution)")
                    st.bar_chart(pd.Series(hist, index=[f"{x:.2f}" for x in edges[:-1]]))
                    st.caption(f"Pixels to the right of {ndvi_thr:.2f} count as irrigated. Bare desert is about 0.05-0.15; "
                               "healthy crops reach 0.5-0.8.")
                if result["irrigated_acres"] / max(result["total_acres"], 1e-9) < 0.05:
                    st.info("Very little was detected. Try: a longer period, 'Peak greenness', or a lower NDVI threshold. "
                            "Turn on the **NDVI greenness** layer on the map to see what the satellite measured.")

        with t_patch:
            if result["patches"]:
                df = pd.DataFrame(result["patches"])
                st.dataframe(df, width="stretch", hide_index=True)
                d1, d2 = st.columns(2)
                d1.download_button("⬇️ Patches (CSV)", df.to_csv(index=False), "irrigated_patches.csv")
                if result["geojson"]:
                    import json
                    d2.download_button("⬇️ Polygons (GeoJSON)", json.dumps(result["geojson"]), "irrigated_land.geojson")
            else:
                st.info("No patch table in demo mode, or nothing was detected.")

        with t_ndvi:
            if real and result["ndvi"] is not None:
                fig, ax = plt.subplots(figsize=(9, 6))
                im = ax.imshow(result["ndvi"], cmap="RdYlGn", vmin=-0.1, vmax=0.9)
                ax.contour(result["mask"].astype(float), levels=[0.5], colors="#1e6eff", linewidths=1.2)
                ax.set_title("NDVI with detected irrigated land outlined in blue")
                ax.axis("off")
                fig.colorbar(im, ax=ax, shrink=0.8, label="NDVI (greenness)")
                st.pyplot(fig)
            else:
                st.info("NDVI view needs real detection (enter your Earth Engine project ID).")

        with t_method:
            if real:
                s_, e_ = result["period"]
                st.markdown(
                    f"""
**Data:** Sentinel-2 surface reflectance, cloud/shadow masked, **{result['scenes']} scenes**,
{s_} to {e_}, at **{result['scale']} m** resolution.

**Rule:** a pixel counts as irrigated when its {'**peak (90th-percentile) NDVI** over the whole period' if result['peak'] else '**median NDVI**'}
is at least **{ndvi_thr:.2f}**{', it is not water (NDWI > 0.1 is removed)' if excl_water else ''}, and it belongs to a patch of at
least **{min_acres:g} acres**.

**Why peak greenness?** In dry regions crops cannot grow without water, so any pixel that becomes green at some point
in the year is being irrigated. A single snapshot misses fields that are fallow or harvested on that date.

**Limits:** outside dry regions, green pixels can also be rain-fed crops, forest or natural vegetation. Very small
fields and fields under cloud all year can be missed. This is an estimate from satellite greenness, not a field survey.
                """
                )
            else:
                st.info("Method details appear for real detection.")


    # --------------------------------------------------------------------------- #
    # Optional: satellite maps for the same box (VV / VH / NDVI + classification)
    # --------------------------------------------------------------------------- #


    if run:
        if not project.strip():
            st.error("Enter your Earth Engine project ID in the sidebar.")
            st.stop()
        import gee_pipeline as gp  # loaded only when needed

        tif = os.path.join(tempfile.gettempdir(), "farm_stack.tif")
        with st.spinner("Fetching data from Google Earth Engine..."):
            try:
                gp.export_stack(project.strip(), list(bbox), sat_start.isoformat(), tif)
            except Exception as exc:
                st.error(f"Earth Engine error: {exc}")
                st.stop()

        with rasterio.open(tif) as ds:
            arr = ds.read().astype("float32")  # (3, H, W) = VV, VH, NDVI
        h, w = arr.shape[1:]
        valid = ~(arr == NODATA).any(axis=0)

        vv = np.where(valid, arr[0], np.nan)
        vh = np.where(valid, arr[1], np.nan)
        ndvi = np.where(valid, arr[2], np.nan)

        pred, codes = None, None
        if os.path.exists(model_path):
            clf = load_model(model_path)
            codes = {c: i for i, c in enumerate(clf.classes_)}
            res = np.full(h * w, np.nan)
            mask = valid.ravel()
            res[mask] = [codes[p] for p in clf.predict(make_features(arr)[mask])]
            pred = res.reshape(h, w)
        else:
            st.info("No model found, showing satellite maps only.")

        n = 4 if pred is not None else 3
        fig, axs = plt.subplots(1, n, figsize=(5 * n, 5))
        show_map(axs[0], vv, "VV (dB)", "gray")
        show_map(axs[1], vh, "VH (dB)", "gray")
        show_map(axs[2], ndvi, "NDVI", "RdYlGn", -0.1, 0.9)
        if pred is not None:
            show_map(axs[3], pred, "Predicted class", "tab10")
        st.pyplot(fig)

        if codes:
            st.caption("Classes: " + ", ".join(f"{i}={c}" for c, i in codes.items()))
        st.metric("Mean NDVI", f"{np.nanmean(ndvi):.2f}")

        with open(tif, "rb") as f:
            st.download_button("Download stack (.tif)", f, "farm_stack.tif")



# --------------------------------------------------------------------------- #
# VIEW 2: Crop Rotation Strategy (reads the box + date chosen in the Irrigation Detection view)
# --------------------------------------------------------------------------- #
SEASON_CROPS = {
    "Water Conservation Strategy": {"Summer": "Sorghum or pearl millet", "Winter": "Barley"},
    "Soil Health Strategy": {"Summer": "Cowpea, or alfalfa if irrigation is reliable",
                             "Winter": "Faba bean, berseem clover or chickpea"},
    "High Yield Strategy": {"Summer": "Corn (maize)", "Winter": "Wheat"},
}


def crop_rotation_view():
    st.title("🔄 Crop Rotation Strategy")
    sel = st.session_state.get("_sel")
    if sel is None:
        sel = {"bbox": DEFAULT_BBOX, "sat_start": dt.date.today() - dt.timedelta(days=60), "project": ""}
        st.info("No area chosen yet, so the default area is used. Pick a box in **Irrigation Detection** first.")
    bbox, d0, project = sel["bbox"], sel["sat_start"], sel["project"]
    clat, clon = (bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2
    d1 = d0 + dt.timedelta(days=WINDOW_DAYS - 1)

    with st.spinner("Retrieving NASA data for your area..."):
        nasa = get_nasa_profile(clat, clon, d0, project, live=True)
    st.session_state.setdefault("_nasa_live", {})[(round(clat, 2), round(clon, 2), d0.isoformat())] = nasa

    st.caption(f"Area centre **{clat:.4f}°, {clon:.4f}°** · box W/S/E/N = {bbox} · "
               f"NASA window **{d0} → {d1}** (the 'Radar/NDVI window starts' date from Irrigation Detection)")
    rain_lbl, temp_lbl, smap_lbl = nasa_labels(nasa)

    st.subheader("NASA environmental data")
    c1, c2, c3 = st.columns(3)
    c1.metric("🌧️ NASA GPM · Rainfall", f"{nasa['rain']:.1f} mm", f"30-day total · {rain_lbl}", delta_color="off")
    c2.metric("💧 NASA SMAP · Soil moisture", f"{nasa['smap']:.0f} %", f"surface · {smap_lbl}", delta_color="off")
    c3.metric("🌡️ NASA POWER · Temperature", f"{nasa['temp']:.1f} °C", f"mean · {temp_lbl}", delta_color="off")
    for col, key in zip((c1, c2, c3), ("rain", "smap", "temp")):
        col.caption(("🟢 " if nasa["live"][key] else "⚪ ") + nasa["src"][key])
    if not any(nasa["live"].values()):
        st.warning("No live NASA connection, so these are deterministic climatology values for these exact "
                   "coordinates and dates (not a live measurement).")
    if nasa["notes"]:
        with st.expander("Data source notes"):
            for n in nasa["notes"]:
                st.markdown(f"- {n}")

    strategies, scores, order, wstress = recommend_rotations(nasa)
    best = order[0]
    top = strategies[best]
    season = "Summer" if ((5 <= (d0 + dt.timedelta(days=15)).month <= 10) == (clat >= 0)) else "Winter"
    stress = "high" if wstress > 0.6 else "moderate" if wstress > 0.35 else "low"

    st.subheader("Recommended strategy")
    st.success(f"**{top['icon']} {top['name']}** · fit score {scores[best]}/100 · "
               f"for the {season.lower()} season: **{SEASON_CROPS[top['name']][season]}**.")

    st.subheader("Why this strategy was chosen")
    rain, temp, smap = nasa["rain"], nasa["temp"], nasa["smap"]
    st.markdown(
        f"- **Rainfall (GPM) {rain:.1f} mm in 30 days** - {rain_lbl.lower()}. "
        f"{'Natural rainfall supplies almost no water, so crops depend entirely on irrigation.' if rain <= 5 else 'Rain helps, but not enough to rely on alone.' if rain <= 15 else 'Rainfall supplies a meaningful share of crop water.'}\n"
        f"- **Temperature (POWER) {temp:.1f} °C** - {temp_lbl.lower()}. "
        f"{'Heat raises crop water demand and limits high-demand crops such as corn and wheat.' if temp >= 33 else 'A workable range for most crops.' if temp >= 22 else 'Cool conditions suit winter crops.'}\n"
        f"- **Soil moisture (SMAP) {smap:.0f} %** - {smap_lbl.lower()}. "
        f"{'Little water is stored in the root zone.' if smap < 20 else 'A moderate water reserve.' if smap < 30 else 'A healthy water reserve.'}\n"
        f"- **Combined water-stress index {wstress:.2f}** (0 = none, 1 = extreme) = **{stress}** water stress."
    )
    st.markdown(f"**Therefore:** {top['why']}")
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**Benefits**")
        for b in top["benefits"]:
            st.markdown(f"- {b}")
    with b2:
        st.markdown(f"**Seasonal timing ({season})**")
        st.markdown(f"- Plant now-window: {SEASON_CROPS[top['name']][season]}")
        other = "Winter" if season == "Summer" else "Summer"
        st.markdown(f"- Next season ({other.lower()}): {SEASON_CROPS[top['name']][other]}")
        st.caption(f"Watch out: {top['watch']}")

    with st.expander("Why not the other strategies?"):
        for i in order[1:]:
            s = strategies[i]
            st.markdown(f"**{s['icon']} {s['name']}** (score {scores[i]}/100): {s['why']}")
    with st.expander("How the rules score each strategy"):
        st.markdown(
            "**Water stress** = 45% dry soil (SMAP below 35%) + 35% heat (POWER above 25 °C) + 20% lack of rain "
            "(GPM below 10 mm).\n\n"
            "- **Water Conservation** rises with water stress.\n"
            "- **Soil Health** peaks at moderate soil moisture (~27%), cooler temperature and some rain.\n"
            "- **High Yield** rises when water stress is low and temperature is at or below 30 °C.\n\n"
            "Same inputs always give the same answer. These are simple agronomic rules for decision support, "
            "**not** a validated yield model."
        )





# Step 4: routing
if view == "Irrigation Detection":
    if st.session_state.get("_prev_view") != "Irrigation Detection":
        st.session_state["drawn_bbox"] = None   # fresh map component: do not mistake its empty drawings for a deletion
    irrigation_detection_view()
else:
    crop_rotation_view()
st.session_state["_prev_view"] = view
