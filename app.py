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
GEE_PROJECT = "stately-lodge-514511-m4"       # Google Earth Engine project used for every ee.Initialize(project=...)
MAX_POLYGONS = 1500
BASIS_PEAK = "Peak greenness over the period (best for irrigation)"
BASIS_MEDIAN = "Median (a typical snapshot)"

st.set_page_config(page_title="HydroScan AI", page_icon="💧", layout="wide", initial_sidebar_state="collapsed")
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

st.session_state.setdefault("project", GEE_PROJECT)   # Expert mode's project box starts with the real GEE project
# Keep sidebar widget values alive while the other tool is shown (Streamlit drops widgets that are not rendered).
for _k in ("west", "south", "east", "north", "project", "start", "days", "basis", "ndvi_thr", "excl_water",
           "min_acres", "sat_start", "model_path"):
    if _k in st.session_state:
        st.session_state[_k] = st.session_state[_k]

# (The original tool router now lives in the "Expert mode" section at the bottom of this file.)

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





# =========================================================================== #
# NEW UI LAYER (presentation only). Everything above this line is the untouched backend.
# The wizard below only CALLS the backend: base_map, fetch_indices, classify, demo_area,
# get_nasa_profile, recommend_rotations, nasa_labels, SEASON_CROPS, metres_per_degree, bbox_from_drawings.
# =========================================================================== #
FEDDAN_M2 = 4200.83
BAND_HEALTHY, BAND_ATTN = 0.50, 0.40   # display-only NDVI bands inside detected irrigated land

TR = {
    "en": {
        "step_location": "Location", "step_services": "Services", "step_analyze": "Analyze", "step_results": "Results",
        "welcome": "Welcome to HydroScan AI.", "welcome_sub": "Find your farm on the map in three easy steps.",
        "draw_hint": "Use the rectangle tool at the top-left of the map and drag over your farm.",
        "area_selected": "Farm area selected", "no_area": "No farm area yet",
        "your_farm": "Your farm", "feddan": "feddan", "about_ha": "about {ha} hectares",
        "coords_title": "Or type coordinates", "west": "West", "south": "South", "east": "East", "north": "North",
        "apply_coords": "Use these coordinates", "adv_settings": "Advanced settings",
        "ee_project": "Earth Engine project ID",
        "ee_help": "Needed for real satellite detection. Leave empty to see demo shapes.",
        "period_start": "Period starts", "period_days": "Period length (days)", "basis": "Detection based on",
        "basis_peak": "Peak greenness over the period (best for irrigation)", "basis_median": "Median (a typical snapshot)",
        "ndvi_thr": "Green-vegetation threshold (NDVI)", "excl_water": "Exclude water",
        "min_acres": "Ignore patches smaller than (acres)", "sat_start": "NASA data window starts (30 days)",
        "confirm": "Confirm location & start", "services_title": "What do you need today?",
        "services_sub": "Tap one or more services. You can choose all three.",
        "svc_water": "Manage my water", "svc_water_sub": "Water & irrigation",
        "svc_water_d": "See if your crops have enough water, and where.",
        "svc_crops": "Choose my crops", "svc_crops_sub": "Crop selection & rotation",
        "svc_crops_d": "Find the best crops for the coming seasons.",
        "svc_soil": "Understand my soil", "svc_soil_sub": "Soil health",
        "svc_soil_d": "Follow soil moisture and how well plants are growing.",
        "select": "Select", "selected": "Selected", "n_selected": "{n} services selected",
        "analyze_btn": "Analyze my location", "change": "Change", "back": "Back",
        "analyzing": "Analyzing your farm…", "wait": "Please wait a moment.",
        "tag_water": "Water", "tag_soil": "Soil", "tag_crops": "Crops",
        "st_sat": "Reading satellite images", "st_ndvi": "Checking plant greenness and water",
        "st_rain": "Getting rain and weather data", "st_smap": "Reading soil moisture",
        "st_prep": "Preparing your results",
        "crops_skipped": "Crop planning was not selected, so it is skipped.",
        "err_process": "Could not process this area: {e}",
        "tab_water": "Water & irrigation", "tab_crops": "Crops & rotation", "tab_soil": "Soil health",
        "title_water": "Water & irrigation", "title_crops": "Crop selection & rotation", "title_soil": "Soil health",
        "measured": "Measured", "estimated": "Estimated",
        "card_cond": "Crop water condition", "cond_healthy": "Healthy", "cond_attn": "Needs attention", "cond_dry": "Dry",
        "cond_good": "Most of your irrigated land looks healthy.",
        "cond_mixed": "Parts of your irrigated land need attention.",
        "cond_caption": "{h}% healthy · {a}% needs attention · {d}% dry · Sentinel-2 NDVI "
                        "(0.50 and above is healthy, 0.40 to 0.50 needs attention, below 0.40 is dry)",
        "cond_none": "Crop condition needs real satellite detection. Add an Earth Engine project ID in Advanced settings.",
        "irr_detected": "Irrigation detected",
        "irr_msg": "{pct}% of the selected area ({fed} feddan) is irrigated, in {n} patches.",
        "irr_little": "Very little irrigation was detected. Try a longer period or a lower threshold in Advanced settings.",
        "rain_30": "Rain · last 30 days", "smap_title": "Soil moisture", "temp_title": "Temperature",
        "nasa_ok": "NASA data confirms low recent rainfall, which supports the AI detection of human-driven irrigation.",
        "nasa_warn": "NASA data shows notable recent rainfall, so some green area may be rain-fed rather than irrigated. "
                     "Treat the detection with more caution.",
        "nasa_est": "NASA values are a deterministic climatology estimate for this location (no live connection).",
        "demo_warn": "DEMO MODE: these shapes are sample data. Add your Earth Engine project ID in Advanced settings "
                     "to detect real irrigated land.",
        "legend": "Crop water condition", "sat_view": "Satellite view",
        "per_30": "30-day total", "surface": "surface", "mean": "mean",
        "rec_title": "Recommended strategy", "fit": "fit score {s}/100", "for_season": "for the {season} season",
        "summer": "Summer", "winter": "Winter", "season_now": "Now · {s}", "season_next": "Next · {s}",
        "season_then": "Then · {s}", "suggestions": "Suggestions, not guarantees",
        "why_title": "Why this strategy was chosen", "plan_uses": "This plan uses:",
        "uses1": "NASA GPM rainfall", "uses2": "NASA SMAP soil moisture", "uses3": "NASA POWER temperature",
        "uses4": "A transparent rule-based engine", "benefits": "Benefits", "watch": "Watch out",
        "other_strats": "Why not the other strategies?", "rules_title": "How the rules score each strategy",
        "rules_body": "**Water stress** = 45% dry soil (SMAP below 35%) + 35% heat (POWER above 25 °C) + 20% lack of rain "
                      "(GPM below 10 mm).\n\n"
                      "- **Water Conservation** rises with water stress.\n"
                      "- **Soil Health** peaks at moderate soil moisture (about 27%), cooler temperature and some rain.\n"
                      "- **High Yield** rises when water stress is low and temperature is at or below 30 °C.\n\n"
                      "The same inputs always give the same answer. These are simple agronomic rules for decision "
                      "support, **not** a validated yield model.",
        "b_rain": "**Rainfall (GPM) {v:.1f} mm in 30 days** · {lbl}. {txt}",
        "rain_t1": "Natural rainfall supplies almost no water, so crops depend entirely on irrigation.",
        "rain_t2": "Rain helps, but not enough to rely on alone.", "rain_t3": "Rainfall supplies a meaningful share of crop water.",
        "b_temp": "**Temperature (POWER) {v:.1f} °C** · {lbl}. {txt}",
        "temp_t1": "Heat raises crop water demand and limits high-demand crops such as corn and wheat.",
        "temp_t2": "A workable range for most crops.", "temp_t3": "Cool conditions suit winter crops.",
        "b_smap": "**Soil moisture (SMAP) {v:.0f} %** · {lbl}. {txt}",
        "smap_t1": "Little water is stored in the root zone.", "smap_t2": "A moderate water reserve.",
        "smap_t3": "A healthy water reserve.",
        "b_stress": "**Combined water-stress index {v:.2f}** (0 = none, 1 = extreme) = **{lvl}** water stress.",
        "stress_low": "low", "stress_moderate": "moderate", "stress_high": "high",
        "therefore": "**Therefore:** {why}",
        "soil_note": "Soil moisture comes from NASA SMAP. Plant vigour comes from Sentinel-2 NDVI.",
        "ndvi_irr": "Mean NDVI, irrigated land", "ndvi_all": "Mean NDVI, whole area", "scenes": "Sentinel-2 scenes used",
        "ndvi_dist": "How green is the area? (NDVI distribution)",
        "ndvi_note": "Bare desert is about 0.05 to 0.15. Healthy crops reach 0.5 to 0.8.",
        "ndvi_none": "Plant growth numbers need real satellite detection (add an Earth Engine project ID).",
        "new_analysis": "Start a new analysis", "expert": "Expert mode (original tools)",
        "choose_tool": "Choose Tool:", "tool_irr": "Irrigation Detection", "tool_rot": "Crop Rotation Strategy",
        "source": "Source",
    },
    "ar": {
        "step_location": "الموقع", "step_services": "الخدمات", "step_analyze": "التحليل", "step_results": "النتائج",
        "welcome": "مرحبًا بك في HydroScan AI.", "welcome_sub": "حدّد مزرعتك على الخريطة في ثلاث خطوات بسيطة.",
        "draw_hint": "استخدم أداة المستطيل أعلى يسار الخريطة واسحب فوق مزرعتك.",
        "area_selected": "تم تحديد مساحة المزرعة", "no_area": "لم تُحدَّد مساحة المزرعة بعد",
        "your_farm": "مزرعتك", "feddan": "فدان", "about_ha": "حوالي {ha} هكتار",
        "coords_title": "أو اكتب الإحداثيات", "west": "غرب", "south": "جنوب", "east": "شرق", "north": "شمال",
        "apply_coords": "استخدم هذه الإحداثيات", "adv_settings": "إعدادات متقدمة",
        "ee_project": "معرّف مشروع Earth Engine",
        "ee_help": "مطلوب للكشف الفعلي عبر الأقمار الصناعية. اتركه فارغًا لعرض أشكال تجريبية.",
        "period_start": "بداية الفترة", "period_days": "طول الفترة (بالأيام)", "basis": "أساس الكشف",
        "basis_peak": "أعلى خضرة خلال الفترة (الأفضل لكشف الري)", "basis_median": "الوسيط (لقطة نموذجية)",
        "ndvi_thr": "حد الغطاء النباتي الأخضر (NDVI)", "excl_water": "استبعاد المياه",
        "min_acres": "تجاهل البقع الأصغر من (إيكر)", "sat_start": "بداية نافذة بيانات ناسا (30 يومًا)",
        "confirm": "تأكيد الموقع والبدء", "services_title": "ماذا تحتاج اليوم؟",
        "services_sub": "اختر خدمة أو أكثر. يمكنك اختيار الثلاث معًا.",
        "svc_water": "إدارة مياهي", "svc_water_sub": "المياه والري",
        "svc_water_d": "تعرّف إن كانت محاصيلك تحصل على مياه كافية، وفي أي مكان.",
        "svc_crops": "اختيار محاصيلي", "svc_crops_sub": "اختيار المحاصيل والدورة الزراعية",
        "svc_crops_d": "اكتشف أفضل المحاصيل للمواسم القادمة.",
        "svc_soil": "فهم تربتي", "svc_soil_sub": "صحة التربة",
        "svc_soil_d": "تابع رطوبة التربة ومدى جودة نمو النباتات.",
        "select": "اختيار", "selected": "تم الاختيار", "n_selected": "تم اختيار {n} من الخدمات",
        "analyze_btn": "حلّل موقعي", "change": "تغيير", "back": "رجوع",
        "analyzing": "جارٍ تحليل مزرعتك…", "wait": "من فضلك انتظر قليلًا.",
        "tag_water": "المياه", "tag_soil": "التربة", "tag_crops": "المحاصيل",
        "st_sat": "قراءة صور الأقمار الصناعية", "st_ndvi": "فحص خضرة النباتات والمياه",
        "st_rain": "جلب بيانات الأمطار والطقس", "st_smap": "قراءة رطوبة التربة",
        "st_prep": "تجهيز النتائج",
        "crops_skipped": "لم يتم اختيار تخطيط المحاصيل، لذا تم تخطّيه.",
        "err_process": "تعذّرت معالجة هذه المنطقة: {e}",
        "tab_water": "المياه والري", "tab_crops": "المحاصيل والدورة", "tab_soil": "صحة التربة",
        "title_water": "المياه والري", "title_crops": "اختيار المحاصيل والدورة الزراعية", "title_soil": "صحة التربة",
        "measured": "مُقاس", "estimated": "تقديري",
        "card_cond": "حالة المحاصيل المائية", "cond_healthy": "سليمة", "cond_attn": "تحتاج انتباهًا", "cond_dry": "جافة",
        "cond_good": "معظم أراضيك المروية تبدو بحالة جيدة.",
        "cond_mixed": "أجزاء من أراضيك المروية تحتاج انتباهًا.",
        "cond_caption": "{h}% سليمة · {a}% تحتاج انتباهًا · {d}% جافة · مؤشر NDVI من Sentinel-2 "
                        "(0.50 فأكثر سليمة، من 0.40 إلى 0.50 تحتاج انتباهًا، وأقل من 0.40 جافة)",
        "cond_none": "حالة المحاصيل تتطلب كشفًا فعليًا بالأقمار الصناعية. أضف معرّف مشروع Earth Engine في الإعدادات المتقدمة.",
        "irr_detected": "تم رصد الري",
        "irr_msg": "{pct}% من المنطقة المحددة ({fed} فدان) مروية، موزعة على {n} بقعة.",
        "irr_little": "تم رصد ري قليل جدًا. جرّب فترة أطول أو حدًّا أقل في الإعدادات المتقدمة.",
        "rain_30": "الأمطار · آخر 30 يومًا", "smap_title": "رطوبة التربة", "temp_title": "درجة الحرارة",
        "nasa_ok": "بيانات ناسا تؤكد قلة الأمطار مؤخرًا، مما يدعم كشف الذكاء الاصطناعي للري الناتج عن تدخل الإنسان.",
        "nasa_warn": "بيانات ناسا تُظهر أمطارًا ملحوظة مؤخرًا، فقد تكون بعض المساحات الخضراء مروية بالمطر وليس بالري. "
                     "تعامل مع نتيجة الكشف بحذر أكبر.",
        "nasa_est": "قيم ناسا هنا تقدير مناخي ثابت لهذا الموقع (لا يوجد اتصال مباشر).",
        "demo_warn": "وضع تجريبي: هذه الأشكال بيانات نموذجية. أضف معرّف مشروع Earth Engine في الإعدادات المتقدمة "
                     "لكشف الأراضي المروية الحقيقية.",
        "legend": "حالة المحاصيل المائية", "sat_view": "عرض القمر الصناعي",
        "per_30": "إجمالي 30 يومًا", "surface": "السطحية", "mean": "المتوسط",
        "rec_title": "الاستراتيجية الموصى بها", "fit": "درجة الملاءمة {s}/100", "for_season": "لموسم {season}",
        "summer": "الصيف", "winter": "الشتاء", "season_now": "الآن · {s}", "season_next": "التالي · {s}",
        "season_then": "ثم · {s}", "suggestions": "اقتراحات وليست ضمانات",
        "why_title": "لماذا اختيرت هذه الاستراتيجية", "plan_uses": "تعتمد هذه الخطة على:",
        "uses1": "أمطار ناسا GPM", "uses2": "رطوبة التربة من ناسا SMAP", "uses3": "حرارة ناسا POWER",
        "uses4": "محرك قواعد شفاف وقابل للتفسير", "benefits": "الفوائد", "watch": "انتبه",
        "other_strats": "لماذا لم تُختَر الاستراتيجيات الأخرى؟", "rules_title": "كيف تقيّم القواعد كل استراتيجية",
        "rules_body": "**إجهاد المياه** = 45% جفاف التربة (SMAP أقل من 35%) + 35% حرارة (POWER فوق 25 °م) + 20% قلة المطر "
                      "(GPM أقل من 10 مم).\n\n"
                      "- **توفير المياه** يرتفع مع ارتفاع إجهاد المياه.\n"
                      "- **صحة التربة** تبلغ ذروتها عند رطوبة تربة متوسطة (حوالي 27%) وحرارة أقل وبعض الأمطار.\n"
                      "- **الإنتاج العالي** يرتفع عندما يكون إجهاد المياه منخفضًا والحرارة 30 °م أو أقل.\n\n"
                      "نفس المدخلات تعطي دائمًا نفس النتيجة. هذه قواعد زراعية بسيطة لدعم القرار، "
                      "**وليست** نموذجًا معتمدًا لتقدير المحصول.",
        "b_rain": "**الأمطار (GPM) {v:.1f} مم خلال 30 يومًا** · {lbl}. {txt}",
        "rain_t1": "الأمطار الطبيعية لا تكاد توفر مياهًا، لذا تعتمد المحاصيل كليًا على الري.",
        "rain_t2": "المطر يساعد لكن لا يكفي للاعتماد عليه وحده.", "rain_t3": "الأمطار توفر حصة معتبرة من احتياج المحاصيل.",
        "b_temp": "**درجة الحرارة (POWER) {v:.1f} °م** · {lbl}. {txt}",
        "temp_t1": "الحرارة ترفع احتياج المحاصيل للمياه وتحدّ من المحاصيل كثيفة الاستهلاك مثل الذرة والقمح.",
        "temp_t2": "نطاق مناسب لمعظم المحاصيل.", "temp_t3": "الأجواء الباردة تناسب المحاصيل الشتوية.",
        "b_smap": "**رطوبة التربة (SMAP) {v:.0f}%** · {lbl}. {txt}",
        "smap_t1": "مياه قليلة مخزنة في منطقة الجذور.", "smap_t2": "مخزون مياه متوسط.", "smap_t3": "مخزون مياه جيد.",
        "b_stress": "**مؤشر إجهاد المياه المجمّع {v:.2f}** (0 = لا يوجد، 1 = شديد) = إجهاد مياه **{lvl}**.",
        "stress_low": "منخفض", "stress_moderate": "متوسط", "stress_high": "مرتفع",
        "therefore": "**الخلاصة:** {why}",
        "soil_note": "رطوبة التربة من ناسا SMAP. قوة نمو النباتات من مؤشر NDVI في Sentinel-2.",
        "ndvi_irr": "متوسط NDVI للأراضي المروية", "ndvi_all": "متوسط NDVI للمنطقة كاملة", "scenes": "عدد مشاهد Sentinel-2",
        "ndvi_dist": "ما مدى خضرة المنطقة؟ (توزيع NDVI)",
        "ndvi_note": "الصحراء العارية حوالي 0.05 إلى 0.15. المحاصيل السليمة تصل إلى 0.5 – 0.8.",
        "ndvi_none": "أرقام نمو النباتات تتطلب كشفًا فعليًا بالأقمار الصناعية (أضف معرّف مشروع Earth Engine).",
        "new_analysis": "ابدأ تحليلًا جديدًا", "expert": "الوضع المتقدم (الأدوات الأصلية)",
        "choose_tool": "اختر الأداة:", "tool_irr": "كشف الري", "tool_rot": "استراتيجية الدورة الزراعية",
        "source": "المصدر",
    },
}

# Arabic names for text that the (untouched) backend returns in English.
STRAT_AR = {"Water Conservation Strategy": "استراتيجية توفير المياه", "Soil Health Strategy": "استراتيجية صحة التربة",
            "High Yield Strategy": "استراتيجية الإنتاج العالي"}
CROP_AR = {"Sorghum": "الذرة الرفيعة", "Barley": "الشعير", "Pearl millet": "الدخن اللؤلؤي", "Alfalfa": "البرسيم الحجازي",
           "Faba bean": "الفول البلدي", "Chickpea": "الحمص", "Clover": "البرسيم", "Wheat": "القمح",
           "Corn (maize)": "الذرة الشامية"}
SEASON_AR = {"Sorghum or pearl millet": "الذرة الرفيعة أو الدخن اللؤلؤي", "Barley": "الشعير",
             "Cowpea, or alfalfa if irrigation is reliable": "اللوبيا، أو البرسيم الحجازي إذا كان الري مضمونًا",
             "Faba bean, berseem clover or chickpea": "الفول البلدي أو البرسيم المصري أو الحمص",
             "Corn (maize)": "الذرة الشامية", "Wheat": "القمح"}
BENEFIT_AR = {"Lowest irrigation demand": "أقل احتياج للري",
              "Tolerates heat and dry spells": "يتحمل الحرارة وفترات الجفاف",
              "Keeps land productive when water is scarce": "يُبقي الأرض منتجة عند شح المياه",
              "Legumes fix nitrogen, so less fertiliser is needed": "البقوليات تثبّت النيتروجين فيقل الاحتياج للسماد",
              "Breaks cereal pest and disease cycles": "يقطع دورات آفات وأمراض الحبوب",
              "Builds soil organic matter": "يزيد المادة العضوية في التربة",
              "Highest yield and income potential": "أعلى إنتاج ودخل محتمل",
              "Strong, established markets": "أسواق قوية ومستقرة",
              "Well-known farming practices": "ممارسات زراعية معروفة"}
WATCH_AR = {"Lower market price and yield than wheat or maize.": "سعر وإنتاج أقل من القمح أو الذرة.",
            "Alfalfa is water-hungry; avoid it when water stress is high.": "البرسيم الحجازي يستهلك مياهًا كثيرة؛ تجنّبه عند ارتفاع إجهاد المياه.",
            "Highest water and fertiliser demand; heat at flowering cuts yield.": "أعلى احتياج للمياه والسماد؛ والحرارة وقت الإزهار تقلل المحصول."}
LBL_AR = {"Very low": "منخفضة جدًا", "Moderate": "متوسطة", "High": "مرتفعة", "Very hot": "شديدة الحرارة", "Hot": "حارة",
          "Warm": "دافئة", "Mild": "معتدلة", "Very dry": "جافة جدًا", "Moist": "رطبة"}
LBL_EN = {k: k for k in LBL_AR}


def lang():
    return "ar" if st.session_state.get("lang_choice") == "عربي" else "en"


def t(key, **kw):
    s = TR[lang()].get(key) or TR["en"][key]
    return s.format(**kw) if kw else s


def tr_map(table, text):
    """Translate a backend English string when Arabic is active (falls back to the English original)."""
    return table.get(text, text) if lang() == "ar" else text


def tr_crops(text):
    """Translate a free-text crop list such as 'Cowpea, or alfalfa ...' using SEASON_AR."""
    return tr_map(SEASON_AR, text)


def inject_css():
    rtl = lang() == "ar"
    font = "'Cairo','Noto Naskh Arabic',Tahoma,sans-serif" if rtl else "'Inter','Segoe UI',sans-serif"
    serif = "'Cairo','Noto Naskh Arabic',serif" if rtl else "'Playfair Display','Georgia',serif"
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&family=Inter:wght@400;500;600&family=Playfair+Display:wght@600;700&display=swap');
        :root {{ --hs-dark:#2b4d3f; --hs-mid:#4f7d65; --hs-soft:#e4efe7; --hs-card:#ffffff; }}
        .stApp {{ background: linear-gradient(135deg,#e9f0ea 0%,#dce8df 100%); font-family:{font}; }}
        .block-container {{ max-width: 1400px; padding-top: 3.2rem; }}
        {"[data-testid='stAppViewContainer'] .main, .block-container {direction: rtl; text-align: right;}" if rtl else ""}
        {"iframe, .folium-map, .leaflet-container {direction: ltr;}" if rtl else ""}
        h1,h2,h3,.hs-serif {{ font-family:{serif}; color:var(--hs-dark); letter-spacing:0; }}
        .hs-logo {{ font-family:{serif}; font-size:1.7rem; font-weight:700; color:var(--hs-dark); }}
        .hs-stepper {{ display:flex; gap:14px; align-items:center; justify-content:center; flex-wrap:wrap; }}
        .hs-step {{ display:flex; align-items:center; gap:8px; color:#6b7f74; font-weight:500; }}
        .hs-dot {{ width:30px; height:30px; border-radius:50%; display:grid; place-items:center; font-size:.85rem;
                   border:2px solid #b9c9bf; background:#fff; color:#6b7f74; }}
        .hs-step.done .hs-dot {{ background:var(--hs-mid); border-color:var(--hs-mid); color:#fff; }}
        .hs-step.active .hs-dot {{ background:var(--hs-dark); border-color:var(--hs-dark); color:#fff; }}
        .hs-step.active {{ color:var(--hs-dark); font-weight:700; }}
        .hs-line {{ width:34px; height:2px; background:#b9c9bf; }}
        div[class*="st-key-card"] {{ background:var(--hs-card); border-radius:24px; border:1px solid #dbe6de;
                  box-shadow:0 10px 28px rgba(35,70,55,.10); padding:.4rem .6rem; }}
        div[class*="st-key-sel"] {{ border:3px solid var(--hs-dark) !important; }}
        div[data-testid="stMetric"] {{ background:#fff; border:1px solid #dbe6de; border-radius:16px; padding:.7rem 1rem; }}
        .stButton > button {{ border-radius:999px; font-weight:600; padding:.55rem 1.4rem; }}
        .stButton > button[kind="primary"] {{ background:var(--hs-dark); border-color:var(--hs-dark); color:#fff; }}
        .hs-chip {{ display:inline-block; padding:.3rem .8rem; border-radius:999px; background:var(--hs-soft);
                    color:var(--hs-dark); font-weight:600; font-size:.85rem; margin:.15rem .25rem; }}
        .hs-chip.warn {{ background:#fdf0d5; color:#7a5200; }}
        .hs-tile {{ height:150px; border-radius:18px; display:grid; place-items:center; font-size:3.4rem; }}
        .hs-ring {{ width:104px; height:104px; border-radius:50%; display:grid; place-items:center;
                    background:conic-gradient(var(--hs-dark) calc(var(--p)*1%), #dfe8e2 0); }}
        .hs-ring span {{ width:78px; height:78px; border-radius:50%; background:#fff; display:grid; place-items:center;
                         font-family:{serif}; font-weight:700; color:var(--hs-dark); font-size:1.2rem; }}
        .hs-row {{ display:flex; gap:12px; align-items:flex-start; margin:.7rem 0; }}
        .hs-ck {{ min-width:30px; height:30px; border-radius:50%; display:grid; place-items:center; font-size:.9rem;
                  border:2px solid #cfdad2; color:#fff; }}
        .hs-row.done .hs-ck {{ background:var(--hs-mid); border-color:var(--hs-mid); }}
        .hs-row.now .hs-ck {{ border-color:var(--hs-dark); border-top-color:transparent; animation:hsspin 1s linear infinite; }}
        .hs-row.todo {{ opacity:.5; }}
        .hs-row b {{ display:block; color:#1f3329; }} .hs-row small {{ color:#6b7f74; }}
        .hs-bar {{ display:flex; height:12px; border-radius:99px; overflow:hidden; margin:.5rem 0; }}
        @keyframes hsspin {{ to {{ transform:rotate(360deg); }} }}
        /* ---- Large typography for projector / distance reading ---- */
        html {{ font-size: {"21px" if rtl else "19px"} !important; }}
        .stApp p, .stApp li, .stApp label, .stApp span, .stApp div[data-testid="stMarkdownContainer"] {{
            font-size: 1.08rem; line-height: {"1.9" if rtl else "1.6"}; }}
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ font-size: 1rem !important; }}
        .stApp h1 {{ font-size: 3.2rem !important; font-weight: 800 !important; }}
        .stApp h2 {{ font-size: 2.6rem !important; font-weight: 800 !important; }}
        .stApp h3 {{ font-size: 1.9rem !important; font-weight: 700 !important; }}
        .stApp h4 {{ font-size: 1.5rem !important; font-weight: 700 !important; }}
        div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {{
            font-size: 2.8rem !important; font-weight: 800 !important; line-height: 1.2 !important; }}
        div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * {{
            font-size: 1.2rem !important; font-weight: 600 !important; }}
        div[data-testid="stMetricDelta"], div[data-testid="stMetricDelta"] * {{ font-size: 1rem !important; }}
        .stButton > button, .stButton > button * {{ font-size: 1.15rem !important; }}
        button[data-baseweb="tab"] p {{ font-size: 1.25rem !important; font-weight: 600 !important; }}
        .hs-chip {{ font-size: 1.05rem !important; }} .hs-step {{ font-size: 1.05rem; }}
        .hs-row b {{ font-size: 1.15rem; }} .hs-row small {{ font-size: .95rem; }}
        .hs-logo {{ font-size: 2.1rem !important; }} .hs-ring span {{ font-size: 1.5rem; }}
        /* ---- Layout / typography fixes ---- */
        div.block-container {{ max-width: 95% !important; padding-top: 2rem !important; }}
        .hs-logo {{ white-space: nowrap !important; display: inline-block; }}
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {{ font-size: 1.3rem !important; font-weight: bold !important; }}
        div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {{ font-size: 2.4rem !important; }}
        [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
        [data-testid="stExpander"] p, [data-testid="stExpander"] li {{
            font-size: 1.2rem !important; line-height: 1.8 !important; }}
        [data-testid="stMetricDelta"], [data-testid="stMetricDelta"] *, [data-testid="stMetricLabel"] * {{
            white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }}
        {'''
        .stApp .block-container, .stApp [data-testid="stVerticalBlock"], .stApp [data-testid="stHorizontalBlock"],
        .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stExpander"], .stApp [data-testid="stMetric"], .stApp [data-testid="stAlert"],
        .stApp p, .stApp ul, .stApp ol, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp label {
            direction: rtl !important; text-align: right !important; }
        .stApp ul, .stApp ol { padding-right: 1.6rem !important; padding-left: 0 !important;
            list-style-position: outside !important; }
        .stApp iframe, .stApp .folium-map, .stApp .leaflet-container { direction: ltr !important; }
        ''' if rtl else ""}
        </style>
        """,
        unsafe_allow_html=True,
    )


def farm_area_feddan(bbox):
    mx, my = metres_per_degree(bbox[1], bbox[3])
    return (bbox[2] - bbox[0]) * mx * (bbox[3] - bbox[1]) * my / FEDDAN_M2


def stepper_html(step):
    names = [t("step_location"), t("step_services"), t("step_analyze"), t("step_results")]
    parts = []
    for i, n in enumerate(names, 1):
        cls = "done" if i < step else "active" if i == step else ""
        dot = "✓" if i < step else str(i)
        parts.append(f'<div class="hs-step {cls}"><span class="hs-dot">{dot}</span>{n}</div>')
        if i < len(names):
            parts.append('<div class="hs-line"></div>')
    return '<div class="hs-stepper">' + "".join(parts) + "</div>"


def wizard_header(step):
    c1, c2, c3 = st.columns([1.5, 2.6, 1.4], vertical_alignment="center")
    with c3:
        st.radio("🌐 Language / اللغة", ["English", "عربي"], key="lang_choice", horizontal=True)
    inject_css()   # after the radio so the first paint already uses the chosen language
    with c1:
        st.markdown('<span class="hs-logo">💧 HydroScan AI</span>', unsafe_allow_html=True)
    with c2:
        st.markdown(stepper_html(step), unsafe_allow_html=True)


def go(step):
    st.session_state["hs_step"] = step


# --- Step 1 -------------------------------------------------------------------------------------
HS_DEFAULTS = {"hs_project": GEE_PROJECT, "hs_start": None, "hs_days": 365, "hs_basis": BASIS_PEAK, "hs_ndvi_thr": 0.30,
               "hs_excl_water": True, "hs_min_acres": 1.0, "hs_sat_start": None}


def init_state():
    ss = st.session_state
    ss.setdefault("hs_step", 1)
    ss.setdefault("hs_bbox", None)
    ss.setdefault("hs_drawn", None)
    ss.setdefault("hs_services", {"water": True, "crops": False, "soil": True})
    today = dt.date.today()
    for k, v in HS_DEFAULTS.items():
        if v is None:
            v = today - dt.timedelta(days=365 if k == "hs_start" else 60)
        ss.setdefault(k, v)
        ss[k] = ss[k]   # keep values alive while the widgets are not on screen


def location_step():
    ss = st.session_state
    st.markdown(f"## {t('welcome')}")
    st.caption(t("welcome_sub"))

    fg = folium.FeatureGroup(name="Your farm")
    if ss["hs_bbox"]:
        w, s, e, n = ss["hs_bbox"]
        folium.Rectangle([[s, w], [n, e]], color="#19c3e6", weight=2, dash_array="6", fill=False).add_to(fg)
    out = st_folium(base_map(), feature_group_to_add=[fg], layer_control=folium.LayerControl(collapsed=True),
                    height=520, use_container_width=True, returned_objects=["all_drawings"], key="hs_map")
    drawn = bbox_from_drawings((out or {}).get("all_drawings"))
    if drawn and drawn != ss["hs_drawn"]:
        ss["hs_drawn"], ss["hs_bbox"] = drawn, drawn
        st.rerun()
    if drawn is None and ss["hs_drawn"] is not None and out and out.get("all_drawings") == []:
        ss["hs_drawn"], ss["hs_bbox"] = None, None   # the user deleted the rectangle
        st.rerun()

    if ss["hs_bbox"]:
        fed = farm_area_feddan(ss["hs_bbox"])
        st.markdown(f'<span class="hs-chip">✓ {t("area_selected")}</span>'
                    f'<span class="hs-chip">{t("your_farm")} · {fed:,.1f} {t("feddan")} · '
                    f'{t("about_ha", ha=f"{fed * FEDDAN_M2 / 10000:,.1f}")}</span>', unsafe_allow_html=True)
    else:
        st.info(t("draw_hint"), icon=":material/crop_square:")

    with st.expander(t("coords_title")):
        cur = ss["hs_bbox"] or DEFAULT_BBOX
        c = st.columns(4)
        vals = [c[i].number_input(t(k), value=float(cur[i]), format="%.5f", step=0.001, key=f"hs_in_{k}")
                for i, k in enumerate(("west", "south", "east", "north"))]
        if st.button(t("apply_coords")):
            ss["hs_bbox"] = tuple(round(float(v), 6) for v in vals)
            st.rerun()

    with st.expander(t("adv_settings")):
        st.text_input(t("ee_project"), key="hs_project", help=t("ee_help"))
        a, b = st.columns(2)
        a.date_input(t("period_start"), key="hs_start")
        b.slider(t("period_days"), 30, 730, step=5, key="hs_days")
        st.radio(t("basis"), [BASIS_PEAK, BASIS_MEDIAN], key="hs_basis",
                 format_func=lambda v: t("basis_peak") if v == BASIS_PEAK else t("basis_median"))
        a, b = st.columns(2)
        a.slider(t("ndvi_thr"), 0.15, 0.70, step=0.01, key="hs_ndvi_thr")
        b.slider(t("min_acres"), 0.2, 20.0, step=0.2, key="hs_min_acres")
        a, b = st.columns(2)
        a.checkbox(t("excl_water"), key="hs_excl_water")
        b.date_input(t("sat_start"), key="hs_sat_start")

    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        st.button(t("confirm") + " →", type="primary", width="stretch", disabled=not ss["hs_bbox"],
                  on_click=go, args=(2,), key="hs_go2")


# --- Step 2 -------------------------------------------------------------------------------------
def toggle_service(name):
    sv = st.session_state["hs_services"]
    st.session_state["hs_services"] = {**sv, name: not sv[name]}


def services_step():
    ss = st.session_state
    st.markdown(f"<h2 style='text-align:center'>{t('services_title')}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;color:#587063'>{t('services_sub')}</p>", unsafe_allow_html=True)
    cards = [("water", "💧", "linear-gradient(135deg,#bfe3a8,#7fbf6e)", "svc_water", "svc_water_sub", "svc_water_d"),
             ("crops", "🌾", "linear-gradient(135deg,#f3e3a1,#cfd98a)", "svc_crops", "svc_crops_sub", "svc_crops_d"),
             ("soil", "🪨", "linear-gradient(135deg,#d2b48c,#8b6a47)", "svc_soil", "svc_soil_sub", "svc_soil_d")]
    cols = st.columns(3)
    for col, (key, icon, grad, title, sub, desc) in zip(cols, cards):
        on = ss["hs_services"][key]
        with col, st.container(key=f"card_{'sel_' if on else ''}svc_{key}"):
            st.markdown(f'<div class="hs-tile" style="background:{grad}">{icon}</div>', unsafe_allow_html=True)
            st.markdown(f"### {t(title)}")
            st.caption(f"**{t(sub)}**")
            st.write(t(desc))
            st.button(("✓ " + t("selected")) if on else t("select"), key=f"hs_tg_{key}", width="stretch",
                      type="primary" if on else "secondary", on_click=toggle_service, args=(key,))
    n = sum(ss["hs_services"].values())
    a, b, c = st.columns([1.2, 1.2, 1], vertical_alignment="center")
    with a:
        fed = farm_area_feddan(ss["hs_bbox"])
        st.markdown(f'<span class="hs-chip">📍 {t("your_farm")} · {fed:,.1f} {t("feddan")}</span>', unsafe_allow_html=True)
        st.button(t("change"), key="hs_chg", on_click=go, args=(1,))
    with b:
        st.button(t("analyze_btn") + " →", type="primary", width="stretch", disabled=n == 0,
                  on_click=go, args=(3,), key="hs_go3")
    with c:
        st.markdown(f"**{t('n_selected', n=n)}**")


# --- Step 3: run the untouched backend -----------------------------------------------------------
def progress_html(pct, stage):
    items = [("st_sat", "Sentinel-2"), ("st_ndvi", "NDVI · NDWI"), ("st_rain", "NASA GPM · NASA POWER"),
             ("st_smap", "NASA SMAP"), ("st_prep", "")]
    rows = []
    for i, (k, sub) in enumerate(items):
        cls = "done" if i < stage else "now" if i == stage else "todo"
        rows.append(f'<div class="hs-row {cls}"><span class="hs-ck">{"✓" if i < stage else ""}</span>'
                    f'<div><b>{t(k)}</b><small>{sub}</small></div></div>')
    return (f'<div style="display:flex;gap:18px;align-items:center"><div class="hs-ring" style="--p:{pct}">'
            f'<span>{pct}%</span></div><div><h3 style="margin:0">{t("analyzing")}</h3>'
            f'<small>{t("wait")}</small></div></div><div style="margin-top:14px">{"".join(rows)}</div>')


def analyze_step():
    ss = st.session_state
    bbox = ss["hs_bbox"]
    left, right = st.columns([2.5, 1])
    with left:
        m = folium.Map(tiles=None, zoom_control=False)
        folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                         attr="Esri World Imagery").add_to(m)
        folium.Rectangle([[bbox[1], bbox[0]], [bbox[3], bbox[2]]], color="#ffffff", weight=3, fill=True,
                         fill_opacity=0.15).add_to(m)
        m.fit_bounds([[bbox[1], bbox[0]], [bbox[3], bbox[2]]])
        st_folium(m, height=480, use_container_width=True, returned_objects=[], key="hs_prog_map")
    with right, st.container(key="card_progress"):
        box = st.empty()
        sv = ss["hs_services"]
        tags = "".join(f'<span class="hs-chip">{t(k)}</span>' for flag, k in
                       ((sv["water"], "tag_water"), (sv["soil"], "tag_soil"), (sv["crops"], "tag_crops")) if flag)
        st.markdown(tags, unsafe_allow_html=True)
        if not sv["crops"]:
            st.caption("🌱 " + t("crops_skipped"))
    project, start, days = ss["hs_project"].strip() or GEE_PROJECT, ss["hs_start"], int(ss["hs_days"])
    peak = ss["hs_basis"] == BASIS_PEAK
    sat_start = ss["hs_sat_start"]
    clat, clon = (bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2
    result = nasa = None
    try:
        box.markdown(progress_html(5, 0), unsafe_allow_html=True)
        if project:
            ind = fetch_indices(project, bbox, start.isoformat(), days, peak)
            box.markdown(progress_html(35, 1), unsafe_allow_html=True)
            result = classify(ind, ss["hs_ndvi_thr"], ss["hs_excl_water"], ss["hs_min_acres"])
            result.update(mode="real", scale=ind["scale"], scenes=ind["scenes"], period=ind["period"], peak=peak)
        else:
            result = dict(demo_area(bbox), mode="demo")
            box.markdown(progress_html(35, 1), unsafe_allow_html=True)
        box.markdown(progress_html(55, 2), unsafe_allow_html=True)
        nasa = get_nasa_profile(clat, clon, sat_start, project, live=True)
        st.session_state.setdefault("_nasa_live", {})[(round(clat, 2), round(clon, 2), sat_start.isoformat())] = nasa
        box.markdown(progress_html(80, 3), unsafe_allow_html=True)
        box.markdown(progress_html(95, 4), unsafe_allow_html=True)
    except Exception as exc:
        st.error(t("err_process", e=str(exc)))
        st.button(t("back"), on_click=go, args=(2,), key="hs_err_back")
        return
    ss["hs_result"], ss["hs_nasa"] = result, nasa
    ss["hs_step"] = 4
    st.rerun()


# --- Step 4: results ----------------------------------------------------------------------------
def band_overlay(result):
    """Display-only: colour irrigated pixels by NDVI band (healthy / needs attention / dry)."""
    nd, mask = result["ndvi"], result["mask"]
    rgba = np.zeros(nd.shape + (4,), np.uint8)
    n0 = np.nan_to_num(nd, nan=-1)
    rgba[mask & (n0 >= BAND_HEALTHY)] = (30, 140, 80, 190)
    rgba[mask & (n0 >= BAND_ATTN) & (n0 < BAND_HEALTHY)] = (240, 180, 50, 200)
    rgba[mask & (n0 < BAND_ATTN)] = (225, 100, 60, 210)
    return rgba


def tag_pill(live):
    return f'<span class="hs-chip">{"✓ " + t("measured") if live else "≈ " + t("estimated")}</span>'


def nasa_cards(nasa, stacked=False):
    items = [("rain", "🌧️ " + t("rain_30"), f"{nasa['rain']:.1f} mm", f"NASA GPM · {t('per_30')}"),
             ("smap", "💧 " + t("smap_title"), f"{nasa['smap']:.0f} %", f"NASA SMAP · {t('surface')}"),
             ("temp", "🌡️ " + t("temp_title"), f"{nasa['temp']:.1f} °C", f"NASA POWER · {t('mean')}")]
    # stacked = narrow side column, one metric per row so the values are not truncated
    cols = [st.container() for _ in items] if stacked else st.columns(3)
    for col, (key, label, value, delta) in zip(cols, items):
        col.metric(label, value, delta, delta_color="off")
        col.markdown(tag_pill(nasa["live"][key]), unsafe_allow_html=True)


def results_water(bbox, result, nasa):
    st.markdown(f"## {t('title_water')}")
    if result["mode"] == "demo":
        st.warning(t("demo_warn"))
    left, right = st.columns([2.5, 1])
    west, south, east, north = bbox
    with left:
        m = folium.Map(tiles=None, zoom_control=True)
        folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                         attr="Esri World Imagery", name=t("sat_view")).add_to(m)
        folium.Rectangle([[south, west], [north, east]], color="#ffffff", weight=3, fill=False).add_to(m)
        if result["mode"] == "real" and result.get("ndvi") is not None:
            w_, s_, e_, n_ = result["bounds"]
            folium.raster_layers.ImageOverlay(band_overlay(result), bounds=[[s_, w_], [n_, e_]], opacity=1,
                                              name=t("legend")).add_to(m)
        elif result["geojson"]:
            folium.GeoJson(result["geojson"], style_function=lambda _f: {"color": "#fff", "weight": 1,
                           "fillColor": "#1e8c50", "fillOpacity": 0.55}).add_to(m)
        m.fit_bounds([[south, west], [north, east]])
        st_folium(m, height=540, use_container_width=True, returned_objects=[], key="hs_res_map")
        st.markdown(f'<span class="hs-chip" style="background:#1e8c50;color:#fff">{t("cond_healthy")}</span>'
                    f'<span class="hs-chip" style="background:#f0b432">{t("cond_attn")}</span>'
                    f'<span class="hs-chip" style="background:#e1643c;color:#fff">{t("cond_dry")}</span>',
                    unsafe_allow_html=True)
        st.button(t("new_analysis"), on_click=go, args=(1,), key="hs_new")   # directly under the legend
    with right:
        with st.container(key="card_cond"):
            st.markdown(f"### {t('card_cond')}")
            if result["mode"] == "real" and result["mask"].any():
                nd, mk = result["ndvi"], result["mask"]
                v = nd[mk]
                h, a = float((v >= BAND_HEALTHY).mean()), float(((v >= BAND_ATTN) & (v < BAND_HEALTHY)).mean())
                d = max(0.0, 1 - h - a)
                st.write(t("cond_good") if h >= 0.6 else t("cond_mixed"))
                st.markdown(f'<div class="hs-bar"><div style="width:{h*100:.0f}%;background:#1e8c50"></div>'
                            f'<div style="width:{a*100:.0f}%;background:#f0b432"></div>'
                            f'<div style="width:{d*100:.0f}%;background:#e1643c"></div></div>', unsafe_allow_html=True)
                st.caption(t("cond_caption", h=f"{h*100:.0f}", a=f"{a*100:.0f}", d=f"{d*100:.0f}"))
            else:
                st.info(t("cond_none"))
        with st.container(key="card_irr"):
            st.markdown(f"### {t('irr_detected')}")
            pct = 100 * result["irrigated_acres"] / max(result["total_acres"], 1e-9)
            fed = result["irrigated_acres"] * ACRE_M2 / FEDDAN_M2
            if pct < 5 and result["mode"] == "real":
                st.write(t("irr_little"))
            else:
                st.write("✅ " + t("irr_msg", pct=f"{pct:.0f}", fed=f"{fed:,.0f}", n=f"{result['n_fields']:,}"))
        with st.container(key="card_nasa_water"):
            nasa_cards(nasa, stacked=True)
            if nasa["rain"] <= 5:
                st.success(t("nasa_ok"))
            else:
                st.warning(t("nasa_warn"))
            if not any(nasa["live"].values()):
                st.caption(t("nasa_est"))


def results_crops(bbox, result, nasa):
    ss = st.session_state
    st.markdown(f"## {t('title_crops')}")
    st.markdown(f'<span class="hs-chip warn">≈ {t("suggestions")}</span>', unsafe_allow_html=True)
    clat = (bbox[1] + bbox[3]) / 2
    d0 = ss["hs_sat_start"]
    strategies, scores, order, wstress = recommend_rotations(nasa)
    best = order[0]
    top = strategies[best]
    season = "Summer" if ((5 <= (d0 + dt.timedelta(days=15)).month <= 10) == (clat >= 0)) else "Winter"
    other = "Winter" if season == "Summer" else "Summer"
    sname = lambda s: t("summer") if s == "Summer" else t("winter")
    stress_key = "high" if wstress > 0.6 else "moderate" if wstress > 0.35 else "low"
    stress = t(f"stress_{stress_key}")
    top_name = tr_map(STRAT_AR, top["name"])
    plan = SEASON_CROPS[top["name"]]

    left, right = st.columns([1, 2.6])
    with left, st.container(key="card_plan_uses"):
        st.markdown(f"**{t('plan_uses')}**")
        for k in ("uses1", "uses2", "uses3", "uses4"):
            st.markdown(f"- {t(k)}")
        st.caption(f"{t('source')}: " + " · ".join(sorted({nasa['src'][k] for k in ('rain', 'temp', 'smap')})))
    with right:
        cols = st.columns(3)
        for col, (lbl, s) in zip(cols, (("season_now", season), ("season_next", other), ("season_then", season))):
            with col, st.container(key=f"card_{lbl}"):
                st.markdown(f"<div class='hs-serif' style='font-weight:700;color:#2b4d3f'>{t(lbl, s=sname(s))}</div>",
                            unsafe_allow_html=True)
                st.markdown(f"#### {top['icon']} {tr_crops(plan[s])}")
        st.success(f"**{top['icon']} {top_name}** · {t('fit', s=scores[best])} · "
                   f"{t('for_season', season=sname(season))}")

    st.markdown(f"### {t('why_title')}")
    rain, temp, smap = nasa["rain"], nasa["temp"], nasa["smap"]
    rl, tl, sl = nasa_labels(nasa)
    L = lambda x: tr_map(LBL_AR, x).lower() if lang() == "en" else tr_map(LBL_AR, x)
    st.markdown("\n".join([
        "- " + t("b_rain", v=rain, lbl=L(rl), txt=t("rain_t1") if rain <= 5 else t("rain_t2") if rain <= 15 else t("rain_t3")),
        "- " + t("b_temp", v=temp, lbl=L(tl), txt=t("temp_t1") if temp >= 33 else t("temp_t2") if temp >= 22 else t("temp_t3")),
        "- " + t("b_smap", v=smap, lbl=L(sl), txt=t("smap_t1") if smap < 20 else t("smap_t2") if smap < 30 else t("smap_t3")),
        "- " + t("b_stress", v=wstress, lvl=stress),
    ]))
    if lang() == "en":
        st.markdown(t("therefore", why=top["why"]))
    b1, b2 = st.columns(2)
    with b1:
        st.markdown(f"**{t('benefits')}**")
        for b in top["benefits"]:
            st.markdown(f"- {tr_map(BENEFIT_AR, b)}")
    with b2:
        st.caption(f"{t('watch')}: {tr_map(WATCH_AR, top['watch'])}")
    with st.expander(t("other_strats")):
        for i in order[1:]:
            s = strategies[i]
            line = f"**{s['icon']} {tr_map(STRAT_AR, s['name'])}** ({scores[i]}/100)"
            if lang() == "en":
                line += f": {s['why']}"
            st.markdown(line)
    with st.expander(t("rules_title")):
        st.markdown(t("rules_body"))


def results_soil(bbox, result, nasa):
    st.markdown(f"## {t('title_soil')}")
    c1, c2, c3 = st.columns(3)
    c1.metric("💧 " + t("smap_title"), f"{nasa['smap']:.0f} %", f"NASA SMAP · {t('surface')}", delta_color="off")
    c1.markdown(tag_pill(nasa["live"]["smap"]), unsafe_allow_html=True)
    if result["mode"] == "real":
        c2.metric(t("ndvi_irr"), f"{result['ndvi_in']:.2f}")
        c3.metric(t("ndvi_all"), f"{result['ndvi_all']:.2f}")
        hist, edges = result["hist"]
        if len(hist):
            st.markdown(f"### {t('ndvi_dist')}")
            st.bar_chart(pd.Series(hist, index=[f"{x:.2f}" for x in edges[:-1]]))
            st.caption(t("ndvi_note"))
        st.caption(f"{t('scenes')}: {result['scenes']}")
    else:
        st.info(t("ndvi_none"))
    st.caption(t("soil_note"))


def results_step():
    ss = st.session_state
    bbox, result, nasa, sv = ss["hs_bbox"], ss["hs_result"], ss["hs_nasa"], ss["hs_services"]
    fed = farm_area_feddan(bbox)
    st.markdown(f'<span class="hs-chip">📍 {t("your_farm")} · {fed:,.1f} {t("feddan")}</span>', unsafe_allow_html=True)
    tabs_def = [(k, lab, fn) for k, lab, fn in (("water", "tab_water", results_water), ("crops", "tab_crops", results_crops),
                                               ("soil", "tab_soil", results_soil)) if sv[k]]
    for tab, (_, _, fn) in zip(st.tabs([t(lab) for _, lab, _ in tabs_def]), tabs_def):
        with tab:
            fn(bbox, result, nasa)
    if not sv["water"]:   # with the water tab the button sits under the map legend instead
        st.button(t("new_analysis"), on_click=go, args=(1,), key="hs_new")


def wizard():
    init_state()
    step = st.session_state["hs_step"]
    wizard_header(step)
    {1: location_step, 2: services_step, 3: analyze_step, 4: results_step}[step]()


# --------------------------------------------------------------------------- #
# Routing. Default = the new bilingual wizard. The original two tools stay available, unchanged, in Expert mode.
# --------------------------------------------------------------------------- #
expert = st.sidebar.toggle("Expert mode (original tools) / الوضع المتقدم", key="expert_mode")
if expert:
    view = st.sidebar.radio("Choose Tool:", ["Irrigation Detection", "Crop Rotation Strategy"])
    if view == "Irrigation Detection":
        if st.session_state.get("_prev_view") != "Irrigation Detection":
            st.session_state["drawn_bbox"] = None   # fresh map component: do not mistake its empty drawings for a deletion
        irrigation_detection_view()
    else:
        crop_rotation_view()
    st.session_state["_prev_view"] = view
else:
    wizard()
