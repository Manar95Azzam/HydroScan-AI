"""
app.py -- HydroScan AI (NASA Space Apps: Adapting Farms with NASA Data)

Run with:   python -m streamlit run app.py
Install:    pip install streamlit streamlit-folium folium rasterio scipy scikit-learn pandas numpy joblib

Sidebar: navigation (View 1 / View 2) + a global region selector.
View 1  Irrigation Detection       real joblib model inference (flatten -> predict -> reshape -> Folium overlay)
View 2  Crop Rotation Strategies   deterministic rule-based engine on FIXED per-region NASA values

Everything is deterministic: no random buttons, no per-run randomness.
NASA values below are FIXED illustrative values per region (not a live feed). Replace REGIONS[...] with real
GPM / POWER / SMAP downloads to go live. Imagery is ALWAYS a real GeoTIFF: data/<slug>.tif for a preset region,
or an uploaded / local .tif for Custom Upload. No synthetic scenes.
"""

import math
import os
import zlib

import folium
import joblib
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform, reproject
from streamlit_folium import st_folium

from model_training import NODATA, make_features

ACRE_M2 = 4046.8564224
MODEL_CANDIDATES = ["model.pkl", "model.joblib"]
VIEWS = ["View 1: Irrigation Detection", "View 2: Crop Rotation Strategies"]
CUSTOM = "Custom Upload"

# ---- static, deterministic NASA values per region (30-day rain total, mean temperature, surface soil moisture)
REGIONS = {
    "Toshka, Egypt": {
        "Rain_GPM": 0.0, "Temp_POWER": 38.5, "Moisture_SMAP": 15.0, "center": (22.85, 31.30), "slug": "toshka",
        "layout": "pivots", "note": "Hyper-arid desert reclamation area in southern Egypt.",
    },
    "Nile Delta, Egypt": {
        "Rain_GPM": 2.1, "Temp_POWER": 30.0, "Moisture_SMAP": 28.0, "center": (30.90, 31.00), "slug": "nile_delta",
        "layout": "small_fields", "note": "Dense, intensively irrigated smallholder farmland.",
    },
    "California Central Valley, US": {
        "Rain_GPM": 12.5, "Temp_POWER": 26.0, "Moisture_SMAP": 35.0, "center": (36.70, -119.80), "slug": "california_central_valley",
        "layout": "blocks", "note": "Large mechanised fields with a Mediterranean climate.",
    },
}

st.set_page_config(page_title="HydroScan AI", page_icon="💧", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; max-width: 1300px;}
      h1 {color:#0b3d91; margin-bottom:0;}
      div[data-testid="stMetric"] {background:#fff; border:1px solid #e3ebf7; border-radius:14px;
             padding:.7rem 1rem; box-shadow:0 2px 8px rgba(20,60,140,.07);}
      div[data-testid="stMetricValue"] {color:#0b3d91;}
      .tag {display:inline-block; background:#fff3cd; color:#7a5b00; border:1px solid #ffe08a;
            border-radius:999px; padding:.05rem .7rem; font-size:.78rem; font-weight:600;}
      .top {display:inline-block; background:#d1f4dc; color:#0b6b2e; border:1px solid #9ad9b0;
            border-radius:999px; padding:.05rem .7rem; font-size:.8rem; font-weight:700;}
      .chip {display:inline-block; background:#eaf2ff; color:#0b3d91; border-radius:8px;
             padding:.05rem .55rem; margin:0 .25rem .25rem 0; font-size:.85rem; font-weight:600;}
      .ctx {border-left:5px solid #19a7ce; background:#f4fafd; border-radius:8px; padding:.9rem 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Sidebar: navigation, then the global region selector
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 💧 HydroScan AI")
    st.caption("Adapting farms with NASA data")
    view = st.radio("Navigate", VIEWS, key="view", label_visibility="collapsed")
    st.divider()
    region = st.selectbox("📍 Region", list(REGIONS) + [CUSTOM], key="region")
    if region != CUSTOM:
        lat, lon = REGIONS[region]["center"]
        st.caption(f"{REGIONS[region]['note']}  \n{lat:.2f}°, {lon:.2f}°")
    else:
        st.caption("Upload your own GeoTIFF in View 1.")


# --------------------------------------------------------------------------- #
# NASA values for the selected region (always the same for the same region / file name)
# --------------------------------------------------------------------------- #
def custom_nasa(filename):
    """Deterministic mock values from a stable hash of the file name (crc32, NOT Python's salted hash())."""
    store = st.session_state.setdefault("custom_nasa", {})
    if filename not in store:
        rng = np.random.default_rng(zlib.crc32(filename.encode("utf-8")))
        store[filename] = {
            "Rain_GPM": round(float(rng.uniform(0, 15)), 1),
            "Temp_POWER": round(float(rng.uniform(22, 40)), 1),
            "Moisture_SMAP": round(float(rng.uniform(12, 38)), 0),
        }
    return store[filename]


def nasa_for(region):
    if region != CUSTOM:
        r = REGIONS[region]
        return {k: r[k] for k in ("Rain_GPM", "Temp_POWER", "Moisture_SMAP")}
    name = st.session_state.get("custom_name")
    return custom_nasa(name) if name else None


# --------------------------------------------------------------------------- #
# Raster + model helpers
# --------------------------------------------------------------------------- #
def find_local_tifs():
    """Real GeoTIFFs sitting next to app.py (or in its data/ folder)."""
    here = os.path.dirname(os.path.abspath(__file__))
    found = []
    for folder in (here, os.path.join(here, "data")):
        if os.path.isdir(folder):
            found += [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith((".tif", ".tiff"))]
    return found


def region_tif_path(region_name):
    """Real GeoTIFF for a preset region: data/<slug>.tif. There is NO synthetic fallback."""
    slug = REGIONS[region_name]["slug"]
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (os.getcwd(), here):
        p = os.path.join(base, "data", f"{slug}.tif")
        if os.path.isfile(p):
            return p
    return None


def band_indexes(descriptions, count):
    """1-based rasterio band indexes for (VV, VH, NDVI): by band description if present, else by position."""
    names = [(d or "").strip().upper() for d in descriptions]
    if all(n in names for n in ("VV", "VH", "NDVI")):
        return [names.index(n) + 1 for n in ("VV", "VH", "NDVI")], "Bands found by name: VV, VH, NDVI."
    if count < 3:
        raise ValueError(f"This file has {count} band(s); the model needs 3 (VV, VH, NDVI).")
    note = "Bands taken by position: 1=VV, 2=VH, 3=NDVI."
    if count > 3:
        note += f" (File has {count} bands; extra bands ignored.)"
    return [1, 2, 3], note


def find_model():
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in MODEL_CANDIDATES:
        for p in (cand, os.path.join(here, cand)):
            if os.path.isfile(p):
                return p
    return None


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def default_irrigated_classes(classes):
    names = [str(c) for c in classes]
    pick = [c for c, n in zip(classes, names) if "irrig" in n.lower() or "dense" in n.lower()]
    if pick:
        return pick
    if set(names) <= {"0", "1"}:
        return [c for c, n in zip(classes, names) if n == "1"]
    return [classes[-1]]


def run_inference(tif_bytes, model):
    """
    Real GeoTIFF inference, in the raster's NATIVE grid and CRS:
      read VV/VH/NDVI -> flatten -> drop NoData/NaN -> model.predict -> reshape to (H, W)
      -> reproject the class-index map to EPSG:4326 (nearest neighbour, so classes are never blended).
    Returns a dict with the class-index grid in lat/lon and its exact bounds.
    """
    classes = list(model.classes_)
    if len(classes) > 250:
        raise ValueError("Model has too many classes for the index map.")
    with MemoryFile(tif_bytes) as mf, mf.open() as src:
        if src.crs is None:
            raise ValueError("The GeoTIFF has no CRS, so it cannot be placed on the map.")
        idx, note = band_indexes(src.descriptions, src.count)
        stack = src.read(idx).astype("float32")                       # (3, H, W) native grid
        nd = src.nodata
        src_crs, src_transform, src_bounds = src.crs, src.transform, src.bounds
        H, W = src.height, src.width

        # ---- flatten, remove NoData / NaN / inf, predict, reshape back to (H, W)
        bad = ~np.isfinite(stack)
        for v in {NODATA, nd}:
            if v is not None and np.isfinite(v):
                bad |= stack == v
        valid = ~bad.any(axis=0)
        X_all = stack.reshape(3, -1).T                                 # (H*W, 3) = VV, VH, NDVI
        flat_valid = valid.ravel()
        n_in = getattr(model, "n_features_in_", 4)
        if n_in not in (3, 4):
            raise ValueError(f"The model expects {n_in} features; this app supplies 3 or 4.")
        if not flat_valid.any():
            raise ValueError("Every pixel is NoData; nothing to classify.")
        X = X_all[flat_valid]
        if n_in == 4:
            X = make_features(X)                                       # VV, VH, NDVI, VH-VV (as in training)
        class_index = {c: i for i, c in enumerate(classes)}
        pred_valid = np.array([class_index[p] for p in model.predict(X)], dtype=np.uint8)
        pred = np.full(H * W, 255, dtype=np.uint8)                     # 255 = NoData
        pred[flat_valid] = pred_valid
        pred = pred.reshape(H, W)

        # ---- reproject the prediction grid to EPSG:4326 (Folium's CRS)
        dst_crs = "EPSG:4326"
        if rasterio.crs.CRS.from_user_input(src_crs) == rasterio.crs.CRS.from_user_input(dst_crs):
            dst, dst_transform = pred, src_transform
            dst_w, dst_h = W, H
        else:
            dst_transform, dst_w, dst_h = calculate_default_transform(src_crs, dst_crs, W, H, *src_bounds)
            dst = np.full((dst_h, dst_w), 255, dtype=np.uint8)
            reproject(source=pred, destination=dst, src_transform=src_transform, src_crs=src_crs,
                      src_nodata=255, dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=255,
                      resampling=Resampling.nearest)
    # exact bounds of the destination grid (west, south, east, north)
    west, north = dst_transform * (0, 0)
    east, south = dst_transform * (dst_w, dst_h)
    return {"pred": dst, "classes": classes, "bounds": (west, south, east, north), "note": note,
            "src_shape": (H, W), "src_crs": str(src_crs)}


def pixel_area_m2(bounds, shape):
    w, s, e, n = bounds
    h_px, w_px = shape
    return ((e - w) / w_px * 111_320 * math.cos(math.radians((s + n) / 2))) * ((n - s) / h_px * 110_540)


def base_map(bounds):
    w_, s_, e_, n_ = bounds
    m = folium.Map(location=[(s_ + n_) / 2, (w_ + e_) / 2], zoom_start=13, tiles=None, control_scale=True)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri World Imagery", name="Satellite").add_to(m)
    m.fit_bounds([[s_, w_], [n_, e_]])
    return m


# =========================================================================== #
# VIEW 1: IRRIGATION DETECTION
# =========================================================================== #
if view == VIEWS[0]:
    st.title("Irrigation Detection")
    st.caption(f"Region: **{region}**")

    top_l, top_r = st.columns([3, 2])
    tif_bytes, tif_name = None, None   # the real GeoTIFF that will be analysed
    with top_l:
        if region == CUSTOM:
            upload = st.file_uploader("GeoTIFF (.tif) with VV, VH, NDVI", type=["tif", "tiff"], key="custom_upload")
            if upload is not None:
                tif_bytes, tif_name = upload.getvalue(), upload.name
            else:
                local = find_local_tifs()
                choice = st.selectbox("...or use a real local GeoTIFF", ["(none)"] + local,
                                      format_func=lambda p: p if p == "(none)" else os.path.relpath(p, os.path.dirname(os.path.abspath(__file__))))
                if choice != "(none)":
                    with open(choice, "rb") as f:
                        tif_bytes, tif_name = f.read(), os.path.basename(choice)
            if tif_name:
                st.session_state["custom_name"] = tif_name
        else:
            p = region_tif_path(region)
            if p:
                with open(p, "rb") as f:
                    tif_bytes, tif_name = f.read(), os.path.basename(p)
                st.info(f"Using real GeoTIFF `data/{tif_name}`. Press **Run AI Detection**.")
            else:
                st.error(f"No real imagery for this region: put a GeoTIFF at `data/{REGIONS[region]['slug']}.tif`, "
                         "or choose **Custom Upload**.")
    with top_r:
        model_path = find_model()
        if model_path:
            st.success(f"Model ready: `{os.path.basename(model_path)}`")
        else:
            st.error("No `model.pkl` / `model.joblib` next to app.py. Run `python model_training.py` first.")

    ready = model_path is not None and tif_bytes is not None
    go = st.button("🚀 Run AI Detection", type="primary", disabled=not ready)
    run_key = f"{region}|{tif_name or ''}"

    if go:
        with st.spinner("Reading raster, running the model and reprojecting to EPSG:4326..."):
            try:
                model = load_model(model_path)
                res = run_inference(tif_bytes, model)
                res["irr_classes"] = default_irrigated_classes(res["classes"])
                res["key"] = run_key
                res["source"] = f"{tif_name} ({res['src_crs']}, {res['src_shape'][1]}x{res['src_shape'][0]} px)"
                res["px_area"] = pixel_area_m2(res["bounds"], res["pred"].shape)
                st.session_state["det"] = res
            except Exception as exc:
                st.session_state.pop("det", None)
                st.error(f"Detection failed: {exc}")

    det = st.session_state.get("det")
    if det and det["key"] == run_key:
        with st.expander("Which model classes count as irrigated?"):
            chosen = st.multiselect("Classes", det["classes"], default=det["irr_classes"], key=f"pick_{run_key}")
        chosen = chosen or det["irr_classes"]
        valid = det["pred"] != 255
        chosen_idx = [det["classes"].index(c) for c in chosen]
        mask = np.isin(det["pred"], chosen_idx) & valid

        n_valid = int(valid.sum())
        irr_acres = mask.sum() * det["px_area"] / ACRE_M2
        tot_acres = n_valid * det["px_area"] / ACRE_M2
        share = irr_acres / max(tot_acres, 1e-9)

        m1, m2, m3 = st.columns(3)
        m1.metric("💧 Irrigated land", f"{irr_acres:,.0f} acres")
        m2.metric("Share of valid area", f"{100 * share:.1f} %")
        m3.metric("NoData excluded", f"{100 * (1 - n_valid / mask.size):.1f} %")

        map_col, ctx_col = st.columns([3, 1.3])
        with map_col:
            w_, s_, e_, n_ = det["bounds"]                      # exact EPSG:4326 bounds of the reprojected grid
            m = base_map(det["bounds"])
            rgba = np.zeros((*mask.shape, 4), np.uint8)         # transparent everywhere ...
            rgba[mask] = (30, 110, 255, 180)                    # ... except irrigated pixels (blue)
            folium.raster_layers.ImageOverlay(rgba, bounds=[[s_, w_], [n_, e_]], opacity=1.0,
                                              name="Irrigated land (model)").add_to(m)
            folium.LayerControl(collapsed=True).add_to(m)
            st_folium(m, height=540, use_container_width=True, returned_objects=[], key=f"map_{run_key}")
            st.caption(f"Source: {det['source']} · blue = model class {', '.join(map(str, chosen))} · {det['note']}")

        with ctx_col:
            st.markdown("#### 🛰️ NASA Context")
            nasa = nasa_for(region)
            if region == CUSTOM:
                st.markdown('<span class="tag">mock values from file name</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="tag">fixed illustrative values</span>', unsafe_allow_html=True)
            rain, smap = nasa["Rain_GPM"], nasa["Moisture_SMAP"]
            if share < 0.05:
                body = ("<b>AI detects little vegetation</b> in this scene, so there is no irrigation signal to "
                        "check against rainfall or soil moisture.")
            else:
                rain_txt = ("confirms <b>no meaningful rainfall</b>" if rain <= 5 else "shows <b>recent rainfall</b>")
                smap_txt = (f"shows <b>active soil moisture ({smap:.0f}%)</b>" if smap >= 25 else
                            f"shows <b>low regional soil moisture ({smap:.0f}%)</b>")
                if rain <= 5:
                    verdict = "Confirmed Human Irrigation"
                elif rain <= 20:
                    verdict = "Probable irrigation; recent rain may add to the greenness"
                else:
                    verdict = "Possibly rain-fed; irrigation not confirmed"
                body = (f"<b>AI detects high vegetation.</b><br>NASA GPM {rain_txt} ({rain:.1f} mm, 30 days).<br>"
                        f"NASA SMAP {smap_txt}.<br><br><b>Conclusion: {verdict}.</b>")
            st.markdown(f'<div class="ctx">{body}</div>', unsafe_allow_html=True)
            st.caption("SMAP is ~10 km coarse: it describes the region, not single fields.")
    elif tif_bytes is None:
        st.info("Select or upload a real `.tif` to begin.")
    else:
        st.info("Press **Run AI Detection** to analyse the GeoTIFF.")


# =========================================================================== #
# VIEW 2: CROP ROTATION STRATEGIES
# =========================================================================== #
else:
    st.title("Crop Rotation Strategies")
    nasa = nasa_for(region)
    if nasa is None:
        st.info("Custom Upload is selected. Upload a `.tif` in View 1 first; its file name defines this region's NASA values.")
        st.stop()

    rain, temp, smap = nasa["Rain_GPM"], nasa["Temp_POWER"], nasa["Moisture_SMAP"]
    label = region if region != CUSTOM else f"{CUSTOM} ({st.session_state['custom_name']})"
    st.caption(f"Region: **{label}**")
    st.markdown('<span class="tag">fixed illustrative NASA values · not a live feed</span>' if region != CUSTOM
                else '<span class="tag">mock values from a stable hash of the file name</span>', unsafe_allow_html=True)

    rain_lbl = "Very low" if rain <= 5 else "Moderate" if rain <= 15 else "High"
    temp_lbl = "Very hot" if temp >= 35 else "Hot" if temp >= 30 else "Warm" if temp >= 25 else "Mild"
    smap_lbl = "Very dry" if smap < 20 else "Moderate" if smap < 30 else "Moist"
    c1, c2, c3 = st.columns(3)
    c1.metric("🌧️ Rainfall · NASA GPM", f"{rain:.1f} mm", f"30-day total · {rain_lbl}", delta_color="off")
    c2.metric("🌡️ Temperature · NASA POWER", f"{temp:.1f} °C", f"mean · {temp_lbl}", delta_color="off")
    c3.metric("💧 Soil moisture · NASA SMAP", f"{smap:.0f} %", f"surface · {smap_lbl}", delta_color="off")

    # ---- deterministic rule-based engine -----------------------------------------
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
    order = sorted(range(3), key=lambda i: (-raw[i], i))  # ties broken by fixed order
    best = order[0]
    stress_word = "high" if water_stress > 0.6 else "moderate" if water_stress > 0.35 else "low"

    strategies = [
        {"name": "Water Conservation", "icon": "💧", "crops": ["Sorghum", "Barley", "Pearl millet"],
         "timing": [("☀️ Summer", "Sorghum, pearl millet (heat tolerant)"), ("❄️ Winter", "Barley (matures before the hottest months)")],
         "benefits": ["Lowest irrigation demand", "Tolerates heat and dry spells", "Keeps land productive when water is scarce"],
         "watch": "Lower market price and yield than wheat or maize.",
         "fit": (f"Water stress is **{stress_word}** here ({smap_lbl.lower()} soil at {smap:.0f}%, {temp_lbl.lower()} at {temp:.1f} °C, "
                 f"{rain:.1f} mm rain). Crops that need less irrigation water protect scarce supplies.")},
        {"name": "Soil Health", "icon": "🌱", "crops": ["Faba bean", "Berseem clover", "Chickpea", "Cowpea", "Alfalfa"],
         "timing": [("❄️ Winter", "Faba bean, berseem clover, chickpea (best window)"),
                    ("☀️ Summer", "Cowpea; alfalfa only with reliable irrigation")],
         "benefits": ["Fixes nitrogen, so less fertiliser is needed", "Breaks cereal pest and disease cycles", "Builds soil organic matter"],
         "watch": "Alfalfa uses a lot of water; avoid it when water stress is high.",
         "fit": (f"Soil moisture of {smap:.0f}% is {'in the range where legumes establish well' if moist_mid > 0.6 else 'not ideal for legume establishment, so irrigation must be managed'}, "
                 f"and {temp:.1f} °C is {'cool enough for most legumes' if temp < 32 else 'hot, so choose cool-season (winter) legumes'}.")},
        {"name": "Market Yield", "icon": "📈", "crops": ["Wheat", "Maize"],
         "timing": [("❄️ Winter", "Wheat (core crop in Egypt; best choice)"), ("☀️ Summer", "Maize, only if irrigation is guaranteed")],
         "benefits": ["Highest yield and income potential", "Strong, established markets", "Well-known farming practices"],
         "watch": "Highest water and fertiliser demand; heat at flowering cuts yield.",
         "fit": (f"Water stress is **{stress_word}** and {temp:.1f} °C is {'inside' if temp <= 30 else 'above'} the comfortable range for "
                 f"wheat and maize, with {rain:.1f} mm of recent rain and {smap:.0f}% soil moisture to support high-demand crops.")},
    ]

    top = strategies[best]
    st.markdown("### Recommendation")
    st.success(f"**{top['icon']} {top['name']}** is the best fit for {label} "
               f"(fit score {scores[best]}/100). Crops: {', '.join(top['crops'])}.")
    with st.expander(f"Why {top['name']} fits {label}", expanded=True):
        st.markdown(f"- **Rainfall (GPM): {rain:.1f} mm** - {rain_lbl.lower()}; "
                    f"{'almost no natural water input, so crops depend on irrigation.' if rain <= 5 else 'some natural water input, but not enough to rely on alone.' if rain <= 15 else 'substantial natural water input.'}")
        st.markdown(f"- **Temperature (POWER): {temp:.1f} °C** - {temp_lbl.lower()}; "
                    f"{'heat raises crop water demand and limits high-demand crops.' if temp >= 33 else 'within a workable range for most crops.' if temp >= 25 else 'mild.'}")
        st.markdown(f"- **Soil moisture (SMAP): {smap:.0f}%** - {smap_lbl.lower()}; "
                    f"{'little stored water in the root zone.' if smap < 20 else 'a moderate water reserve.' if smap < 30 else 'a healthy water reserve.'}")
        st.markdown(f"**Reasoning:** {top['fit']}")
        st.markdown("**Seasonal timing**")
        for season, text in top["timing"]:
            st.markdown(f"- {season}: {text}")
        st.markdown("**Benefits**")
        for b in top["benefits"]:
            st.markdown(f"- {b}")
        st.caption(f"Watch out: {top['watch']}")

    st.markdown("### All three strategies")
    cols = st.columns(3)
    for col, i in zip(cols, range(3)):
        s = strategies[i]
        with col, st.container(border=True):
            st.markdown('<span class="top">✅ Recommended</span>' if i == best else f"Priority {order.index(i) + 1}",
                        unsafe_allow_html=True)
            st.markdown(f"#### {s['icon']} {s['name']}")
            st.markdown("".join(f'<span class="chip">{c}</span>' for c in s["crops"][:3]), unsafe_allow_html=True)
            st.metric("Fit score", f"{scores[i]} / 100")
            st.progress(scores[i] / 100)
            st.markdown(f"**Why:** {s['fit']}")

    with st.expander("How the rules score each strategy"):
        st.markdown(
            """
**Water stress** = 45% dry soil (SMAP below 35%) + 35% heat (POWER above 25 °C) + 20% lack of rain (GPM below 10 mm).
- **Water Conservation** rises with water stress.
- **Soil Health** peaks at moderate soil moisture (~27%), cooler temperature and some rain.
- **Market Yield** rises when water stress is low and temperature is at or below 30 °C.

Same inputs always give the same answer. These are simple agronomic rules for decision support, **not** a validated yield model.
            """
        )
