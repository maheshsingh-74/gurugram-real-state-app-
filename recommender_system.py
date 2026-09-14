"""
Gurgaon apartment recommender system - cleaned and bug-fixed.
Requires: pip install scikit-learn joblib pandas numpy

Bugs fixed vs. the original recommender-system.ipynb (full audit covered
these in detail earlier in this conversation - summarized here):

  1. Row 22 (in the original file) is the CSV header duplicated as a data
     row - a scraping artifact, not real data. The original notebook
     dropped it with a hardcoded `.drop(22)`. This version detects it
     programmatically (any row that equals the header exactly), so it
     doesn't silently break if the raw file is ever re-scraped/reordered.

  2. Facilities extraction: the original regex r"'(.*?)'" truncates any
     facility name containing an apostrophe - confirmed "Children's Play
     Area" breaks it, corrupting facility lists on 19 rows. Fixed by using
     ast.literal_eval (which the original notebook already used correctly
     elsewhere - this is a consistency fix, not new machinery).

  3. Price parsing: the original parser only handled two-part ranges
     ("X - Y") and silently dropped single fixed prices with no dash
     (confirmed 94 such entries, e.g. "₹ 1.67 Cr", discarded entirely even
     though perfectly parseable). Fixed to handle both single values and
     ranges, including mixed Lac/Crore ranges (which the original DID
     already handle correctly - kept as-is).

  4. Distance-unit parsing: the original distance_to_meters() only
     recognized 'Km'/'KM'/'Meter'/'meter' via case-sensitive substring
     checks, silently failing on km/m/mtr(s)/min(s)/Metre/Minute(s) in any
     case. Verified this corrupted 20.3% of all 2,554 distance entries in
     the dataset, with 9+ properties having EVERY listed distance wrong.
     Fixed with a proper case-insensitive unit parser. Travel-TIME entries
     ("3 mins") are detected and excluded rather than silently miscoded as
     a physical distance.

  5. Location NAME fragmentation: 1,070 unique location strings exist for a
     much smaller set of real places - e.g. "Indira Gandhi International
     Airport" was split across 10 spellings (194 mentions total), diluting
     it into 10 sparse near-empty columns instead of one clean feature.
     Fixed with a curated canonical-name map for the confirmed high-
     frequency offenders. This is a hand-reviewed list, NOT blind fuzzy
     matching - automated similarity also flags genuinely DIFFERENT places
     that share words (e.g. "Dwarka Expressway" vs "KMP Expressway" vs
     "Western Peripheral Expressway" are three different real roads).
     CANONICAL_LOCATION_MAP below is deliberately conservative; extend it
     as more true duplicates are confirmed, but verify each one manually.

  6. Land plots (30 rows, no BHK at all) are now explicitly flagged via an
     `is_land` feature instead of silently falling through BHK-shaped
     parsing logic that assumes every row has a building type/area/price
     per BHK config.

  7. Missing location distances (property genuinely doesn't list that
     amenity - a true NaN, not a parsing failure) are filled with a
     PER-COLUMN sentinel (that column's own observed max distance x 1.2),
     not one flat 54,000m applied to every column regardless of scale -
     the original flat sentinel is what made bug #4 so damaging once it
     silently fired on data that was actually parseable.
"""

import ast
import difflib
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

RAW_CSV = "appartments.csv"
OUTPUT_DIR = "recommender_artifacts"

BHK_CONFIGS = ["1 RK", "1 BHK", "2 BHK", "3 BHK", "4 BHK", "5 BHK", "6 BHK", "Land"]

# Similarity blend weights - kept proportional to the original notebook's
# (30, 20, 8) weighting (facilities weighted highest, then price/area, then
# location), just normalized to sum to 1 for clarity. Tune freely.
FACILITY_WEIGHT = 30 / 58
PRICE_WEIGHT = 20 / 58
LOCATION_WEIGHT = 8 / 58

# --- fix #5: curated canonical-name map for CONFIRMED duplicates only -----
CANONICAL_LOCATION_MAP = {
    # Indira Gandhi International Airport - 10 spellings, 194 total mentions
    "Indira Gandhi International Airport": "Indira Gandhi International Airport",
    "Indira Gandhi Intl Airport": "Indira Gandhi International Airport",
    "IGI Airport": "Indira Gandhi International Airport",
    "Delhi International Airport": "Indira Gandhi International Airport",
    "Indira Gandhi Int. Airport": "Indira Gandhi International Airport",
    "Indira Gandhi Airport": "Indira Gandhi International Airport",
    "International Airport": "Indira Gandhi International Airport",
    "IG International Airport": "Indira Gandhi International Airport",
    "IGIA Airport": "Indira Gandhi International Airport",
    "Airport": "Indira Gandhi International Airport",

    # Dwarka Expressway - genuine spelling/case variants ONLY. Does NOT
    # include KMP Expressway / Western Peripheral Expressway / Delhi Jaipur
    # Expressway - those are different real roads, confirmed during audit.
    "Dwarka Expressway": "Dwarka Expressway",
    "Dwarka Expy": "Dwarka Expressway",
    "Dwaraka Expressway": "Dwarka Expressway",
    "Dwarka expressway": "Dwarka Expressway",
    "Dwarka Expressway Link Road": "Dwarka Expressway",
    "Dwarka expressway Basai crossing": "Dwarka Expressway",

    # Gurgaon Railway Station - genuine spelling/case variants ONLY. Does
    # NOT include Basai Dhankot / Farrukh Nagar / Bijwasan / Patli / Sealdah
    # / Kheri stations - those are different real stations, confirmed
    # during audit.
    "Gurgaon Railway Station": "Gurgaon Railway Station",
    "Gurgaon railway station": "Gurgaon Railway Station",
    "Gurugram Railway Station": "Gurgaon Railway Station",
    "Gurgaon Old Railway Station": "Gurgaon Railway Station",

    "SkyJumper Trampoline Park": "SkyJumper Trampoline Park",
    "SkyJumper Trampoline Park Gurgaon": "SkyJumper Trampoline Park",
}

KM_UNITS = {"km", "kms", "kilometer", "kilometers", "kilometre", "kilometres"}
M_UNITS = {"m", "mtr", "mtrs", "meter", "meters", "metre", "metres"}
TIME_UNITS = {"min", "mins", "minute", "minutes"}
# search (not match-from-start) so "Within 1.4km", "2.9 Km Away", and
# "10 minutes drive" are all found regardless of leading/trailing filler words
_DIST_RE = re.compile(r"([\d,.]+)\s*([A-Za-z]+)", re.IGNORECASE)
# discovered while test-running this pipeline against the real data: several
# entries give no number at all, just a qualitative "it's close" phrase
_QUALITATIVE_CLOSE_PHRASES = (
    "close proximity", "closeby", "close by", "in proximity",
    "located nearby", "within reach", "nearby",
)
QUALITATIVE_CLOSE_METERS = 200.0  # documented assumption: "very near, no exact figure given"


# ---------------------------------------------------------------------------
# 1. Load + drop the corrupted header-as-data row (fix #1)
# ---------------------------------------------------------------------------
def load_raw(path=RAW_CSV):
    df = pd.read_csv(path)
    header_dupe_mask = df.apply(lambda r: list(r.astype(str)) == list(df.columns), axis=1)
    n_dupes = int(header_dupe_mask.sum())
    if n_dupes:
        print(f"Dropping {n_dupes} corrupted header-as-data row(s) at index "
              f"{df.index[header_dupe_mask].tolist()}")
    df = df.loc[~header_dupe_mask].reset_index(drop=True)
    print(f"Loaded {len(df)} properties after cleanup.")
    return df


# ---------------------------------------------------------------------------
# 2. Facilities (fix #2)
# ---------------------------------------------------------------------------
def parse_facilities(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []


# ---------------------------------------------------------------------------
# 3. Location distances (fixes #4, #5, #7)
# ---------------------------------------------------------------------------
def canonicalize_location(name):
    name = name.strip()
    return CANONICAL_LOCATION_MAP.get(name, name)


def parse_distance(distance_str):
    """Returns (meters, category) where category is one of:
    "distance" (a real physical distance), "time" (travel time - not
    convertible to meters, excluded rather than miscoded), "qualitative"
    (a "close by"/"nearby" style phrase with no number - treated as
    QUALITATIVE_CLOSE_METERS rather than lumped in with truly-missing data),
    or "unparsed" (genuinely couldn't make sense of it, meters is None).
    """
    if not isinstance(distance_str, str):
        return None, "unparsed"
    s = distance_str.strip()

    m = _DIST_RE.search(s)
    if m:
        value_str, unit = m.groups()
        try:
            value = float(value_str.replace(",", ""))
        except ValueError:
            value = None
        if value is not None:
            unit_clean = unit.rstrip(".").lower()
            if unit_clean in KM_UNITS:
                return value * 1000, "distance"
            if unit_clean in M_UNITS:
                return value, "distance"
            if unit_clean in TIME_UNITS:
                return None, "time"

    # no unit matched at all - check for a qualitative "it's close" phrase
    # (discovered by actually running this against the real data: "Close
    # Proximity", "closeby", "In close proximity", "Within reach", etc.)
    s_lower = s.lower()
    if any(phrase in s_lower for phrase in _QUALITATIVE_CLOSE_PHRASES):
        return QUALITATIVE_CLOSE_METERS, "qualitative"

    # bare number with no unit at all (one confirmed occurrence: "1.4") -
    # every other entry in this dataset is km-scale, so km is the documented
    # assumption here, not m
    if re.fullmatch(r"[\d,.]+", s):
        try:
            return float(s.replace(",", "")) * 1000, "distance"
        except ValueError:
            pass

    return None, "unparsed"


def build_location_distance_matrix(df):
    records = {}
    dropped_time_entries = 0
    qualitative_close_entries = 0
    unparsed_entries = []

    for idx, val in df["LocationAdvantages"].items():
        try:
            raw_dict = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            raw_dict = {}

        distances = {}
        for raw_name, raw_dist in raw_dict.items():
            canon_name = canonicalize_location(raw_name)
            meters, category = parse_distance(raw_dist)
            if category == "time":
                dropped_time_entries += 1
                continue
            if category == "unparsed":
                unparsed_entries.append((idx, raw_name, raw_dist))
                continue
            if category == "qualitative":
                qualitative_close_entries += 1
            # if canonicalization merges two spellings that BOTH appear in
            # the same row, keep the closer (smaller) of the two distances
            if canon_name not in distances or meters < distances[canon_name]:
                distances[canon_name] = meters
        records[idx] = distances

    print(f"Distance parsing: {dropped_time_entries} travel-time entries excluded "
          f"(not a physical distance), {qualitative_close_entries} qualitative "
          f"'close by'/'nearby' phrases treated as {QUALITATIVE_CLOSE_METERS:.0f}m, "
          f"{len(unparsed_entries)} entries genuinely unparseable")
    if unparsed_entries:
        print("  unparsed entries:", unparsed_entries)

    location_df = pd.DataFrame.from_dict(records, orient="index")
    location_df.index = df.loc[location_df.index, "PropertyName"].values
    return location_df


def fill_missing_distances(location_df):
    """Missing = property genuinely doesn't list that amenity. Filled with a
    PER-COLUMN sentinel instead of one flat number for every column - see
    fix #7 in the module docstring."""
    filled = location_df.copy()
    for col in filled.columns:
        observed_max = filled[col].max(skipna=True)
        sentinel = observed_max * 1.2 if pd.notna(observed_max) else 20000.0
        filled[col] = filled[col].fillna(sentinel)
    return filled


# ---------------------------------------------------------------------------
# 4. Price / area / building type (fixes #3, #6)
# ---------------------------------------------------------------------------
def _clean_num(s):
    return float(s.replace(",", "").replace("sq.ft.", "").strip())


def parse_price_details(detail_str):
    try:
        details = ast.literal_eval(detail_str)
    except (ValueError, SyntaxError):
        return {}

    extracted = {}
    for bhk, detail in details.items():
        extracted[f"building_type_{bhk}"] = detail.get("building_type") or ("Land" if bhk == "Land" else "")

        area = (detail.get("area") or "").strip()
        area_parts = [p.strip() for p in area.split("-")]
        try:
            if len(area_parts) == 1 and area_parts[0]:
                v = _clean_num(area_parts[0])
                extracted[f"area_low_{bhk}"] = v
                extracted[f"area_high_{bhk}"] = v
            elif len(area_parts) == 2:
                extracted[f"area_low_{bhk}"] = _clean_num(area_parts[0])
                extracted[f"area_high_{bhk}"] = _clean_num(area_parts[1])
        except ValueError:
            extracted[f"area_low_{bhk}"] = None
            extracted[f"area_high_{bhk}"] = None

        price_range = (detail.get("price-range") or "").strip()
        if price_range and price_range != "Price on Request":
            price_parts = [p.strip() for p in price_range.split("-")]
            try:
                if len(price_parts) == 1:
                    # fix #3: single fixed price (no dash) - the original
                    # parser silently dropped these (94 entries, confirmed)
                    raw = price_parts[0].replace("₹", "").strip()
                    is_lac = raw.endswith("L")
                    v = float(raw.replace("Cr", "").replace("L", "").strip())
                    if is_lac:
                        v /= 100
                    extracted[f"price_low_{bhk}"] = v
                    extracted[f"price_high_{bhk}"] = v
                elif len(price_parts) == 2:
                    lo_raw, hi_raw = price_parts
                    lo = float(lo_raw.replace("₹", "").replace("Cr", "").replace("L", "").strip())
                    hi = float(hi_raw.replace("₹", "").replace("Cr", "").replace("L", "").strip())
                    if lo_raw.endswith("L"):
                        lo /= 100
                    if hi_raw.endswith("L"):
                        hi /= 100
                    extracted[f"price_low_{bhk}"] = lo
                    extracted[f"price_high_{bhk}"] = hi
            except ValueError:
                extracted[f"price_low_{bhk}"] = None
                extracted[f"price_high_{bhk}"] = None

    return extracted


def build_price_area_features(df):
    rows = []
    for _, row in df.iterrows():
        features = parse_price_details(row["PriceDetails"])
        new_row = {"PropertyName": row["PropertyName"]}
        for config in BHK_CONFIGS:
            new_row[f"building_type_{config}"] = features.get(f"building_type_{config}")
            new_row[f"area_low_{config}"] = features.get(f"area_low_{config}")
            new_row[f"area_high_{config}"] = features.get(f"area_high_{config}")
            new_row[f"price_low_{config}"] = features.get(f"price_low_{config}")
            new_row[f"price_high_{config}"] = features.get(f"price_high_{config}")
        # fix #6: explicit land flag instead of silently falling through
        # BHK-shaped logic
        new_row["is_land"] = int("BHK" not in str(row["PropertySubName"]))
        rows.append(new_row)
    return pd.DataFrame(rows).set_index("PropertyName")


# ---------------------------------------------------------------------------
# 5. Build all three similarity matrices, and a filter-friendly metadata
#    table (locality / price range / BHK configs / link) for the search page
# ---------------------------------------------------------------------------
def parse_locality(subname):
    """Extracts the locality/sector text from PropertySubName, e.g.
    '3 BHK Apartment in Sector 86, Gurgaon' -> 'Sector 86'.
    Handles both '... in X, Gurgaon' and '... in X Gurgaon' (no comma -
    both forms exist in the real data)."""
    m = re.search(r"\bin\s+(.*?)\s*,?\s*Gurgaon\s*$", str(subname), flags=re.IGNORECASE)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return re.sub(r",?\s*Gurgaon\s*$", "", str(subname), flags=re.IGNORECASE).strip()


def build_properties_meta(df):
    """Everything a filter-based search page needs (locality, overall price
    range, BHK configs, property types, link) WITHOUT re-running the full
    parsing pipeline - saved alongside the similarity matrices for the
    Streamlit Recommendations page to load directly."""
    price_area_df = build_price_area_features(df)
    price_area_df = price_area_df.reindex(df["PropertyName"])

    records = []
    for name, row in price_area_df.iterrows():
        bhk_configs = []
        prices = []
        property_types = set()
        for config in BHK_CONFIGS:
            lo, hi = row.get(f"price_low_{config}"), row.get(f"price_high_{config}")
            if pd.notna(lo) and pd.notna(hi):
                bhk_configs.append(config)
                prices.extend([lo, hi])
            btype = row.get(f"building_type_{config}")
            if isinstance(btype, str) and btype.strip():
                property_types.add(btype.strip())
        subname = df.loc[df["PropertyName"] == name, "PropertySubName"].iloc[0]
        link = df.loc[df["PropertyName"] == name, "Link"].iloc[0]
        records.append({
            "PropertyName": name,
            "Locality": parse_locality(subname),
            "BHKConfigs": bhk_configs,
            "PropertyTypes": sorted(property_types),
            "IsLand": int(row.get("is_land", 0)),
            "MinPriceCr": min(prices) if prices else None,
            "MaxPriceCr": max(prices) if prices else None,
            "Link": link,
        })
    return pd.DataFrame(records)


def top_landmarks(location_df, min_count=8, top_n=40):
    """The most commonly-mentioned canonical locations, for a landmark
    dropdown that's actually useful (out of ~1,000 raw location strings,
    most appear once or twice - not worth showing as a filter option)."""
    counts = location_df.notna().sum().sort_values(ascending=False)
    counts = counts[counts >= min_count]
    return counts.head(top_n).index.tolist()


def build_similarity_matrices(df):
    df = df.copy()

    # --- facilities ---
    df["TopFacilities"] = df["TopFacilities"].apply(parse_facilities)
    df["FacilitiesStr"] = df["TopFacilities"].apply(lambda lst: " ".join(lst))
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df["FacilitiesStr"])
    sim_facilities = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # --- price / area / building type ---
    price_area_df = build_price_area_features(df)
    price_area_df = price_area_df.reindex(df["PropertyName"])  # guarantee row order matches df
    categorical_cols = [c for c in price_area_df.columns if c.startswith("building_type_")]
    ohe_df = pd.get_dummies(price_area_df, columns=categorical_cols, drop_first=True)
    ohe_df = ohe_df.fillna(0)
    scaler_price = StandardScaler()
    ohe_scaled = pd.DataFrame(scaler_price.fit_transform(ohe_df), columns=ohe_df.columns, index=ohe_df.index)
    sim_price = cosine_similarity(ohe_scaled)

    # --- location distances ---
    location_df = build_location_distance_matrix(df)
    location_df = location_df.reindex(df["PropertyName"])  # guarantee row order matches df
    location_df = fill_missing_distances(location_df)
    scaler_loc = StandardScaler()
    location_scaled = pd.DataFrame(scaler_loc.fit_transform(location_df), columns=location_df.columns,
                                   index=location_df.index)
    sim_location = cosine_similarity(location_scaled)

    property_names = df["PropertyName"].tolist()
    assert sim_facilities.shape[0] == sim_price.shape[0] == sim_location.shape[0] == len(property_names)

    return {
        "property_names": property_names,
        "sim_facilities": sim_facilities,
        "sim_price": sim_price,
        "sim_location": sim_location,
    }


def combined_similarity(mats, w_facility=FACILITY_WEIGHT, w_price=PRICE_WEIGHT, w_location=LOCATION_WEIGHT):
    return (w_facility * mats["sim_facilities"]
            + w_price * mats["sim_price"]
            + w_location * mats["sim_location"])


# ---------------------------------------------------------------------------
# 6. Recommend - reusable both here and (after loading the saved artifacts)
#    in the Streamlit page, with no dependency on sklearn/pandas pipeline
#    machinery at inference time.
# ---------------------------------------------------------------------------
def recommend_properties(property_name, property_names, similarity_matrix, top_n=5):
    if property_name not in property_names:
        suggestions = difflib.get_close_matches(property_name, property_names, n=3)
        hint = f" Did you mean: {suggestions}?" if suggestions else ""
        raise ValueError(f"{property_name!r} not found in the dataset.{hint}")

    idx = property_names.index(property_name)
    scores = [(i, s) for i, s in enumerate(similarity_matrix[idx]) if i != idx]
    scores.sort(key=lambda x: x[1], reverse=True)
    scores = scores[:top_n]

    return pd.DataFrame({
        "PropertyName": [property_names[i] for i, _ in scores],
        "SimilarityScore": [round(float(s), 4) for _, s in scores],
    })


# ---------------------------------------------------------------------------
# 7. Save artifacts for the Streamlit recommender page
# ---------------------------------------------------------------------------
def save_artifacts(mats, combined_sim, properties_meta, location_df, landmarks, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "recommender_artifacts.pkl")
    joblib.dump({
        "property_names": mats["property_names"],
        "combined_similarity": combined_sim,
        "sim_facilities": mats["sim_facilities"],
        "sim_price": mats["sim_price"],
        "sim_location": mats["sim_location"],
        "weights": {"facility": FACILITY_WEIGHT, "price": PRICE_WEIGHT, "location": LOCATION_WEIGHT},
        "properties_meta": properties_meta,
        "location_df": location_df,  # property x canonical-landmark distance in meters (for landmark filtering)
        "landmarks": landmarks,  # shortlist of the most commonly-mentioned landmarks
    }, path)
    print(f"Saved recommender artifacts -> {path}")
    return path


if __name__ == "__main__":
    df = load_raw()
    mats = build_similarity_matrices(df)
    combined_sim = combined_similarity(mats)
    properties_meta = build_properties_meta(df)

    location_df = build_location_distance_matrix(df)
    location_df = location_df.reindex(df["PropertyName"])
    location_df = fill_missing_distances(location_df)
    landmarks = top_landmarks(location_df)
    print(f"Landmark shortlist ({len(landmarks)} of {location_df.shape[1]} total): {landmarks[:10]}...")

    save_artifacts(mats, combined_sim, properties_meta, location_df, landmarks)

    sample_name = mats["property_names"][0]
    print(f"\nSample recommendations for {sample_name!r}:")
    print(recommend_properties(sample_name, mats["property_names"], combined_sim, top_n=5))
