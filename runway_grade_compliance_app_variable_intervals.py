# runway_grade_compliance_app.py
# Streamlit app for FAA runway longitudinal and transverse grade compliance screening
#
# Run with:
#   streamlit run runway_grade_compliance_app.py
#
# Recommended packages:
#   pip install streamlit pandas numpy plotly openpyxl

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# App configuration
# ============================================================

st.set_page_config(
    page_title="FAA Runway Grade Compliance Checker",
    page_icon="🛫",
    layout="wide",
)

st.title("FAA Runway Longitudinal and Transverse Grade Compliance Checker")

st.caption(
    "Screen runway surface grades using user-defined station spacing and transverse elevation columns. "
    "Criteria are based on FAA AC 150/5300-13B runway geometric design standards."
)


# ============================================================
# Helper functions
# ============================================================

def generate_station_values(runway_length_ft: float, interval_ft: float = 25.0) -> np.ndarray:
    """Generate longitudinal station array from 0 to runway length using user-selected spacing."""
    if interval_ft <= 0:
        raise ValueError("Longitudinal station interval must be greater than zero.")

    stations = np.arange(0, runway_length_ft + interval_ft, interval_ft)
    stations = stations[stations <= runway_length_ft]

    if len(stations) == 0 or stations[-1] < runway_length_ft:
        stations = np.append(stations, runway_length_ft)

    return np.round(stations, 3)


def default_four_column_distances(runway_width_ft: float) -> list:
    """Return four default distances on each side of centerline, ending at the pavement edge."""
    half_width = runway_width_ft / 2.0
    return [round(half_width * i / 4.0, 3) for i in range(1, 5)]


def parse_offset_distances(offset_text: str, side_label: str, half_width_ft: float) -> list:
    """Parse exactly four comma-separated transverse distances measured from centerline."""
    if offset_text is None or str(offset_text).strip() == "":
        raise ValueError(f"Enter four {side_label} offset distances from centerline.")

    cleaned_text = str(offset_text).replace(";", ",")
    parts = [p.strip() for p in cleaned_text.split(",") if p.strip() != ""]

    if len(parts) != 4:
        raise ValueError(
            f"The {side_label} side must have exactly four distances from centerline. "
            f"Example: 18.75, 37.5, 56.25, 75"
        )

    values = []
    for part in parts:
        try:
            value = float(part.replace("ft", "").replace("'", "").strip())
        except Exception:
            raise ValueError(f"The {side_label} offset value '{part}' is not numeric.")

        if value <= 0:
            raise ValueError(f"The {side_label} offset distances must be positive numbers.")

        if value > half_width_ft + 1e-9:
            raise ValueError(
                f"The {side_label} offset distance {value:g} ft exceeds the half-width "
                f"of the runway ({half_width_ft:g} ft)."
            )

        values.append(value)

    rounded_check = [round(v, 3) for v in values]
    if len(set(rounded_check)) != 4:
        raise ValueError(f"The {side_label} offset distances must be unique.")

    return sorted(values)


def generate_offset_values(
    runway_width_ft: float,
    left_distances_ft=None,
    right_distances_ft=None,
) -> np.ndarray:
    """Generate transverse offsets with four columns left, centerline, and four columns right."""
    if left_distances_ft is None:
        left_distances_ft = default_four_column_distances(runway_width_ft)

    if right_distances_ft is None:
        right_distances_ft = default_four_column_distances(runway_width_ft)

    if len(left_distances_ft) != 4 or len(right_distances_ft) != 4:
        raise ValueError("Exactly four offset distances are required on each side of centerline.")

    left_offsets = [-d for d in sorted(left_distances_ft, reverse=True)]
    right_offsets = sorted(right_distances_ft)
    offsets = np.array(left_offsets + [0.0] + right_offsets, dtype=float)

    rounded_names = [f"{o:.2f}" for o in offsets]
    if len(set(rounded_names)) != len(rounded_names):
        raise ValueError(
            "Offset columns must be unique after rounding to 0.01 ft. "
            "Increase spacing between adjacent offsets."
        )

    return offsets


def create_blank_elevation_grid(
    runway_length_ft: float,
    runway_width_ft: float,
    station_interval_ft: float = 25.0,
    left_distances_ft=None,
    right_distances_ft=None,
) -> pd.DataFrame:
    """
    Create editable grid where rows are longitudinal stations and columns are transverse offsets.
    The default template has four columns left of centerline, one centerline column, and four columns right.
    """
    stations = generate_station_values(runway_length_ft, station_interval_ft)
    offsets = generate_offset_values(runway_width_ft, left_distances_ft, right_distances_ft)

    base_elev = 600.00
    longitudinal_grade_decimal = 0.0020   # 0.20%
    crown_cross_slope_decimal = 0.0125    # 1.25%

    data = {"Station_ft": stations}

    for offset in offsets:
        elevation = (
            base_elev
            + longitudinal_grade_decimal * stations
            - crown_cross_slope_decimal * np.abs(offset)
        )
        data[f"{offset:.2f}"] = np.round(elevation, 3)

    return pd.DataFrame(data)


def parse_uploaded_grid(uploaded_file) -> pd.DataFrame:
    """Read uploaded CSV or Excel file."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file type. Please upload a CSV, XLSX, or XLS file.")


def clean_elevation_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the elevation grid.

    Required format:
      First column: Station_ft
      Remaining columns: transverse offsets in feet.

    Example:
      Station_ft, -75, -56.25, -37.5, -18.75, 0, 18.75, 37.5, 56.25, 75
    """
    df = df.copy()

    if df.empty:
        raise ValueError("The elevation grid is empty.")

    if "Station_ft" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "Station_ft"})

    df["Station_ft"] = pd.to_numeric(df["Station_ft"], errors="coerce")

    offset_cols = [c for c in df.columns if c != "Station_ft"]

    if len(offset_cols) < 2:
        raise ValueError("At least two transverse offset columns are required.")

    new_cols = {"Station_ft": "Station_ft"}

    for col in offset_cols:
        try:
            cleaned = str(col).replace("ft", "").replace("'", "").strip()
            offset_value = float(cleaned)
            new_cols[col] = f"{offset_value:.2f}"
        except Exception:
            raise ValueError(
                f"Offset column '{col}' could not be interpreted as a numeric transverse offset in feet."
            )

    df = df.rename(columns=new_cols)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Station_ft"])
    df = df.sort_values("Station_ft").reset_index(drop=True)

    if len(df) < 2:
        raise ValueError("At least two longitudinal stations are required.")

    return df


def get_offset_columns(df: pd.DataFrame) -> list:
    """Return transverse offset columns sorted numerically."""
    offset_cols = [c for c in df.columns if c != "Station_ft"]
    return sorted(offset_cols, key=lambda x: float(x))


def nearest_column(offset_cols: list, target_offset: float) -> str:
    """Find offset column nearest to target offset."""
    offsets = np.array([float(c) for c in offset_cols])
    idx = int(np.argmin(np.abs(offsets - target_offset)))
    return offset_cols[idx]


def compute_segment_grades(stations: np.ndarray, elevations: np.ndarray) -> pd.DataFrame:
    """Compute longitudinal grades between adjacent stations."""
    rows = []

    for i in range(len(stations) - 1):
        s1 = stations[i]
        s2 = stations[i + 1]
        e1 = elevations[i]
        e2 = elevations[i + 1]
        ds = s2 - s1

        if ds == 0:
            grade_pct = np.nan
        else:
            grade_pct = 100.0 * (e2 - e1) / ds

        rows.append(
            {
                "Segment": i + 1,
                "Start Station ft": s1,
                "End Station ft": s2,
                "Segment Midpoint ft": (s1 + s2) / 2.0,
                "Start Elev ft": e1,
                "End Elev ft": e2,
                "Segment Length ft": ds,
                "Longitudinal Grade %": grade_pct,
                "Abs Grade %": abs(grade_pct) if pd.notna(grade_pct) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def compute_grade_changes(grade_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute change in longitudinal grade between adjacent sampled segments.

    Note:
    This is a screening check from adjacent sampled station segments.
    Final design should use actual PVI locations and vertical curve lengths.
    """
    rows = []

    if len(grade_df) < 2:
        return pd.DataFrame(
            columns=[
                "PVI Approx Station ft",
                "Incoming Grade %",
                "Outgoing Grade %",
                "Grade Change %",
                "Abs Grade Change %",
            ]
        )

    grades = grade_df["Longitudinal Grade %"].to_numpy()
    pvi_stations = grade_df["End Station ft"].to_numpy()

    for i in range(len(grades) - 1):
        g1 = grades[i]
        g2 = grades[i + 1]
        delta_g = g2 - g1

        rows.append(
            {
                "PVI Approx Station ft": pvi_stations[i],
                "Incoming Grade %": g1,
                "Outgoing Grade %": g2,
                "Grade Change %": delta_g,
                "Abs Grade Change %": abs(delta_g),
            }
        )

    return pd.DataFrame(rows)


def compute_transverse_grades(df: pd.DataFrame, crown_offset_ft: float):
    """
    Compute transverse drainage slope from crown to each offset.

    Positive value means elevation drops away from the crown.
    """
    offset_cols = get_offset_columns(df)
    crown_col = nearest_column(offset_cols, crown_offset_ft)
    crown_actual = float(crown_col)

    rows = []

    for _, row in df.iterrows():
        station = row["Station_ft"]
        z_crown = row[crown_col]

        for col in offset_cols:
            offset = float(col)

            if np.isclose(offset, crown_actual):
                continue

            z_point = row[col]
            distance = abs(offset - crown_actual)

            if distance == 0:
                continue

            drainage_slope_pct = 100.0 * (z_crown - z_point) / distance
            side = "Left of Crown" if offset < crown_actual else "Right of Crown"

            rows.append(
                {
                    "Station ft": station,
                    "Offset ft": offset,
                    "Side": side,
                    "Crown Offset Used ft": crown_actual,
                    "Crown Elev ft": z_crown,
                    "Point Elev ft": z_point,
                    "Distance From Crown ft": distance,
                    "Transverse Drainage Slope %": drainage_slope_pct,
                    "Abs Transverse Slope %": abs(drainage_slope_pct),
                }
            )

    return pd.DataFrame(rows), crown_col, crown_actual


def classify_longitudinal_compliance(
    grade_df: pd.DataFrame,
    grade_change_df: pd.DataFrame,
    runway_length_ft: float,
    aircraft_approach_category: str,
):
    """
    Apply FAA runway longitudinal grade screening criteria.

    Criteria included in this app:
    - AAC A/B: max longitudinal grade = 2.0%
    - AAC C/D/E: max longitudinal grade = 1.5%
    - AAC C/D/E runway ends: max grade = 0.8% in first/last quarter or 2,500 ft, whichever is less
    - Grade change screening:
        A/B = 2.0%
        C/D/E = 1.5%
    """
    grade_df = grade_df.copy()
    grade_change_df = grade_change_df.copy()

    if aircraft_approach_category in ["A", "B"]:
        max_long_grade_pct = 2.0
        max_grade_change_pct = 2.0
        end_zone_limit_pct = None
        end_zone_length_ft = None
    else:
        max_long_grade_pct = 1.5
        max_grade_change_pct = 1.5
        end_zone_limit_pct = 0.8
        end_zone_length_ft = min(runway_length_ft / 4.0, 2500.0)

    grade_df["Max Allowed Grade %"] = max_long_grade_pct

    grade_df["Longitudinal Grade Compliance"] = np.where(
        grade_df["Abs Grade %"] <= max_long_grade_pct,
        "PASS",
        "FAIL",
    )

    if end_zone_limit_pct is not None:
        first_end_zone_limit = end_zone_length_ft
        last_end_zone_start = runway_length_ft - end_zone_length_ft

        grade_df["Inside C/D/E End Zone"] = (
            (grade_df["Segment Midpoint ft"] <= first_end_zone_limit)
            | (grade_df["Segment Midpoint ft"] >= last_end_zone_start)
        )

        grade_df["End Zone Max Allowed Grade %"] = np.where(
            grade_df["Inside C/D/E End Zone"],
            end_zone_limit_pct,
            np.nan,
        )

        grade_df["C/D/E End Zone Compliance"] = np.where(
            grade_df["Inside C/D/E End Zone"],
            np.where(grade_df["Abs Grade %"] <= end_zone_limit_pct, "PASS", "FAIL"),
            "N/A",
        )
    else:
        grade_df["Inside C/D/E End Zone"] = False
        grade_df["End Zone Max Allowed Grade %"] = np.nan
        grade_df["C/D/E End Zone Compliance"] = "N/A"

    if grade_change_df.empty:
        grade_change_df["Max Allowed Grade Change %"] = []
        grade_change_df["Grade Change Compliance"] = []
    else:
        grade_change_df["Max Allowed Grade Change %"] = max_grade_change_pct
        grade_change_df["Grade Change Compliance"] = np.where(
            grade_change_df["Abs Grade Change %"] <= max_grade_change_pct,
            "PASS",
            "FAIL",
        )

    criteria = {
        "max_long_grade_pct": max_long_grade_pct,
        "max_grade_change_pct": max_grade_change_pct,
        "end_zone_limit_pct": end_zone_limit_pct,
        "end_zone_length_ft": end_zone_length_ft,
    }

    return grade_df, grade_change_df, criteria


def classify_transverse_compliance(
    trans_df: pd.DataFrame,
    min_transverse_pct: float,
    max_transverse_pct: float,
) -> pd.DataFrame:
    """Apply transverse slope screening criteria."""
    trans_df = trans_df.copy()

    if trans_df.empty:
        trans_df["Min Allowed Transverse Slope %"] = []
        trans_df["Max Allowed Transverse Slope %"] = []
        trans_df["Transverse Slope Compliance"] = []
        return trans_df

    trans_df["Min Allowed Transverse Slope %"] = min_transverse_pct
    trans_df["Max Allowed Transverse Slope %"] = max_transverse_pct

    trans_df["Transverse Slope Compliance"] = np.where(
        (trans_df["Transverse Drainage Slope %"] >= min_transverse_pct)
        & (trans_df["Transverse Drainage Slope %"] <= max_transverse_pct),
        "PASS",
        "FAIL",
    )

    return trans_df


def summarize_pass_fail(label: str, status_series: pd.Series) -> dict:
    """Summarize PASS/FAIL/N/A counts."""
    if status_series.empty:
        return {
            "Check": label,
            "PASS Count": 0,
            "FAIL Count": 0,
            "N/A Count": 0,
            "Overall": "N/A",
        }

    pass_count = int((status_series == "PASS").sum())
    fail_count = int((status_series == "FAIL").sum())
    na_count = int((status_series == "N/A").sum())

    overall = "FAIL" if fail_count > 0 else "PASS"

    return {
        "Check": label,
        "PASS Count": pass_count,
        "FAIL Count": fail_count,
        "N/A Count": na_count,
        "Overall": overall,
    }


def style_pass_fail(df: pd.DataFrame):
    """
    Apply PASS/FAIL/N/A styling.

    This version works with newer pandas versions where Styler.applymap()
    has been replaced by Styler.map().
    """
    def color_status(val):
        if val == "PASS":
            return "background-color: #d8f3dc; color: #1b4332;"
        if val == "FAIL":
            return "background-color: #ffd6d6; color: #7f0000;"
        if val == "N/A":
            return "background-color: #eeeeee; color: #555555;"
        return ""

    styler = df.style

    # pandas >= 2.1 uses Styler.map()
    if hasattr(styler, "map"):
        return styler.map(color_status)

    # older pandas fallback
    return styler.applymap(color_status)


def plot_longitudinal_profile(
    grade_df: pd.DataFrame,
    stations: np.ndarray,
    elevations: np.ndarray,
):
    """Plot centerline longitudinal profile."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=stations,
            y=elevations,
            mode="lines+markers",
            name="Centerline Elevation",
            line=dict(width=3),
        )
    )

    fail_segments = grade_df[grade_df["Longitudinal Grade Compliance"] == "FAIL"]

    for _, row in fail_segments.iterrows():
        fig.add_vrect(
            x0=row["Start Station ft"],
            x1=row["End Station ft"],
            fillcolor="red",
            opacity=0.15,
            line_width=0,
        )

    fig.update_layout(
        title="Runway Centerline Longitudinal Profile",
        xaxis_title="Station, ft",
        yaxis_title="Elevation, ft",
        hovermode="x unified",
        height=500,
    )

    return fig


def plot_centerline_grade(grade_df: pd.DataFrame, criteria: dict):
    """Plot centerline segment longitudinal grades."""
    fig = go.Figure()

    if "Segment Length ft" in grade_df.columns and not grade_df.empty:
        bar_width = np.maximum(grade_df["Segment Length ft"].to_numpy(dtype=float) * 0.80, 1.0)
    else:
        bar_width = None

    fig.add_trace(
        go.Bar(
            x=grade_df["Segment Midpoint ft"],
            y=grade_df["Longitudinal Grade %"],
            name="Segment Grade %",
            width=bar_width,
        )
    )

    max_g = criteria["max_long_grade_pct"]

    fig.add_hline(
        y=max_g,
        line_dash="dash",
        annotation_text=f"+{max_g:.2f}% longitudinal limit",
    )

    fig.add_hline(
        y=-max_g,
        line_dash="dash",
        annotation_text=f"-{max_g:.2f}% longitudinal limit",
    )

    if criteria["end_zone_limit_pct"] is not None:
        end_g = criteria["end_zone_limit_pct"]

        fig.add_hline(
            y=end_g,
            line_dash="dot",
            annotation_text=f"+{end_g:.2f}% C/D/E end-zone limit",
        )

        fig.add_hline(
            y=-end_g,
            line_dash="dot",
            annotation_text=f"-{end_g:.2f}% C/D/E end-zone limit",
        )

    fig.update_layout(
        title="Centerline Longitudinal Segment Grades",
        xaxis_title="Segment Midpoint Station, ft",
        yaxis_title="Longitudinal Grade, %",
        height=450,
    )

    return fig


def plot_transverse_profiles(df: pd.DataFrame, selected_stations: list):
    """Plot transverse elevation profiles at selected stations."""
    offset_cols = get_offset_columns(df)
    offsets = np.array([float(c) for c in offset_cols])

    fig = go.Figure()

    for station in selected_stations:
        idx = (df["Station_ft"] - station).abs().idxmin()
        actual_station = df.loc[idx, "Station_ft"]
        elevations = df.loc[idx, offset_cols].to_numpy(dtype=float)

        fig.add_trace(
            go.Scatter(
                x=offsets,
                y=elevations,
                mode="lines+markers",
                name=f"Sta. {actual_station:.0f} ft",
            )
        )

    fig.update_layout(
        title="Transverse Runway Profiles",
        xaxis_title="Offset from Centerline/Crown Reference, ft",
        yaxis_title="Elevation, ft",
        height=500,
    )

    return fig


def plot_surface_3d(df: pd.DataFrame):
    """Plot runway elevation surface."""
    offset_cols = get_offset_columns(df)

    x = df["Station_ft"].to_numpy(dtype=float)
    y = np.array([float(c) for c in offset_cols])
    z = df[offset_cols].to_numpy(dtype=float).T

    fig = go.Figure(
        data=[
            go.Surface(
                x=x,
                y=y,
                z=z,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Elevation ft"),
            )
        ]
    )

    fig.update_layout(
        title="3D Runway Surface Elevation Model",
        scene=dict(
            xaxis_title="Station, ft",
            yaxis_title="Offset, ft",
            zaxis_title="Elevation, ft",
        ),
        height=650,
    )

    return fig


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Convert dataframe to CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


# ============================================================
# Sidebar inputs
# ============================================================

st.sidebar.header("Runway Geometry")

runway_length_ft = st.sidebar.number_input(
    "Runway Length, ft",
    min_value=500.0,
    max_value=25000.0,
    value=8000.0,
    step=100.0,
)

runway_width_ft = st.sidebar.number_input(
    "Runway Width, ft",
    min_value=25.0,
    max_value=300.0,
    value=150.0,
    step=25.0,
)

station_interval_ft = st.sidebar.number_input(
    "Longitudinal station interval, ft",
    min_value=1.0,
    max_value=1000.0,
    value=25.0,
    step=5.0,
    help="Used only when generating the editable template. Uploaded files use their own station rows.",
)

st.sidebar.header("Transverse Elevation Columns")

half_width_ft = runway_width_ft / 2.0
default_offset_distances = default_four_column_distances(runway_width_ft)
default_offset_text = ", ".join(f"{value:g}" for value in default_offset_distances)

offset_layout_method = st.sidebar.radio(
    "Offset layout for generated template",
    options=[
        "Auto-fit 4 columns per side",
        "Enter custom 4-column distances",
    ],
    help=(
        "The generated grid always includes four elevation columns left of centerline, "
        "one centerline column, and four elevation columns right of centerline."
    ),
)

if offset_layout_method == "Auto-fit 4 columns per side":
    left_distances_ft = default_offset_distances
    right_distances_ft = default_offset_distances
else:
    left_offset_text = st.sidebar.text_input(
        "Left-side distances from CL, ft",
        value=default_offset_text,
        help="Enter exactly four positive distances from centerline. Example: 18.75, 37.5, 56.25, 75",
    )

    right_offset_text = st.sidebar.text_input(
        "Right-side distances from CL, ft",
        value=default_offset_text,
        help="Enter exactly four positive distances from centerline. Example: 18.75, 37.5, 56.25, 75",
    )

    try:
        left_distances_ft = parse_offset_distances(left_offset_text, "left", half_width_ft)
        right_distances_ft = parse_offset_distances(right_offset_text, "right", half_width_ft)
    except Exception as e:
        st.sidebar.error(str(e))
        st.stop()

generated_offsets = generate_offset_values(
    runway_width_ft=runway_width_ft,
    left_distances_ft=left_distances_ft,
    right_distances_ft=right_distances_ft,
)

aircraft_approach_category = st.sidebar.selectbox(
    "Aircraft Approach Category",
    options=["A", "B", "C", "D", "E"],
    index=3,
    help="Select the controlling Aircraft Approach Category for FAA longitudinal grade criteria.",
)

crown_offset_ft = st.sidebar.number_input(
    "Runway Crown Offset, ft",
    min_value=-runway_width_ft / 2.0,
    max_value=runway_width_ft / 2.0,
    value=0.0,
    step=1.0,
    help="Use 0 ft for normal centerline crown. Use another value for an off-center crown.",
)

st.sidebar.header("Transverse Slope Criteria")

min_transverse_pct = st.sidebar.number_input(
    "Minimum transverse drainage slope, %",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1,
)

max_transverse_pct = st.sidebar.number_input(
    "Maximum transverse drainage slope, %",
    min_value=0.0,
    max_value=10.0,
    value=1.5,
    step=0.1,
)

if max_transverse_pct < min_transverse_pct:
    st.sidebar.error("Maximum transverse slope must be greater than or equal to minimum transverse slope.")
    st.stop()

st.sidebar.header("Data Input")

input_method = st.sidebar.radio(
    "Elevation Input Method",
    options=[
        "Edit generated variable-interval grid",
        "Upload CSV or Excel grid",
    ],
)

uploaded_file = None

if input_method == "Upload CSV or Excel grid":
    uploaded_file = st.sidebar.file_uploader(
        "Upload runway elevation grid",
        type=["csv", "xlsx", "xls"],
        help=(
            "Required format: first column = Station_ft; remaining columns = transverse offsets in ft. "
            "Example columns: Station_ft, -75, -56.25, -37.5, -18.75, 0, 18.75, 37.5, 56.25, 75"
        ),
    )


# ============================================================
# Data input section
# ============================================================

st.subheader("1. Elevation Input Grid")

st.markdown(
    """
Enter elevations in **feet**.

- Rows are longitudinal stations.
- Columns are transverse offsets.
- The generated template uses your selected **longitudinal station interval**.
- The generated template provides **four elevation columns left of centerline, one centerline column, and four elevation columns right of centerline**.
- The transverse column distances can be auto-fitted to the runway width or entered manually in the sidebar.
"""
)

if input_method == "Upload CSV or Excel grid" and uploaded_file is not None:
    try:
        initial_df = parse_uploaded_grid(uploaded_file)
        initial_df = clean_elevation_grid(initial_df)
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        st.stop()
else:
    initial_df = create_blank_elevation_grid(
        runway_length_ft=runway_length_ft,
        runway_width_ft=runway_width_ft,
        station_interval_ft=station_interval_ft,
        left_distances_ft=left_distances_ft,
        right_distances_ft=right_distances_ft,
    )

grid_key = (
    f"elevation_grid_{runway_length_ft}_{runway_width_ft}_{station_interval_ft}_"
    f"{tuple(round(v, 3) for v in left_distances_ft)}_"
    f"{tuple(round(v, 3) for v in right_distances_ft)}_"
    f"{input_method}"
)

edited_df = st.data_editor(
    initial_df,
    use_container_width=True,
    num_rows="dynamic",
    height=500,
    key=grid_key,
)

try:
    runway_df = clean_elevation_grid(edited_df)
except Exception as e:
    st.error(f"Input grid error: {e}")
    st.stop()

offset_cols = get_offset_columns(runway_df)

centerline_col = nearest_column(offset_cols, 0.0)
crown_col = nearest_column(offset_cols, crown_offset_ft)

st.info(
    f"Centerline elevation column used for longitudinal profile: offset **{float(centerline_col):.2f} ft**. "
    f"Crown column used for transverse drainage checks: offset **{float(crown_col):.2f} ft**."
)

left_count = int((np.array([float(c) for c in offset_cols]) < 0).sum())
right_count = int((np.array([float(c) for c in offset_cols]) > 0).sum())
has_centerline_col = bool(np.any(np.isclose(np.array([float(c) for c in offset_cols]), 0.0)))

if input_method == "Edit generated variable-interval grid":
    st.success(
        f"Generated grid: station interval = {station_interval_ft:g} ft, "
        f"left offset columns = {left_count}, centerline column = {'yes' if has_centerline_col else 'no'}, "
        f"right offset columns = {right_count}."
    )
else:
    if left_count != 4 or right_count != 4 or not has_centerline_col:
        st.warning(
            "The uploaded grid was accepted, but it does not have exactly four offset columns on each side "
            "plus a centerline column. The app will still calculate using the uploaded columns."
        )

template_csv = convert_df_to_csv(
    create_blank_elevation_grid(
        runway_length_ft=runway_length_ft,
        runway_width_ft=runway_width_ft,
        station_interval_ft=station_interval_ft,
        left_distances_ft=left_distances_ft,
        right_distances_ft=right_distances_ft,
    )
)

st.download_button(
    label="Download blank/sample elevation template CSV",
    data=template_csv,
    file_name="runway_elevation_template.csv",
    mime="text/csv",
)


# ============================================================
# Compliance calculations
# ============================================================

stations = runway_df["Station_ft"].to_numpy(dtype=float)
centerline_elev = runway_df[centerline_col].to_numpy(dtype=float)

grade_df = compute_segment_grades(stations, centerline_elev)
grade_change_df = compute_grade_changes(grade_df)

grade_df, grade_change_df, longitudinal_criteria = classify_longitudinal_compliance(
    grade_df=grade_df,
    grade_change_df=grade_change_df,
    runway_length_ft=runway_length_ft,
    aircraft_approach_category=aircraft_approach_category,
)

trans_df, crown_col, crown_actual = compute_transverse_grades(runway_df, crown_offset_ft)

trans_df = classify_transverse_compliance(
    trans_df=trans_df,
    min_transverse_pct=min_transverse_pct,
    max_transverse_pct=max_transverse_pct,
)

summary_rows = [
    summarize_pass_fail(
        "Centerline longitudinal grade",
        grade_df["Longitudinal Grade Compliance"],
    ),
    summarize_pass_fail(
        "Longitudinal grade change between adjacent station segments",
        grade_change_df["Grade Change Compliance"],
    ),
    summarize_pass_fail(
        "C/D/E end-zone longitudinal grade",
        grade_df["C/D/E End Zone Compliance"],
    ),
    summarize_pass_fail(
        "Transverse drainage slope from crown",
        trans_df["Transverse Slope Compliance"],
    ),
]

summary_df = pd.DataFrame(summary_rows)


# ============================================================
# Results dashboard
# ============================================================

st.subheader("2. Compliance Summary")

overall_status = "FAIL" if (summary_df["Overall"] == "FAIL").any() else "PASS"

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Overall Screening Result", overall_status)

with col2:
    st.metric(
        "Max Longitudinal Grade Limit",
        f"±{longitudinal_criteria['max_long_grade_pct']:.2f}%",
    )

with col3:
    if longitudinal_criteria["end_zone_limit_pct"] is not None:
        st.metric(
            "C/D/E End-Zone Limit",
            f"±{longitudinal_criteria['end_zone_limit_pct']:.2f}%",
        )
    else:
        st.metric("C/D/E End-Zone Limit", "N/A")

with col4:
    st.metric(
        "Transverse Slope Target",
        f"{min_transverse_pct:.2f}% to {max_transverse_pct:.2f}%",
    )

st.dataframe(
    style_pass_fail(summary_df),
    use_container_width=True,
)

if overall_status == "FAIL":
    st.warning(
        "One or more screening checks failed. Review the detailed tables below. "
        "This tool is a computational screening aid and does not replace Engineer-of-Record review, "
        "survey verification, FAA coordination, or project-specific design criteria."
    )
else:
    st.success(
        "No failures were detected in the selected screening checks. "
        "Confirm applicability of criteria and vertical-curve design requirements during final design."
    )


# ============================================================
# Plots
# ============================================================

st.subheader("3. Runway Plots")

plot_tab1, plot_tab2, plot_tab3, plot_tab4 = st.tabs(
    [
        "Centerline Profile",
        "Longitudinal Grades",
        "Transverse Profiles",
        "3D Surface",
    ]
)

with plot_tab1:
    st.plotly_chart(
        plot_longitudinal_profile(grade_df, stations, centerline_elev),
        use_container_width=True,
    )

with plot_tab2:
    st.plotly_chart(
        plot_centerline_grade(grade_df, longitudinal_criteria),
        use_container_width=True,
    )

with plot_tab3:
    available_stations = runway_df["Station_ft"].to_list()

    if len(available_stations) >= 5:
        default_station_choices = [
            available_stations[0],
            available_stations[len(available_stations) // 4],
            available_stations[len(available_stations) // 2],
            available_stations[3 * len(available_stations) // 4],
            available_stations[-1],
        ]
    else:
        default_station_choices = available_stations

    selected_stations = st.multiselect(
        "Select stations for transverse profile plotting",
        options=available_stations,
        default=default_station_choices,
    )

    if selected_stations:
        st.plotly_chart(
            plot_transverse_profiles(runway_df, selected_stations),
            use_container_width=True,
        )
    else:
        st.info("Select at least one station to plot transverse profiles.")

with plot_tab4:
    st.plotly_chart(
        plot_surface_3d(runway_df),
        use_container_width=True,
    )


# ============================================================
# Detailed compliance tables
# ============================================================

st.subheader("4. Detailed Compliance Tables")

table_tab1, table_tab2, table_tab3 = st.tabs(
    [
        "Longitudinal Grade",
        "Grade Change",
        "Transverse Grade",
    ]
)

with table_tab1:
    st.markdown("### Centerline Longitudinal Grade by Segment")

    display_cols = [
        "Segment",
        "Start Station ft",
        "End Station ft",
        "Segment Midpoint ft",
        "Start Elev ft",
        "End Elev ft",
        "Segment Length ft",
        "Longitudinal Grade %",
        "Abs Grade %",
        "Max Allowed Grade %",
        "Longitudinal Grade Compliance",
        "Inside C/D/E End Zone",
        "End Zone Max Allowed Grade %",
        "C/D/E End Zone Compliance",
    ]

    st.dataframe(
        style_pass_fail(grade_df[display_cols].round(4)),
        use_container_width=True,
    )

    st.download_button(
        label="Download longitudinal grade results CSV",
        data=convert_df_to_csv(grade_df),
        file_name="longitudinal_grade_results.csv",
        mime="text/csv",
    )

with table_tab2:
    st.markdown("### Longitudinal Grade Change Between Adjacent Segments")

    st.info(
        "This is a screening check based on grade differences between adjacent station segments. "
        "Final FAA compliance should be verified using actual PVI locations and vertical curve lengths."
    )

    st.dataframe(
        style_pass_fail(grade_change_df.round(4)),
        use_container_width=True,
    )

    st.download_button(
        label="Download grade change results CSV",
        data=convert_df_to_csv(grade_change_df),
        file_name="longitudinal_grade_change_results.csv",
        mime="text/csv",
    )

with table_tab3:
    st.markdown("### Transverse Drainage Slope from Crown")

    st.dataframe(
        style_pass_fail(trans_df.round(4)),
        use_container_width=True,
    )

    st.download_button(
        label="Download transverse grade results CSV",
        data=convert_df_to_csv(trans_df),
        file_name="transverse_grade_results.csv",
        mime="text/csv",
    )


# ============================================================
# Professional reference and formula section
# ============================================================

st.subheader("5. FAA References, Equations, and Design Notes")

with st.expander("Show FAA references and formulas", expanded=True):

    st.markdown(
        """
### Primary FAA Reference

**FAA AC 150/5300-13B, Airport Design**

Use the current FAA-published edition and any applicable changes, errata, project-specific FAA comments, and sponsor standards for final design decisions.

This app screens:

1. Centerline longitudinal grade.
2. Longitudinal grade changes between adjacent sampled segments.
3. C/D/E end-zone grade limits.
4. Transverse drainage slopes from runway crown.
5. Centerline, transverse, and 3D surface profiles.

---
"""
    )

    st.markdown("### Longitudinal Grade Formula")

    st.latex(
        r"""
        G_i =
        \frac{E_{i+1} - E_i}{S_{i+1} - S_i}
        \times 100
        """
    )

    st.markdown(
        """
Where:

- \(G_i\) = longitudinal grade of segment \(i\), percent
- \(E_i\) = elevation at station \(i\), ft
- \(S_i\) = longitudinal station \(i\), ft
"""
    )

    st.markdown("### Longitudinal Grade Change Formula")

    st.latex(
        r"""
        \Delta G_i = G_{i+1} - G_i
        """
    )

    st.latex(
        r"""
        |\Delta G_i| \leq \Delta G_{\max}
        """
    )

    st.markdown(
        """
Where:

- \(\Delta G_i\) = change in longitudinal grade between adjacent segments, percent
- \(G_i\) = incoming segment grade, percent
- \(G_{i+1}\) = outgoing segment grade, percent
- \(\Delta G_{\max}\) = maximum allowable grade change, percent

For discrete survey or design-grid data, this app approximates grade changes between adjacent sampled station segments.
For final design, verify actual PVI locations, vertical curves, sight-distance criteria, and FAA vertical-curve length requirements.
"""
    )

    st.markdown("### FAA Longitudinal Grade Screening Criteria Used")

    if aircraft_approach_category in ["A", "B"]:
        st.latex(
            r"""
            |G| \leq 2.0\%
            """
        )

        st.latex(
            r"""
            |\Delta G| \leq 2.0\%
            """
        )

        st.markdown(
            """
For **Aircraft Approach Category A or B**, this app uses:

- Maximum runway longitudinal grade: \(\pm 2.0\%\)
- Maximum longitudinal grade change: \(\pm 2.0\%\)
"""
        )

    else:
        st.latex(
            r"""
            |G| \leq 1.5\%
            """
        )

        st.latex(
            r"""
            |G_{\mathrm{end}}| \leq 0.8\%
            """
        )

        st.latex(
            r"""
            L_{\mathrm{end}} =
            \min\left(\frac{L_{\mathrm{runway}}}{4},\ 2500\ \mathrm{ft}\right)
            """
        )

        st.latex(
            r"""
            |\Delta G| \leq 1.5\%
            """
        )

        st.markdown(
            """
For **Aircraft Approach Category C, D, or E**, this app uses:

- Maximum runway longitudinal grade: \(\pm 1.5\%\)
- Maximum grade in the first and last controlled end zone: \(\pm 0.8\%\)
- Controlled end-zone length: lesser of first/last quarter of runway length or first/last 2,500 ft
- Maximum allowable longitudinal grade change: \(\pm 1.5\%\)
"""
        )

    st.markdown("### Transverse Grade Formula")

    st.latex(
        r"""
        X_j =
        \frac{E_{\mathrm{crown}} - E_j}
        {|O_j - O_{\mathrm{crown}}|}
        \times 100
        """
    )

    st.markdown(
        """
Where:

- \(X_j\) = transverse drainage slope from crown to offset point \(j\), percent
- \(E_{\mathrm{crown}}\) = runway crown elevation, ft
- \(E_j\) = elevation at transverse offset \(j\), ft
- \(O_{\mathrm{crown}}\) = crown offset, ft
- \(O_j\) = transverse offset of point \(j\), ft

Positive \(X_j\) means the pavement surface drains away from the crown.
"""
    )

    st.markdown("### FAA Transverse Grade Screening Criteria Used")

    st.latex(
        r"""
        X_{\min} \leq X_j \leq X_{\max}
        """
    )

    st.latex(
        r"""
        1.0\% \leq X_j \leq 1.5\%
        """
    )

    st.markdown(
        """
The default transverse runway pavement slope range used in this app is \(1.0\%\) to \(1.5\%\).
The input boxes allow you to adjust this range if project-specific criteria require different limits.
"""
    )

    st.markdown("### Important Engineering Notes")

    st.markdown(
        """
- This tool is intended for **screening and design review**, not final FAA certification.
- The app assumes uploaded elevations represent the runway pavement surface.
- The centerline profile is taken from the transverse offset closest to 0 ft.
- The crown elevation is taken from the transverse offset closest to the user-selected crown offset.
- For off-center crown sections, confirm that the crown location and drainage intent match the actual design.
- For final compliance, verify vertical curve geometry, line-of-sight, runway safety area grading, runway shoulder grading, pavement smoothness, tie-ins, survey control, and FAA coordination requirements.
"""
    )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Developed as an engineering screening tool for runway grade compliance review. "
    "Always verify against the current FAA AC 150/5300-13B, project-specific FAA comments, and Engineer-of-Record requirements."
    " Copyright by Rafat Sadat."
)
