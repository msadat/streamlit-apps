# runway_grade_compliance_app.py
# Streamlit app for FAA runway longitudinal and transverse grade compliance screening
#
# Run with:
#   streamlit run runway_grade_compliance_app.py
#
# Recommended packages:
#   pip install streamlit pandas numpy plotly openpyxl

import io
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
    "Screen runway surface grades using 25-ft station and offset elevation data. "
    "Criteria are based on FAA AC 150/5300-13B runway geometric design standards."
)


# ============================================================
# Helper functions
# ============================================================

def generate_station_values(runway_length_ft: float, interval_ft: float = 25.0) -> np.ndarray:
    """Generate longitudinal station array from 0 to runway length."""
    stations = np.arange(0, runway_length_ft + interval_ft, interval_ft)
    stations = stations[stations <= runway_length_ft]
    if len(stations) == 0 or stations[-1] < runway_length_ft:
        stations = np.append(stations, runway_length_ft)
    return stations


def generate_offset_values(runway_width_ft: float, interval_ft: float = 25.0) -> np.ndarray:
    """Generate transverse offset array from -width/2 to +width/2."""
    half_width = runway_width_ft / 2.0
    offsets = np.arange(-half_width, half_width + interval_ft, interval_ft)
    offsets = offsets[(offsets >= -half_width) & (offsets <= half_width)]
    if offsets[0] > -half_width:
        offsets = np.insert(offsets, 0, -half_width)
    if offsets[-1] < half_width:
        offsets = np.append(offsets, half_width)

    # Force centerline offset = 0 if not already present
    if not np.any(np.isclose(offsets, 0.0)):
        offsets = np.sort(np.append(offsets, 0.0))

    return offsets


def create_blank_elevation_grid(runway_length_ft: float, runway_width_ft: float) -> pd.DataFrame:
    """
    Create editable grid where rows are longitudinal stations and columns are transverse offsets.
    Default starter values use a simple crowned runway with mild longitudinal grade.
    """
    stations = generate_station_values(runway_length_ft)
    offsets = generate_offset_values(runway_width_ft)

    base_elev = 600.00
    longitudinal_grade = 0.002  # 0.20%
    crown_cross_slope = 0.0125  # 1.25%

    data = {"Station_ft": stations}

    for offset in offsets:
        # Crown at centerline. Edges lower than centerline.
        elevation = (
            base_elev
            + longitudinal_grade * stations
            - crown_cross_slope * np.abs(offset)
        )
        data[f"{offset:.2f}"] = np.round(elevation, 3)

    return pd.DataFrame(data)


def parse_uploaded_grid(uploaded_file) -> pd.DataFrame:
    """Read CSV or Excel file into dataframe."""
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Please upload CSV or Excel.")


def clean_elevation_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate grid.
    Required format:
      First column: Station_ft
      Remaining columns: transverse offsets in feet, such as -75, -50, -25, 0, 25, 50, 75
    """
    df = df.copy()

    if "Station_ft" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "Station_ft"})

    df["Station_ft"] = pd.to_numeric(df["Station_ft"], errors="coerce")

    offset_cols = [c for c in df.columns if c != "Station_ft"]

    new_cols = {"Station_ft": "Station_ft"}
    for col in offset_cols:
        try:
            offset_value = float(str(col).replace("ft", "").replace("'", "").strip())
            new_cols[col] = f"{offset_value:.2f}"
        except Exception:
            raise ValueError(
                f"Offset column '{col}' could not be interpreted as a numeric transverse offset in feet."
            )

    df = df.rename(columns=new_cols)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Station_ft"]).sort_values("Station_ft").reset_index(drop=True)

    return df


def get_offset_columns(df: pd.DataFrame) -> list:
    """Return offset columns sorted numerically."""
    offset_cols = [c for c in df.columns if c != "Station_ft"]
    offset_cols = sorted(offset_cols, key=lambda x: float(x))
    return offset_cols


def nearest_column(offset_cols: list, target_offset: float) -> str:
    """Find offset column nearest to a target offset."""
    offsets = np.array([float(c) for c in offset_cols])
    idx = np.argmin(np.abs(offsets - target_offset))
    return offset_cols[idx]


def compute_segment_grades(stations: np.ndarray, elevations: np.ndarray) -> pd.DataFrame:
    """Compute longitudinal segment grades between adjacent stations."""
    rows = []

    for i in range(len(stations) - 1):
        x1 = stations[i]
        x2 = stations[i + 1]
        z1 = elevations[i]
        z2 = elevations[i + 1]
        dx = x2 - x1

        if dx == 0:
            grade_pct = np.nan
        else:
            grade_pct = 100.0 * (z2 - z1) / dx

        rows.append(
            {
                "Segment": i + 1,
                "Start Station ft": x1,
                "End Station ft": x2,
                "Start Elev ft": z1,
                "End Elev ft": z2,
                "Segment Length ft": dx,
                "Longitudinal Grade %": grade_pct,
                "Abs Grade %": abs(grade_pct) if pd.notna(grade_pct) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def compute_grade_changes(grade_df: pd.DataFrame) -> pd.DataFrame:
    """Compute grade change between consecutive longitudinal grade segments."""
    rows = []
    grades = grade_df["Longitudinal Grade %"].to_numpy()
    stations = grade_df["End Station ft"].to_numpy()

    for i in range(len(grades) - 1):
        g1 = grades[i]
        g2 = grades[i + 1]
        delta_g = g2 - g1

        rows.append(
            {
                "PVI Approx Station ft": stations[i],
                "Incoming Grade %": g1,
                "Outgoing Grade %": g2,
                "Grade Change %": delta_g,
                "Abs Grade Change %": abs(delta_g),
            }
        )

    return pd.DataFrame(rows)


def compute_transverse_grades(df: pd.DataFrame, crown_offset_ft: float) -> pd.DataFrame:
    """
    Compute transverse grades from crown to each offset for each station.

    Convention:
    - Crown elevation should be higher than edge elevation for normal crowned drainage.
    - Left side: offset < crown offset
    - Right side: offset > crown offset
    - Report positive drainage slope away from crown.
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

            z = row[col]
            distance = abs(offset - crown_actual)

            if distance == 0:
                continue

            # Positive drainage slope means elevation drops away from crown.
            drainage_slope_pct = 100.0 * (z_crown - z) / distance

            side = "Left of Crown" if offset < crown_actual else "Right of Crown"

            rows.append(
                {
                    "Station ft": station,
                    "Offset ft": offset,
                    "Side": side,
                    "Crown Offset Used ft": crown_actual,
                    "Crown Elev ft": z_crown,
                    "Point Elev ft": z,
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
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply FAA longitudinal grade compliance checks."""
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

    grade_df = grade_df.copy()
    grade_change_df = grade_change_df.copy()

    grade_df["Max Allowed Grade %"] = max_long_grade_pct
    grade_df["Longitudinal Grade Compliance"] = np.where(
        grade_df["Abs Grade %"] <= max_long_grade_pct,
        "PASS",
        "FAIL",
    )

    if end_zone_limit_pct is not None:
        start_end_zone = end_zone_length_ft
        far_end_zone_start = runway_length_ft - end_zone_length_ft

        # Segment is considered in end zone if its midpoint falls inside either controlled zone.
        grade_df["Segment Midpoint ft"] = (
            grade_df["Start Station ft"] + grade_df["End Station ft"]
        ) / 2.0

        grade_df["Inside C/D/E End Zone"] = (
            (grade_df["Segment Midpoint ft"] <= start_end_zone)
            | (grade_df["Segment Midpoint ft"] >= far_end_zone_start)
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
        grade_df["Segment Midpoint ft"] = (
            grade_df["Start Station ft"] + grade_df["End Station ft"]
        ) / 2.0
        grade_df["Inside C/D/E End Zone"] = False
        grade_df["End Zone Max Allowed Grade %"] = np.nan
        grade_df["C/D/E End Zone Compliance"] = "N/A"

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
    """Apply transverse slope compliance checks."""
    trans_df = trans_df.copy()

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
    """Summarize PASS/FAIL counts."""
    return {
        "Check": label,
        "PASS Count": int((status_series == "PASS").sum()),
        "FAIL Count": int((status_series == "FAIL").sum()),
        "N/A Count": int((status_series == "N/A").sum()),
        "Overall": "PASS" if not (status_series == "FAIL").any() else "FAIL",
    }


def style_pass_fail(df: pd.DataFrame):
    """Apply simple Streamlit dataframe styling."""
    def color_status(val):
        if val == "PASS":
            return "background-color: #d8f3dc; color: #1b4332;"
        if val == "FAIL":
            return "background-color: #ffd6d6; color: #7f0000;"
        if val == "N/A":
            return "background-color: #eeeeee; color: #555555;"
        return ""

    return df.style.applymap(color_status)


def plot_longitudinal_profile(grade_df: pd.DataFrame, stations: np.ndarray, elevations: np.ndarray):
    """Plot centerline longitudinal profile and grade segments."""
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

    fig.add_trace(
        go.Bar(
            x=grade_df["Segment Midpoint ft"],
            y=grade_df["Longitudinal Grade %"],
            name="Segment Grade %",
            width=20,
        )
    )

    max_g = criteria["max_long_grade_pct"]
    fig.add_hline(y=max_g, line_dash="dash", annotation_text=f"+{max_g:.2f}% limit")
    fig.add_hline(y=-max_g, line_dash="dash", annotation_text=f"-{max_g:.2f}% limit")

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


def plot_transverse_profiles(df: pd.DataFrame, selected_stations: list[float]):
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
    help="Use 0 ft for normal centerline crown. Use another value for off-center crown.",
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

st.sidebar.header("Data Input")

input_method = st.sidebar.radio(
    "Elevation Input Method",
    options=[
        "Edit generated 25-ft grid",
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
            "Example columns: Station_ft, -75, -50, -25, 0, 25, 50, 75"
        ),
    )


# ============================================================
# Data input section
# ============================================================

st.subheader("1. Elevation Input Grid")

st.markdown(
    """
Enter elevations in **feet**. Rows are longitudinal stations. Columns are transverse offsets.
The default grid is generated at approximately **25-ft intervals** in both directions.
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
    initial_df = create_blank_elevation_grid(runway_length_ft, runway_width_ft)

edited_df = st.data_editor(
    initial_df,
    use_container_width=True,
    num_rows="dynamic",
    height=400,
)

try:
    runway_df = clean_elevation_grid(edited_df)
except Exception as e:
    st.error(f"Input grid error: {e}")
    st.stop()

offset_cols = get_offset_columns(runway_df)

if len(runway_df) < 2:
    st.error("At least two longitudinal stations are required.")
    st.stop()

if len(offset_cols) < 2:
    st.error("At least two transverse offset columns are required.")
    st.stop()

centerline_col = nearest_column(offset_cols, 0.0)
crown_col = nearest_column(offset_cols, crown_offset_ft)

st.info(
    f"Centerline elevation column used for longitudinal profile: offset **{float(centerline_col):.2f} ft**. "
    f"Crown column used for transverse drainage checks: offset **{float(crown_col):.2f} ft**."
)

template_csv = convert_df_to_csv(create_blank_elevation_grid(runway_length_ft, runway_width_ft))
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
        "Longitudinal grade change between adjacent 25-ft segments",
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

col1, col2, col3, col4 = st.columns(4)

overall_status = "PASS" if not (summary_df["Overall"] == "FAIL").any() else "FAIL"

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

st.dataframe(style_pass_fail(summary_df), use_container_width=True)


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

    default_station_choices = [
        available_stations[0],
        available_stations[len(available_stations) // 4],
        available_stations[len(available_stations) // 2],
        available_stations[3 * len(available_stations) // 4],
        available_stations[-1],
    ]

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
# Detailed tables
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
        "This is a screening check based on adjacent 25-ft segment grade differences. "
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
# Professional reference / formula section
# ============================================================

st.subheader("5. FAA References, Equations, and Design Notes")

with st.expander("Show FAA references and formulas", expanded=True):

    st.markdown(
        """
### Primary FAA Reference

**FAA AC 150/5300-13B, Airport Design**  
Use the current FAA-published edition and any applicable changes/errata for final design decisions.

This app screens the following:

1. Centerline longitudinal grade.
2. Longitudinal grade changes between adjacent sampled segments.
3. C/D/E end-zone grade limits.
4. Transverse drainage slopes from the runway crown.
5. Centerline and transverse profile visualization.

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
- \(\Delta G_{\max}\) = FAA maximum allowable grade change, percent  

For discrete 25-ft survey data, this app approximates grade changes between adjacent sampled segments.
For final design, verify true PVI locations, vertical curves, and FAA vertical-curve length requirements.
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
For **Aircraft Approach Category A or B**, this app uses a maximum runway longitudinal grade of
\(\pm 2.0\%\) and a maximum grade change of \(\pm 2.0\%\).
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
- Maximum allowable grade change: \(\pm 1.5\%\)
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

Positive \(X_j\) indicates the surface drains away from the crown.
"""
    )

    st.markdown("### FAA Transverse Grade Screening Criteria Used")

    st.latex(
        r"""
        1.0\% \leq X_j \leq 1.5\%
        """
    )

    st.markdown(
        """
This app checks runway pavement transverse drainage slope from the crown against the selected range.
The default range is \(1.0\%\) to \(1.5\%\), consistent with FAA AC 150/5300-13B runway transverse slope guidance.
"""
    )

    st.markdown("### Important Engineering Notes")

    st.markdown(
        """
- This tool is intended for **screening and design review**, not final FAA certification.
- The app assumes the uploaded elevations represent the runway pavement surface.
- The centerline profile is taken from the transverse offset closest to 0 ft.
- The crown elevation is taken from the transverse offset closest to the user-selected crown offset.
- For off-center crown sections, confirm that the crown location and drainage intent match the actual design.
- For final compliance, verify vertical curve geometry, line-of-sight, runway safety area grading, runway shoulder grading, pavement smoothness, and project-specific FAA coordination requirements.
"""
    )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Developed as an engineering screening tool for runway grade compliance review. "
    "Always verify against the current FAA AC 150/5300-13B and project-specific FAA approvals."
)
