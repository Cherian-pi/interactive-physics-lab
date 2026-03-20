# modules/binding_energy.py

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# Page / module constants
# =========================================================
MODULE_PREFIX = "be"

AMU_TO_MEV = 931.49410242
PROTON_MASS_U = 1.007276466621
NEUTRON_MASS_U = 1.00866491595
ELECTRON_MASS_U = 0.000548579909

# Semi-empirical mass formula coefficients (MeV)
A_V = 15.75
A_S = 17.8
A_C = 0.711
A_A = 23.7
A_P = 34.0

ELEMENT_SYMBOLS = [
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]

CURATED_ISOTOPES = [
    {"label": "Hydrogen-1", "Z": 1, "A": 1},
    {"label": "Hydrogen-2 (Deuterium)", "Z": 1, "A": 2},
    {"label": "Hydrogen-3 (Tritium)", "Z": 1, "A": 3},
    {"label": "Helium-4", "Z": 2, "A": 4},
    {"label": "Lithium-7", "Z": 3, "A": 7},
    {"label": "Beryllium-9", "Z": 4, "A": 9},
    {"label": "Boron-11", "Z": 5, "A": 11},
    {"label": "Carbon-12", "Z": 6, "A": 12},
    {"label": "Carbon-14", "Z": 6, "A": 14},
    {"label": "Nitrogen-14", "Z": 7, "A": 14},
    {"label": "Oxygen-16", "Z": 8, "A": 16},
    {"label": "Neon-20", "Z": 10, "A": 20},
    {"label": "Magnesium-24", "Z": 12, "A": 24},
    {"label": "Silicon-28", "Z": 14, "A": 28},
    {"label": "Sulfur-32", "Z": 16, "A": 32},
    {"label": "Argon-40", "Z": 18, "A": 40},
    {"label": "Calcium-40", "Z": 20, "A": 40},
    {"label": "Titanium-48", "Z": 22, "A": 48},
    {"label": "Chromium-52", "Z": 24, "A": 52},
    {"label": "Iron-56", "Z": 26, "A": 56},
    {"label": "Nickel-62", "Z": 28, "A": 62},
    {"label": "Copper-63", "Z": 29, "A": 63},
    {"label": "Zinc-64", "Z": 30, "A": 64},
    {"label": "Krypton-84", "Z": 36, "A": 84},
    {"label": "Strontium-88", "Z": 38, "A": 88},
    {"label": "Silver-107", "Z": 47, "A": 107},
    {"label": "Tin-120", "Z": 50, "A": 120},
    {"label": "Iodine-127", "Z": 53, "A": 127},
    {"label": "Xenon-136", "Z": 54, "A": 136},
    {"label": "Barium-138", "Z": 56, "A": 138},
    {"label": "Gold-197", "Z": 79, "A": 197},
    {"label": "Lead-208", "Z": 82, "A": 208},
    {"label": "Bismuth-209", "Z": 83, "A": 209},
    {"label": "Thorium-232", "Z": 90, "A": 232},
    {"label": "Uranium-235", "Z": 92, "A": 235},
    {"label": "Uranium-238", "Z": 92, "A": 238},
    {"label": "Plutonium-239", "Z": 94, "A": 239},
]


# =========================================================
# Utility and physics functions
# =========================================================
def sym(z: int) -> str:
    if 0 < z < len(ELEMENT_SYMBOLS):
        return ELEMENT_SYMBOLS[z]
    return f"Z={z}"


def isotope_name(z: int, a: int) -> str:
    return f"{sym(z)}-{a}"


def pairing_term(a: int, z: int) -> float:
    n = a - z
    if a % 2 == 1:
        return 0.0
    if z % 2 == 0 and n % 2 == 0:
        return +A_P / (a ** 0.75)
    return -A_P / (a ** 0.75)


def semf_binding_energy_mev(a: int, z: int) -> float:
    if a <= 0 or z < 0 or z > a:
        return 0.0
    volume = A_V * a
    surface = A_S * (a ** (2.0 / 3.0))
    coulomb = A_C * z * (z - 1) / (a ** (1.0 / 3.0))
    asymmetry = A_A * ((a - 2 * z) ** 2) / a
    delta = pairing_term(a, z)
    be = volume - surface - coulomb - asymmetry + delta
    return max(be, 0.0)


def mass_defect_u(a: int, z: int) -> float:
    return semf_binding_energy_mev(a, z) / AMU_TO_MEV


def be_per_nucleon(a: int, z: int) -> float:
    if a <= 0:
        return 0.0
    return semf_binding_energy_mev(a, z) / a


def nuclear_mass_u(a: int, z: int) -> float:
    n = a - z
    return z * PROTON_MASS_U + n * NEUTRON_MASS_U - mass_defect_u(a, z)


def atomic_mass_u(a: int, z: int) -> float:
    return nuclear_mass_u(a, z) + z * ELECTRON_MASS_U


def approx_stable_z(a: int) -> float:
    if a <= 0:
        return 0.0
    return a / (2.0 + 0.015 * (a ** (2.0 / 3.0)))


def classify_region(a: int) -> str:
    if a < 56:
        return "light"
    if a <= 70:
        return "near_peak"
    return "heavy"


def stability_comment(a: int, z: int) -> str:
    z0 = approx_stable_z(a)
    delta = z - z0
    bea = be_per_nucleon(a, z)

    if a <= 3:
        return (
            "Very light nucleus. The semi-empirical model is only qualitative here, "
            "so treat the numbers as instructional rather than high-precision."
        )

    if abs(delta) <= 1.2:
        line_text = "close to the valley of stability"
    elif delta > 1.2:
        line_text = "proton-rich relative to the valley of stability"
    else:
        line_text = "neutron-rich relative to the valley of stability"

    if bea > 8.6:
        bind_text = "very strongly bound"
    elif bea > 8.0:
        bind_text = "strongly bound"
    elif bea > 7.0:
        bind_text = "moderately bound"
    else:
        bind_text = "weakly bound"

    return f"This isotope is {line_text} and is {bind_text} in this model."


def reaction_trend_text(a: int, z: int) -> str:
    region = classify_region(a)
    bea = be_per_nucleon(a, z)

    if region == "light":
        return (
            f"{isotope_name(z, a)} lies on the rising side of the binding-energy-per-nucleon curve "
            f"(BE/A ≈ {bea:.3f} MeV). In general, combining light nuclei can move them toward the "
            "iron/nickel region, so fusion tends to release energy."
        )
    if region == "near_peak":
        return (
            f"{isotope_name(z, a)} lies near the peak of the binding-energy-per-nucleon curve "
            f"(BE/A ≈ {bea:.3f} MeV). Nuclei in this region are among the most tightly bound, "
            "so neither fusion nor fission usually gives a large energy gain."
        )
    return (
        f"{isotope_name(z, a)} lies on the heavy side of the curve "
        f"(BE/A ≈ {bea:.3f} MeV). Splitting very heavy nuclei into medium-mass fragments can move "
        "the products toward higher binding energy per nucleon, so fission can release energy."
    )


def shell_hint(z: int, n: int) -> str:
    magic = {2, 8, 20, 28, 50, 82, 126}
    hits = []
    if z in magic:
        hits.append(f"Z={z} is magic")
    if n in magic:
        hits.append(f"N={n} is magic")

    if len(hits) == 2:
        return "This isotope is near a doubly magic configuration, which often enhances stability."
    if len(hits) == 1:
        return f"{hits[0]}, which can support extra stability."
    return "No magic-number enhancement is indicated here in this simplified view."


def compute_isotope_row(a: int, z: int) -> Dict[str, float]:
    n = a - z
    be = semf_binding_energy_mev(a, z)
    dm = mass_defect_u(a, z)
    return {
        "Isotope": isotope_name(z, a),
        "Z": z,
        "N": n,
        "A": a,
        "Binding Energy (MeV)": be,
        "BE/A (MeV)": be / a if a > 0 else 0.0,
        "Mass Defect (u)": dm,
        "Nuclear Mass (u)": nuclear_mass_u(a, z),
        "Atomic Mass (u)": atomic_mass_u(a, z),
    }


# =========================================================
# Data generators for plots
# =========================================================
@st.cache_data(show_spinner=False)
def build_be_curve(max_a: int = 250) -> pd.DataFrame:
    rows = []
    for a in range(2, max_a + 1):
        z_est = approx_stable_z(a)
        candidates = range(max(1, int(round(z_est)) - 3), min(a, int(round(z_est)) + 3) + 1)
        best_z = None
        best_bea = -1.0
        for z in candidates:
            val = be_per_nucleon(a, z)
            if val > best_bea:
                best_bea = val
                best_z = z
        rows.append(
            {
                "A": a,
                "Z": best_z,
                "N": a - best_z,
                "Isotope": isotope_name(best_z, a),
                "BE/A": best_bea,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def build_stability_map(z_max: int = 92, band_width: int = 5) -> pd.DataFrame:
    rows = []
    for z in range(1, z_max + 1):
        a_center = max(z, int(round(2.0 * z + 0.015 * (2.0 * z) ** (5.0 / 3.0))))
        for a in range(max(z, a_center - 14), a_center + 15):
            n = a - z
            if n < 0:
                continue
            bea = be_per_nucleon(a, z)
            stable_line_n = a - approx_stable_z(a)
            distance = abs(n - stable_line_n)

            rows.append(
                {
                    "Z": z,
                    "N": n,
                    "A": a,
                    "BE/A": bea,
                    "DistToValley": distance,
                    "NearValley": distance <= band_width,
                    "Isotope": isotope_name(z, a),
                }
            )
    return pd.DataFrame(rows)


def make_nucleus_points(z: int, n: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    total = z + n
    if total == 0:
        return np.array([]), np.array([]), []

    grid = int(math.ceil(math.sqrt(total)))
    spacing = 1.0
    xs = []
    ys = []
    kinds = []

    count = 0
    for r in range(grid):
        for c in range(grid):
            if count >= total:
                break
            x = (c - (grid - 1) / 2.0) * spacing
            y = ((grid - 1) / 2.0 - r) * spacing
            xs.append(x)
            ys.append(y)
            kinds.append("Proton" if count < z else "Neutron")
            count += 1
        if count >= total:
            break

    xs = np.array(xs)
    ys = np.array(ys)

    radius = np.sqrt(xs**2 + ys**2)
    max_r = np.max(radius) if len(radius) > 0 else 1.0
    if max_r > 0:
        scale = 4.0 / max_r
        xs *= scale
        ys *= scale

    return xs, ys, kinds


# =========================================================
# Plot functions
# =========================================================
def nucleus_figure(z: int, a: int) -> go.Figure:
    n = a - z
    xs, ys, kinds = make_nucleus_points(z, n)

    proton_x = [xs[i] for i, k in enumerate(kinds) if k == "Proton"]
    proton_y = [ys[i] for i, k in enumerate(kinds) if k == "Proton"]
    neutron_x = [xs[i] for i, k in enumerate(kinds) if k == "Neutron"]
    neutron_y = [ys[i] for i, k in enumerate(kinds) if k == "Neutron"]

    fig = go.Figure()

    boundary = np.linspace(0, 2 * np.pi, 300)
    fig.add_trace(
        go.Scatter(
            x=5.3 * np.cos(boundary),
            y=5.3 * np.sin(boundary),
            mode="lines",
            name="Nuclear boundary",
            hoverinfo="skip",
            line=dict(width=2),
            opacity=0.35,
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=proton_x,
            y=proton_y,
            mode="markers",
            name="Protons",
            marker=dict(size=18, line=dict(width=1)),
            text=[f"p ({sym(z)})"] * len(proton_x),
            hovertemplate="%{text}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=neutron_x,
            y=neutron_y,
            mode="markers",
            name="Neutrons",
            marker=dict(size=18, symbol="diamond", line=dict(width=1)),
            text=["n"] * len(neutron_x),
            hovertemplate="%{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Stylized nucleus view: {isotope_name(z, a)}",
        xaxis=dict(visible=False, range=[-6.5, 6.5]),
        yaxis=dict(visible=False, range=[-6.5, 6.5], scaleanchor="x", scaleratio=1),
        height=430,
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation="h", y=1.02, x=0.02),
    )
    return fig


def be_curve_figure(selected_a: int, selected_z: int) -> go.Figure:
    df = build_be_curve()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["A"],
            y=df["BE/A"],
            mode="lines",
            name="Most-bound trend",
            hovertemplate="A=%{x}<br>BE/A=%{y:.3f} MeV<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[selected_a],
            y=[be_per_nucleon(selected_a, selected_z)],
            mode="markers+text",
            name=isotope_name(selected_z, selected_a),
            text=[isotope_name(selected_z, selected_a)],
            textposition="top center",
            marker=dict(size=12),
            hovertemplate=(
                f"{isotope_name(selected_z, selected_a)}"
                "<br>A=%{x}<br>BE/A=%{y:.3f} MeV<extra></extra>"
            ),
        )
    )

    peak_idx = df["BE/A"].idxmax()
    fig.add_trace(
        go.Scatter(
            x=[df.loc[peak_idx, "A"]],
            y=[df.loc[peak_idx, "BE/A"]],
            mode="markers+text",
            name="Peak region",
            text=[df.loc[peak_idx, "Isotope"]],
            textposition="bottom center",
            marker=dict(size=11, symbol="star"),
            hovertemplate=(
                f"{df.loc[peak_idx, 'Isotope']}"
                "<br>A=%{x}<br>BE/A=%{y:.3f} MeV<extra></extra>"
            ),
        )
    )

    fig.add_vline(x=56, line_dash="dash", opacity=0.4)
    fig.add_annotation(
        x=56,
        y=float(df["BE/A"].max()) - 0.15,
        text="Iron/Nickel region",
        showarrow=False,
        yshift=10,
    )

    fig.update_layout(
        title="Binding energy per nucleon curve",
        xaxis_title="Mass number, A",
        yaxis_title="BE/A (MeV)",
        height=440,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def stability_map_figure(selected_a: int, selected_z: int) -> go.Figure:
    selected_n = selected_a - selected_z
    df = build_stability_map()

    near_df = df[df["NearValley"]].copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=near_df["N"],
            y=near_df["Z"],
            mode="markers",
            name="Approx. stable band",
            marker=dict(size=7, opacity=0.55),
            text=near_df["Isotope"],
            customdata=np.stack([near_df["A"], near_df["BE/A"]], axis=1),
            hovertemplate=(
                "%{text}<br>N=%{x}<br>Z=%{y}<br>"
                "A=%{customdata[0]}<br>BE/A=%{customdata[1]:.3f} MeV<extra></extra>"
            ),
        )
    )

    n_vals = np.arange(0, 160)
    z_line = []
    n_line = []
    for a in range(2, 261):
        z0 = approx_stable_z(a)
        n0 = a - z0
        if 0 <= n0 <= 170 and 0 <= z0 <= 100:
            n_line.append(n0)
            z_line.append(z0)

    fig.add_trace(
        go.Scatter(
            x=n_line,
            y=z_line,
            mode="lines",
            name="Valley center",
            line=dict(width=2, dash="dash"),
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[selected_n],
            y=[selected_z],
            mode="markers+text",
            name="Selected isotope",
            text=[isotope_name(selected_z, selected_a)],
            textposition="top center",
            marker=dict(size=13, symbol="x"),
            hovertemplate=(
                f"{isotope_name(selected_z, selected_a)}"
                "<br>N=%{x}<br>Z=%{y}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Approximate valley of stability map",
        xaxis_title="Neutron number, N",
        yaxis_title="Proton number, Z",
        height=470,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def comparison_figure(rows: List[Dict[str, float]]) -> go.Figure:
    labels = [r["Isotope"] for r in rows]
    values = [r["BE/A (MeV)"] for r in rows]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            hovertemplate="%{x}<br>BE/A=%{y:.3f} MeV<extra></extra>",
        )
    )
    fig.update_layout(
        title="Comparison of binding energy per nucleon",
        xaxis_title="Isotope",
        yaxis_title="BE/A (MeV)",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# =========================================================
# Session state helpers
# =========================================================
def state_key(name: str) -> str:
    return f"{MODULE_PREFIX}_{name}"


def init_state() -> None:
    defaults = {
        "mode": "Preset isotope",
        "preset_index": 19,  # Iron-56 default
        "manual_z": 26,
        "manual_a": 56,
        "compare_enabled": False,
        "compare_1": "Helium-4",
        "compare_2": "Iron-56",
        "compare_3": "Uranium-235",
    }
    for k, v in defaults.items():
        key = state_key(k)
        if key not in st.session_state:
            st.session_state[key] = v


def preset_lookup() -> Dict[str, Dict[str, int]]:
    return {item["label"]: item for item in CURATED_ISOTOPES}


def current_selection() -> Tuple[int, int]:
    mode = st.session_state[state_key("mode")]

    if mode == "Preset isotope":
        idx = st.session_state[state_key("preset_index")]
        item = CURATED_ISOTOPES[idx]
        return item["Z"], item["A"]

    z = int(st.session_state[state_key("manual_z")])
    a = int(st.session_state[state_key("manual_a")])
    return z, a


# =========================================================
# UI sections
# =========================================================
def header_section() -> None:
    st.title("Nuclear Binding Energy Explorer")
    st.caption(
        "Explore mass defect, total binding energy, binding energy per nucleon, "
        "the valley of stability, and fusion/fission trends."
    )


def controls_section() -> Tuple[int, int]:
    with st.sidebar:
        st.header("Controls")

        st.radio(
            "Selection mode",
            options=["Preset isotope", "Manual isotope"],
            key=state_key("mode"),
        )

        if st.session_state[state_key("mode")] == "Preset isotope":
            labels = [item["label"] for item in CURATED_ISOTOPES]
            current_idx = int(st.session_state[state_key("preset_index")])

            selected_label = st.selectbox(
                "Choose isotope",
                options=labels,
                index=current_idx,
                key=state_key("preset_selectbox"),
            )
            st.session_state[state_key("preset_index")] = labels.index(selected_label)

        else:
            current_z = int(st.session_state.get(state_key("manual_z"), 26))
            current_a = int(st.session_state.get(state_key("manual_a"), 56))

            z = st.slider(
                "Proton number Z",
                min_value=1,
                max_value=100,
                value=current_z,
                key=state_key("manual_z"),
            )

            a_default = max(current_a, z)

            a = st.slider(
                "Mass number A",
                min_value=z,
                max_value=260,
                value=a_default,
                key=state_key("manual_a"),
            )

        st.markdown("---")
        st.subheader("Comparison")
        st.toggle("Enable comparison mode", key=state_key("compare_enabled"))

        if st.session_state[state_key("compare_enabled")]:
            labels = [item["label"] for item in CURATED_ISOTOPES]
            st.selectbox("Compare isotope 1", labels, key=state_key("compare_1"))
            st.selectbox("Compare isotope 2", labels, key=state_key("compare_2"))
            st.selectbox("Compare isotope 3", labels, key=state_key("compare_3"))

        st.markdown("---")
        st.info(
            "This module uses the semi-empirical mass formula, "
            "and trend analysis rather than exact nuclear data tables."
        )

    return current_selection()


def metrics_section(z: int, a: int) -> None:
    n = a - z
    be = semf_binding_energy_mev(a, z)
    bea = be_per_nucleon(a, z)
    dm = mass_defect_u(a, z)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Isotope", isotope_name(z, a))
    c2.metric("Z", f"{z}")
    c3.metric("N", f"{n}")
    c4.metric("A", f"{a}")
    c5.metric("Binding energy", f"{be:.2f} MeV")
    c6.metric("BE/A", f"{bea:.3f} MeV")

    d1, d2, d3 = st.columns(3)
    d1.metric("Mass defect", f"{dm:.6f} u")
    d2.metric("Nuclear mass", f"{nuclear_mass_u(a, z):.6f} u")
    d3.metric("Atomic mass", f"{atomic_mass_u(a, z):.6f} u")


def explanation_section(z: int, a: int) -> None:
    n = a - z
    z0 = approx_stable_z(a)
    delta_z = z - z0

    with st.expander("Physics interpretation", expanded=True):
        st.markdown(f"**Selected isotope:** {isotope_name(z, a)}")
        st.markdown(f"**Total binding energy:** {semf_binding_energy_mev(a, z):.3f} MeV")
        st.markdown(f"**Binding energy per nucleon:** {be_per_nucleon(a, z):.3f} MeV")

        st.markdown("**Mass defect relation**")
        st.latex(r"\Delta m = Zm_p + Nm_n - m_{\mathrm{nucleus}}")

        st.markdown("**Binding energy relation**")
        st.latex(r"E_b = \Delta m c^2")

        st.markdown("**What this means:**")
        st.write(stability_comment(a, z))

        st.markdown("**Valley-of-stability check:**")
        st.write(
            f"For A = {a}, the approximate stable proton number is "
            f"Z ≈ {z0:.2f}."
        )
        st.write(f"This isotope differs from that trend by ΔZ ≈ {delta_z:+.2f}.")

        st.markdown("**Shell hint:**")
        st.write(shell_hint(z, n))

        st.markdown("**Reaction trend:**")
        st.write(reaction_trend_text(a, z))


def main_visuals(z: int, a: int) -> None:
    left, right = st.columns([1.0, 1.25])

    with left:
        st.plotly_chart(nucleus_figure(z, a), width="stretch")

    with right:
        st.plotly_chart(be_curve_figure(a, z), width="stretch")

    st.plotly_chart(stability_map_figure(a, z), width="stretch")


def comparison_section() -> None:
    if not st.session_state[state_key("compare_enabled")]:
        return

    lookup = preset_lookup()
    chosen = [
        st.session_state[state_key("compare_1")],
        st.session_state[state_key("compare_2")],
        st.session_state[state_key("compare_3")],
    ]

    rows = []
    seen = set()
    for label in chosen:
        if label in lookup and label not in seen:
            item = lookup[label]
            rows.append(compute_isotope_row(item["A"], item["Z"]))
            seen.add(label)

    if not rows:
        return

    st.subheader("Isotope comparison")
    st.plotly_chart(comparison_figure(rows), width="stretch")

    df = pd.DataFrame(rows)
    numeric_cols = [
        "Z", "N", "A",
        "Binding Energy (MeV)",
        "BE/A (MeV)",
        "Mass Defect (u)",
        "Nuclear Mass (u)",
        "Atomic Mass (u)",
    ]
    styled = df.copy()
    for col in numeric_cols:
        if "BE/A" in col:
            styled[col] = styled[col].map(lambda x: f"{x:.3f}")
        elif "Mass" in col or "Defect" in col:
            styled[col] = styled[col].map(lambda x: f"{x:.6f}")
        elif "Binding Energy" in col:
            styled[col] = styled[col].map(lambda x: f"{x:.3f}")
        else:
            styled[col] = styled[col].map(lambda x: f"{int(x)}")

    st.dataframe(styled, width="stretch", hide_index=True)

    best = df.loc[df["BE/A (MeV)"].idxmax()]
    worst = df.loc[df["BE/A (MeV)"].idxmin()]
    st.markdown(
        f"""
**Comparison readout:**  
- Highest BE/A in this set: **{best["Isotope"]}** ({best["BE/A (MeV)"]:.3f} MeV)  
- Lowest BE/A in this set: **{worst["Isotope"]}** ({worst["BE/A (MeV)"]:.3f} MeV)  
This is why light nuclei can gain from fusion and very heavy nuclei can gain from fission: both move toward the more tightly bound mid-mass region.
"""
    )


def educational_notes() -> None:
    with st.expander("How to read the plots"):
        st.markdown(
            """
- **Binding energy per nucleon curve:** the closer a nucleus is to the peak, the more tightly bound it is per particle.
- **Left side of the peak:** fusion can move nuclei upward toward larger BE/A.
- **Right side of the peak:** fission can move very heavy nuclei upward toward larger BE/A.
- **Stability map:** the dashed line is an approximate center of the valley of stability.  
  Nuclei far above or below it tend to be neutron-rich or proton-rich.
- **Stylized nucleus view:** this is a conceptual display of proton and neutron counts, not a literal orbital model of the nucleus.
"""
        )

    with st.expander("Model limitations"):
        st.markdown(
            """
This app uses the **semi-empirical mass formula**. That makes it excellent for:
- trend visualization
- teaching nuclear stability
- comparing fusion/fission intuition

But it is not a replacement for evaluated nuclear data tables.  
For very light nuclei or detailed isotope-by-isotope precision, real experimental mass tables are better.
"""
        )


# =========================================================
# Main entry point
# =========================================================
def run() -> None:
    init_state()
    header_section()
    z, a = controls_section()

    st.markdown("### Selected isotope summary")
    metrics_section(z, a)
    explanation_section(z, a)

    st.markdown("### Visual explorer")
    main_visuals(z, a)

    comparison_section()

    st.markdown("### Notes")
    educational_notes()


if __name__ == "__main__":
    run()