import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# ============================================================
# Electric Field Module
# ============================================================

MODULE_KEY = "electric_field_lab"


# ------------------------------------------------------------
# State helpers
# ------------------------------------------------------------
def ss(key: str) -> str:
    return f"{MODULE_KEY}_{key}"


def init_state():
    defaults = {
        ss("preset"): "Dipole",
        ss("grid_points"): 180,
        ss("plot_range"): 5.0,
        ss("show_heatmap"): True,
        ss("show_stream"): True,
        ss("show_vectors"): False,
        ss("show_potential"): True,
        ss("normalize_vectors"): True,
        ss("vector_density"): 18,
        ss("stream_density"): 1.4,
        ss("probe_x"): 0.0,
        ss("probe_y"): 0.0,
        ss("num_charges"): 2,
        ss("q_0"): 1.0,
        ss("x_0"): -1.5,
        ss("y_0"): 0.0,
        ss("q_1"): -1.0,
        ss("x_1"): 1.5,
        ss("y_1"): 0.0,
        ss("q_2"): 1.0,
        ss("x_2"): -1.5,
        ss("y_2"): 1.5,
        ss("q_3"): -1.0,
        ss("x_3"): 1.5,
        ss("y_3"): -1.5,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ------------------------------------------------------------
# Presets
# ------------------------------------------------------------
def load_preset(preset_name: str):
    presets = {
        "Single Positive Charge": {
            "num_charges": 1,
            "charges": [(1.0, 0.0, 0.0)],
        },
        "Single Negative Charge": {
            "num_charges": 1,
            "charges": [(-1.0, 0.0, 0.0)],
        },
        "Dipole": {
            "num_charges": 2,
            "charges": [(1.0, -1.5, 0.0), (-1.0, 1.5, 0.0)],
        },
        "Two Like Charges": {
            "num_charges": 2,
            "charges": [(1.0, -1.5, 0.0), (1.0, 1.5, 0.0)],
        },
        "Quadrupole": {
            "num_charges": 4,
            "charges": [
                (1.0, -1.5, 1.5),
                (-1.0, 1.5, 1.5),
                (-1.0, -1.5, -1.5),
                (1.0, 1.5, -1.5),
            ],
        },
        "Triangle Configuration": {
            "num_charges": 3,
            "charges": [
                (1.0, -1.8, -1.0),
                (1.0, 1.8, -1.0),
                (-1.5, 0.0, 1.8),
            ],
        },
    }

    preset = presets[preset_name]
    st.session_state[ss("num_charges")] = preset["num_charges"]

    for i in range(4):
        if i < preset["num_charges"]:
            q, x0, y0 = preset["charges"][i]
        else:
            q, x0, y0 = 0.0, 0.0, 0.0

        st.session_state[ss(f"q_{i}")] = q
        st.session_state[ss(f"x_{i}")] = x0
        st.session_state[ss(f"y_{i}")] = y0


# ------------------------------------------------------------
# Physics
# ------------------------------------------------------------
def build_charge_list():
    charges = []
    n = st.session_state[ss("num_charges")]
    for i in range(n):
        q = float(st.session_state[ss(f"q_{i}")])
        x0 = float(st.session_state[ss(f"x_{i}")])
        y0 = float(st.session_state[ss(f"y_{i}")])
        charges.append((q, x0, y0))
    return charges


def compute_field_and_potential(X, Y, charges, mask_radius=0.18):
    Ex = np.zeros_like(X, dtype=float)
    Ey = np.zeros_like(Y, dtype=float)
    V = np.zeros_like(X, dtype=float)

    mask = np.zeros_like(X, dtype=bool)

    for q, cx, cy in charges:
        dx = X - cx
        dy = Y - cy
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)

        local_mask = r < mask_radius
        mask |= local_mask

        safe_r = np.where(local_mask, np.nan, r)
        safe_r3 = safe_r**3

        Ex += q * dx / safe_r3
        Ey += q * dy / safe_r3
        V += q / safe_r

    Ex = np.ma.array(Ex, mask=mask)
    Ey = np.ma.array(Ey, mask=mask)
    V = np.ma.array(V, mask=mask)
    Emag = np.ma.sqrt(Ex**2 + Ey**2)

    return Ex, Ey, V, Emag, mask


def field_at_point(px, py, charges, min_r=0.08):
    Ex = 0.0
    Ey = 0.0
    V = 0.0

    for q, cx, cy in charges:
        dx = px - cx
        dy = py - cy
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)

        if r < min_r:
            return None

        Ex += q * dx / (r**3)
        Ey += q * dy / (r**3)
        V += q / r

    Emag = np.sqrt(Ex**2 + Ey**2)
    angle = np.degrees(np.arctan2(Ey, Ex))
    return Ex, Ey, V, Emag, angle


def detect_pattern(charges):
    n = len(charges)
    total_q = sum(q for q, _, _ in charges)

    if n == 2:
        q1, x1, y1 = charges[0]
        q2, x2, y2 = charges[1]
        if np.isclose(q1, -q2):
            return "Dipole-like configuration"
        if np.sign(q1) == np.sign(q2):
            return "Like-charge repulsion pattern"

    if n == 4 and np.isclose(total_q, 0.0, atol=1e-6):
        return "Quadrupole-like configuration"

    if np.isclose(total_q, 0.0, atol=1e-6):
        return "Net-neutral multi-charge configuration"

    if total_q > 0:
        return "Net positive configuration"
    if total_q < 0:
        return "Net negative configuration"

    return "General multi-charge configuration"


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
def get_symmetric_potential_levels(V, num_levels=14):
    v_data = V.compressed()
    if v_data.size == 0:
        return np.linspace(-1, 1, num_levels)

    vmax = np.nanpercentile(np.abs(v_data), 92)
    vmax = max(vmax, 0.5)
    levels = np.linspace(-vmax, vmax, num_levels)
    levels = levels[np.abs(levels) > 1e-10]
    return levels


def plot_field_map(X, Y, Ex, Ey, V, Emag, charges, plot_range):
    show_heatmap = st.session_state[ss("show_heatmap")]
    show_stream = st.session_state[ss("show_stream")]
    show_vectors = st.session_state[ss("show_vectors")]
    show_potential = st.session_state[ss("show_potential")]
    normalize_vectors = st.session_state[ss("normalize_vectors")]
    vector_density = st.session_state[ss("vector_density")]
    stream_density = st.session_state[ss("stream_density")]

    probe_x = st.session_state[ss("probe_x")]
    probe_y = st.session_state[ss("probe_y")]

    fig, ax = plt.subplots(figsize=(8.8, 8.2))

    # Heatmap of |E|
    if show_heatmap:
        em_data = Emag.compressed()
        if em_data.size > 0:
            vmax = np.nanpercentile(em_data, 95)
            vmax = max(vmax, 1.0)
        else:
            vmax = 1.0

        heat = ax.contourf(
            X,
            Y,
            np.ma.clip(Emag, 0, vmax),
            levels=35,
            cmap="magma",
            alpha=0.88,
        )
        cbar = fig.colorbar(heat, ax=ax, pad=0.02, fraction=0.045)
        cbar.set_label(r"Field strength $|E|$", fontsize=11)

    # Equipotentials
    if show_potential:
        levels = get_symmetric_potential_levels(V)
        contour = ax.contour(
            X,
            Y,
            V,
            levels=levels,
            colors="white" if show_heatmap else "dimgray",
            linewidths=0.8,
            alpha=0.7,
        )
        ax.clabel(contour, inline=True, fontsize=7, fmt="%.1f")

    # Streamlines
    if show_stream:
        lw = 0.9 + 1.8 * (Emag / (np.nanmax(Emag.filled(np.nan)) + 1e-12))
        lw = np.ma.filled(lw, 0.0)
        ax.streamplot(
            X,
            Y,
            Ex.filled(np.nan),
            Ey.filled(np.nan),
            density=stream_density,
            linewidth=lw,
            color="cyan" if show_heatmap else "tab:blue",
            arrowsize=1.1,
            minlength=0.15,
            maxlength=5.0,
            zorder=3,
        )

    # Vectors
    if show_vectors:
        step = max(1, X.shape[0] // vector_density)

        qx = X[::step, ::step]
        qy = Y[::step, ::step]
        qEx = Ex[::step, ::step].filled(np.nan)
        qEy = Ey[::step, ::step].filled(np.nan)

        if normalize_vectors:
            qEmag = np.sqrt(qEx**2 + qEy**2)
            qEx = np.divide(qEx, qEmag, out=np.zeros_like(qEx), where=qEmag > 1e-12)
            qEy = np.divide(qEy, qEmag, out=np.zeros_like(qEy), where=qEmag > 1e-12)

        ax.quiver(
            qx,
            qy,
            qEx,
            qEy,
            color="white" if show_heatmap else "black",
            alpha=0.75,
            scale=24 if normalize_vectors else None,
            width=0.003,
            zorder=4,
        )

    # Charges
    for i, (q, cx, cy) in enumerate(charges, start=1):
        if q > 0:
            facecolor = "#ff4b4b"
            edgecolor = "#7a0000"
            sign = "+"
        else:
            facecolor = "#4b7bff"
            edgecolor = "#001f7a"
            sign = "−"

        ax.scatter(
            cx,
            cy,
            s=220,
            c=facecolor,
            edgecolors=edgecolor,
            linewidths=2.0,
            zorder=6,
        )
        ax.text(
            cx,
            cy,
            sign,
            ha="center",
            va="center",
            color="white",
            fontsize=15,
            fontweight="bold",
            zorder=7,
        )
        ax.text(
            cx + 0.18,
            cy + 0.18,
            f"q{i} = {q:+.1f}",
            fontsize=9,
            color="white" if show_heatmap else "black",
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor=(0, 0, 0, 0.38) if show_heatmap else (1, 1, 1, 0.8),
                edgecolor="none",
            ),
            zorder=8,
        )

    # Probe point
    ax.scatter(
        probe_x,
        probe_y,
        s=90,
        marker="x",
        color="lime" if show_heatmap else "green",
        linewidths=2.2,
        zorder=8,
    )
    ax.text(
        probe_x + 0.15,
        probe_y + 0.15,
        "Probe",
        fontsize=9,
        color="white" if show_heatmap else "black",
        zorder=8,
    )

    ax.set_xlim(-plot_range, plot_range)
    ax.set_ylim(-plot_range, plot_range)
    ax.set_aspect("equal")
    ax.set_xlabel("x position", fontsize=11)
    ax.set_ylabel("y position", fontsize=11)
    ax.set_title("Electric Field, Potential, and Probe Analysis", fontsize=14, pad=12)
    ax.grid(alpha=0.18)

    return fig


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚡ Electric Field Lab")

        preset = st.selectbox(
            "Preset configuration",
            [
                "Dipole",
                "Single Positive Charge",
                "Single Negative Charge",
                "Two Like Charges",
                "Quadrupole",
                "Triangle Configuration",
            ],
            index=0,
            key=ss("preset"),
        )

        if st.button("Load preset", key=ss("load_preset_button"), use_container_width=True):
            load_preset(preset)

        st.markdown("---")
        st.markdown("### Grid and domain")

        st.slider(
            "Grid resolution",
            min_value=80,
            max_value=260,
            value=st.session_state[ss("grid_points")],
            step=10,
            key=ss("grid_points"),
        )

        st.slider(
            "Axis range",
            min_value=2.0,
            max_value=10.0,
            value=st.session_state[ss("plot_range")],
            step=0.5,
            key=ss("plot_range"),
        )

        st.markdown("---")
        st.markdown("### Charge setup")

        st.slider(
            "Number of charges",
            min_value=1,
            max_value=4,
            value=st.session_state[ss("num_charges")],
            step=1,
            key=ss("num_charges"),
        )

        plot_range = float(st.session_state[ss("plot_range")])
        n = st.session_state[ss("num_charges")]

        for i in range(n):
            with st.expander(f"Charge {i + 1}", expanded=(i < 2)):
                st.slider(
                    f"q{i + 1}",
                    min_value=-5.0,
                    max_value=5.0,
                    value=float(st.session_state[ss(f"q_{i}")]),
                    step=0.1,
                    key=ss(f"q_{i}"),
                )
                st.slider(
                    f"x{i + 1}",
                    min_value=-plot_range,
                    max_value=plot_range,
                    value=float(st.session_state[ss(f"x_{i}")]),
                    step=0.1,
                    key=ss(f"x_{i}"),
                )
                st.slider(
                    f"y{i + 1}",
                    min_value=-plot_range,
                    max_value=plot_range,
                    value=float(st.session_state[ss(f"y_{i}")]),
                    step=0.1,
                    key=ss(f"y_{i}"),
                )

        st.markdown("---")
        st.markdown("### Display options")

        st.checkbox("Show field-strength heatmap", key=ss("show_heatmap"))
        st.checkbox("Show field lines", key=ss("show_stream"))
        st.checkbox("Show equipotential contours", key=ss("show_potential"))
        st.checkbox("Show electric field vectors", key=ss("show_vectors"))

        if st.session_state[ss("show_vectors")]:
            st.checkbox("Normalize vectors", key=ss("normalize_vectors"))
            st.slider(
                "Vector density",
                min_value=10,
                max_value=30,
                value=st.session_state[ss("vector_density")],
                step=1,
                key=ss("vector_density"),
            )

        if st.session_state[ss("show_stream")]:
            st.slider(
                "Field-line density",
                min_value=0.6,
                max_value=2.4,
                value=st.session_state[ss("stream_density")],
                step=0.1,
                key=ss("stream_density"),
            )

        st.markdown("---")
        st.markdown("### Probe point")

        st.slider(
            "Probe x",
            min_value=-plot_range,
            max_value=plot_range,
            value=float(st.session_state[ss("probe_x")]),
            step=0.1,
            key=ss("probe_x"),
        )
        st.slider(
            "Probe y",
            min_value=-plot_range,
            max_value=plot_range,
            value=float(st.session_state[ss("probe_y")]),
            step=0.1,
            key=ss("probe_y"),
        )


def render_metrics(charges):
    probe_x = st.session_state[ss("probe_x")]
    probe_y = st.session_state[ss("probe_y")]

    total_q = sum(q for q, _, _ in charges)
    pattern = detect_pattern(charges)
    probe = field_at_point(probe_x, probe_y, charges)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Number of charges", len(charges))
    col2.metric("Net charge", f"{total_q:+.2f}")
    col3.metric("Pattern", pattern)
    col4.metric("Probe point", f"({probe_x:.2f}, {probe_y:.2f})")

    st.markdown("### Probe analysis")

    if probe is None:
        st.warning(
            "The probe point is too close to a charge, so the point-charge model becomes singular there. "
            "Move the probe slightly away to inspect the field safely."
        )
        return

    Ex, Ey, V, Emag, angle = probe

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(r"$E_x$", f"{Ex:.3f}")
    c2.metric(r"$E_y$", f"{Ey:.3f}")
    c3.metric(r"$|E|$", f"{Emag:.3f}")
    c4.metric(r"$V$", f"{V:.3f}")
    c5.metric("Direction", f"{angle:.1f}°")

    with st.expander("Interpretation", expanded=True):
        if np.isclose(Emag, 0.0, atol=1e-3):
            st.info(
                "The electric field at the probe is very small, which suggests local cancellation "
                "between charge contributions."
            )
        else:
            direction_text = (
                "predominantly rightward" if abs(angle) < 45
                else "predominantly upward" if 45 <= angle < 135
                else "predominantly leftward" if abs(angle) >= 135
                else "predominantly downward"
            )
            st.write(
                f"At the probe point, the net electric field is **{direction_text}**. "
                f"The magnitude is **{Emag:.3f}**, and the potential is **{V:.3f}** in arbitrary units."
            )

        st.write(
            f"This configuration is classified as **{pattern.lower()}**. "
            "The plot combines superposition from all selected charges."
        )

def render_theory():
    st.markdown("### Physics notes")
    st.markdown(
    r"""
This module uses the point-charge model in arbitrary units:

- Electric potential:
  $$
  V(\mathbf{r}) = \sum_i \frac{q_i}{r_i}
  $$

- Electric field:
  $$
  \mathbf{E}(\mathbf{r}) = \sum_i q_i \frac{\mathbf{r} - \mathbf{r}_i}{|\mathbf{r} - \mathbf{r}_i|^3}
  $$

where \($$ r_i = |\mathbf{r} - \mathbf{r}_i| $$ \)  is the distance from the field point to the \(i\)-th charge.

**How to read the plot**
- **Heatmap** shows field strength \( $$|\mathbf{E}|$$ \)
- **Field lines** show the direction of the electric field
- **Equipotentials** connect points of equal electric potential
- **Probe point** gives local numerical values of \($$E_x$$\), \($$E_y$$\), \( $$|\mathbf{E}|$$ \), and \(V\)

A small circular region around each charge is masked in the plot to avoid divergence at the exact charge position in the point-charge model.
"""
)

# ------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------
def render():
    init_state()

    st.title("⚡ Electric Field and Equipotential Explorer")
    st.write(
        "Build charge configurations, compare classic electrostatic patterns, and inspect the electric field and potential at any probe point."
    )

    render_sidebar()

    charges = build_charge_list()
    grid_points = int(st.session_state[ss("grid_points")])
    plot_range = float(st.session_state[ss("plot_range")])

    x = np.linspace(-plot_range, plot_range, grid_points)
    y = np.linspace(-plot_range, plot_range, grid_points)
    X, Y = np.meshgrid(x, y)

    Ex, Ey, V, Emag, _ = compute_field_and_potential(X, Y, charges)

    top_left, top_right = st.columns([1.45, 1.0], gap="large")

    with top_left:
        fig = plot_field_map(X, Y, Ex, Ey, V, Emag, charges, plot_range)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    with top_right:
        render_metrics(charges)

    st.markdown("---")
    render_theory()


# Optional direct run
if __name__ == "__main__":
    render()