import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# Visual theme
# ============================================================
THEME = {
    "bg": "#ffffff",
    "grid": "rgba(120, 140, 160, 0.22)",
    "axis": "#31435a",
    "text": "#1f2937",
    "particle": "#ef4444",
    "trail": "#2563eb",
    "vector": "#06b6d4",
    "density_scale": "Viridis",
}


# ============================================================
# Constants
# ============================================================
SECTION = "fluidflow"
MAX_REASONABLE_DIFFUSION = 0.2


# ============================================================
# Helpers
# ============================================================
def nice_number(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = np.floor(np.log10(value))
    fraction = value / (10 ** exponent)
    if fraction < 1.5:
        nice_fraction = 1.0
    elif fraction < 3.0:
        nice_fraction = 2.0
    elif fraction < 7.0:
        nice_fraction = 5.0
    else:
        nice_fraction = 10.0
    return nice_fraction * (10 ** exponent)


def padded_range(values: np.ndarray, pad_frac: float = 0.12):
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        delta = max(abs(vmin) * 0.15, 1.0)
        return [vmin - delta, vmax + delta]
    span = vmax - vmin
    pad = nice_number(span * pad_frac)
    return [vmin - pad, vmax + pad]


def common_axis_style(title: str, axis_range=None):
    style = dict(
        title=title,
        showgrid=True,
        gridcolor=THEME["grid"],
        gridwidth=1,
        zeroline=True,
        zerolinecolor="rgba(80, 100, 120, 0.28)",
        linecolor=THEME["axis"],
        tickfont=dict(size=12, color=THEME["text"]),
        title_font=dict(size=14, color=THEME["text"]),
        tickformat=".3~g",
        exponentformat="power",
        showexponent="last",
        ticks="outside",
        ticklen=6,
    )
    if axis_range is not None:
        style["range"] = axis_range
    return style


def current_signature(*items):
    out = []
    for item in items:
        if isinstance(item, np.ndarray):
            out.append(tuple(np.round(item.astype(float), 12)))
        else:
            out.append(item)
    return tuple(out)


# ============================================================
# Session state
# ============================================================
def init_fluid_state():
    defaults = {
        f"{SECTION}_playing": False,
        f"{SECTION}_last_update": time.time(),
        f"{SECTION}_sim_time": 0.0,
        f"{SECTION}_particles": None,
        f"{SECTION}_history": None,
        f"{SECTION}_history_times": None,
        f"{SECTION}_prev_signature": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_simulation(particles0: np.ndarray, trail_points: int):
    st.session_state[f"{SECTION}_playing"] = False
    st.session_state[f"{SECTION}_last_update"] = time.time()
    st.session_state[f"{SECTION}_sim_time"] = 0.0
    st.session_state[f"{SECTION}_particles"] = particles0.copy()
    st.session_state[f"{SECTION}_history"] = [particles0.copy() for _ in range(trail_points)]
    st.session_state[f"{SECTION}_history_times"] = [0.0 for _ in range(trail_points)]


def reseed_particles(n_particles: int, domain_half: float, injection_mode: str, trail_points: int):
    particles0 = initialize_particles(n_particles, domain_half, injection_mode)
    reset_simulation(particles0, trail_points)


# ============================================================
# Physics
# ============================================================
def velocity_field(x, y, flow_type, strength, core_size):
    """
    Prescribed 2D velocity fields for passive tracer transport.

    Notes:
    - Uniform flow: constant translation
    - Shear flow: u = k y
    - Vortex flow: solid-body rotation, better behaved for teaching
    - Source/Sink: softened radial flow
    """
    eps = core_size

    if flow_type == "Uniform Flow":
        u = strength * np.ones_like(x)
        v = np.zeros_like(y)

    elif flow_type == "Shear Flow":
        u = strength * y
        v = np.zeros_like(y)

    elif flow_type == "Vortex Flow":
        # Solid-body rotation: bounded and physically cleaner for an educational demo
        u = -strength * y
        v = strength * x

    elif flow_type == "Source Flow":
        r = np.sqrt(x**2 + y**2 + eps**2)
        u = strength * x / r
        v = strength * y / r

    elif flow_type == "Sink Flow":
        r = np.sqrt(x**2 + y**2 + eps**2)
        u = -strength * x / r
        v = -strength * y / r

    else:
        u = np.zeros_like(x)
        v = np.zeros_like(y)

    return u, v


def initialize_particles(n_particles: int, domain_half: float, injection_mode: str) -> np.ndarray:
    xmin, xmax = -domain_half, domain_half
    ymin, ymax = -domain_half, domain_half

    if injection_mode == "Random Domain":
        x = np.random.uniform(xmin, xmax, n_particles)
        y = np.random.uniform(ymin, ymax, n_particles)

    elif injection_mode == "Left Edge":
        x = np.random.uniform(xmin, xmin + 0.08 * (xmax - xmin), n_particles)
        y = np.random.uniform(ymin, ymax, n_particles)

    elif injection_mode == "Center Blob":
        x = np.random.normal(0.0, 0.12 * domain_half, n_particles)
        y = np.random.normal(0.0, 0.12 * domain_half, n_particles)

    elif injection_mode == "Ring":
        theta = np.random.uniform(0, 2 * np.pi, n_particles)
        r = np.random.normal(0.55 * domain_half, 0.08 * domain_half, n_particles)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

    else:
        x = np.random.uniform(xmin, xmax, n_particles)
        y = np.random.uniform(ymin, ymax, n_particles)

    x = np.clip(x, xmin, xmax)
    y = np.clip(y, ymin, ymax)
    return np.column_stack([x, y])


def apply_boundaries(particles: np.ndarray, domain_half: float, boundary_mode: str) -> np.ndarray:
    xmin, xmax = -domain_half, domain_half
    ymin, ymax = -domain_half, domain_half

    x = particles[:, 0]
    y = particles[:, 1]

    if boundary_mode == "Wrap":
        Lx = xmax - xmin
        Ly = ymax - ymin
        x = xmin + np.mod(x - xmin, Lx)
        y = ymin + np.mod(y - ymin, Ly)

    elif boundary_mode == "Reflect":
        x = np.where(x < xmin, 2 * xmin - x, x)
        x = np.where(x > xmax, 2 * xmax - x, x)
        y = np.where(y < ymin, 2 * ymin - y, y)
        y = np.where(y > ymax, 2 * ymax - y, y)
        x = np.clip(x, xmin, xmax)
        y = np.clip(y, ymin, ymax)

    elif boundary_mode == "Clamp":
        x = np.clip(x, xmin, xmax)
        y = np.clip(y, ymin, ymax)

    return np.column_stack([x, y])


def rk2_step_particles(
    particles: np.ndarray,
    dt: float,
    flow_type: str,
    strength: float,
    diffusion: float,
    core_size: float,
    domain_half: float,
    boundary_mode: str,
):
    x = particles[:, 0]
    y = particles[:, 1]

    u1, v1 = velocity_field(x, y, flow_type, strength, core_size)
    x_mid = x + 0.5 * dt * u1
    y_mid = y + 0.5 * dt * v1

    u2, v2 = velocity_field(x_mid, y_mid, flow_type, strength, core_size)

    x_new = x + dt * u2
    y_new = y + dt * v2

    if diffusion > 0:
        sigma = np.sqrt(2 * diffusion * dt)
        x_new += np.random.normal(0, sigma, size=len(x_new))
        y_new += np.random.normal(0, sigma, size=len(y_new))

    updated = np.column_stack([x_new, y_new])
    updated = apply_boundaries(updated, domain_half, boundary_mode)

    speed = np.sqrt(u2**2 + v2**2)
    return updated, u2, v2, speed


def update_simulation(
    flow_type: str,
    strength: float,
    diffusion: float,
    core_size: float,
    domain_half: float,
    boundary_mode: str,
    dt_user: float,
    playback_speed: float,
    t_max: float,
    trail_points: int,
):
    now = time.time()
    elapsed_wall = now - st.session_state[f"{SECTION}_last_update"]
    st.session_state[f"{SECTION}_last_update"] = now

    if not st.session_state[f"{SECTION}_playing"]:
        return

    target_sim_advance = elapsed_wall * playback_speed
    if target_sim_advance <= 0:
        return

    n_substeps = max(1, int(np.ceil(target_sim_advance / dt_user)))
    dt = target_sim_advance / n_substeps

    particles = st.session_state[f"{SECTION}_particles"].copy()
    history = list(st.session_state[f"{SECTION}_history"])
    history_times = list(st.session_state[f"{SECTION}_history_times"])

    for _ in range(n_substeps):
        if st.session_state[f"{SECTION}_sim_time"] >= t_max:
            st.session_state[f"{SECTION}_playing"] = False
            break

        particles, _, _, _ = rk2_step_particles(
            particles=particles,
            dt=dt,
            flow_type=flow_type,
            strength=strength,
            diffusion=diffusion,
            core_size=core_size,
            domain_half=domain_half,
            boundary_mode=boundary_mode,
        )

        st.session_state[f"{SECTION}_sim_time"] += dt
        history.append(particles.copy())
        history_times.append(st.session_state[f"{SECTION}_sim_time"])

    history = history[-trail_points:]
    history_times = history_times[-trail_points:]

    st.session_state[f"{SECTION}_particles"] = particles
    st.session_state[f"{SECTION}_history"] = history
    st.session_state[f"{SECTION}_history_times"] = history_times


# ============================================================
# Diagnostics
# ============================================================
def classify_flow(flow_type: str, diffusion: float) -> str:
    if flow_type == "Uniform Flow" and diffusion <= 0:
        return "Pure advection"
    if flow_type == "Shear Flow" and diffusion <= 0:
        return "Advective shear transport"
    if flow_type == "Vortex Flow" and diffusion <= 0:
        return "Rotational transport"
    if diffusion > 0:
        return "Advection–diffusion transport"
    return "Passive tracer transport"


def compute_stats(particles: np.ndarray, flow_type: str, strength: float, core_size: float):
    x = particles[:, 0]
    y = particles[:, 1]
    u, v = velocity_field(x, y, flow_type, strength, core_size)
    speed = np.sqrt(u**2 + v**2)

    return {
        "mean_speed": float(np.mean(speed)),
        "max_speed": float(np.max(speed)),
        "spread_x": float(np.std(x)),
        "spread_y": float(np.std(y)),
        "mean_x": float(np.mean(x)),
        "mean_y": float(np.mean(y)),
    }


# ============================================================
# Plotting
# ============================================================
def density_heatmap_trace(particles: np.ndarray, domain_half: float, bins: int = 40):
    H, xedges, yedges = np.histogram2d(
        particles[:, 0],
        particles[:, 1],
        bins=bins,
        range=[[-domain_half, domain_half], [-domain_half, domain_half]],
    )

    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])

    return go.Heatmap(
        x=xcenters,
        y=ycenters,
        z=H.T,
        colorscale=THEME["density_scale"],
        opacity=0.72,
        colorbar=dict(title="Particle density"),
        zsmooth="best",
        showscale=True,
        name="Density",
        hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>count=%{z}<extra></extra>",
    )


def render_flow_scene(
    particles: np.ndarray,
    history: list,
    flow_type: str,
    strength: float,
    core_size: float,
    domain_half: float,
    show_vectors: bool,
    show_density: bool,
    vector_density: int,
):
    xmin, xmax = -domain_half, domain_half
    ymin, ymax = -domain_half, domain_half

    fig = go.Figure()

    # Density background first
    if show_density:
        fig.add_trace(density_heatmap_trace(particles, domain_half, bins=40))

    # Trails
    if history is not None and len(history) > 1:
        n_particles = particles.shape[0]
        hist_arr = np.array(history)

        for i in range(n_particles):
            fig.add_trace(
                go.Scatter(
                    x=hist_arr[:, i, 0],
                    y=hist_arr[:, i, 1],
                    mode="lines",
                    line=dict(color=THEME["trail"], width=1.2),
                    opacity=0.18,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Vector field
    if show_vectors:
        gx = np.linspace(xmin, xmax, vector_density)
        gy = np.linspace(ymin, ymax, vector_density)
        X, Y = np.meshgrid(gx, gy)
        U, V = velocity_field(X, Y, flow_type, strength, core_size)

        step_scale = 0.18 * (2 * domain_half) / max(vector_density, 10)
        Xf = X.flatten()
        Yf = Y.flatten()
        Uf = U.flatten()
        Vf = V.flatten()

        for x0, y0, u0, v0 in zip(Xf, Yf, Uf, Vf):
            mag = np.hypot(u0, v0)
            if mag < 1e-12:
                continue
            dx = step_scale * u0 / mag
            dy = step_scale * v0 / mag
            fig.add_trace(
                go.Scatter(
                    x=[x0, x0 + dx],
                    y=[y0, y0 + dy],
                    mode="lines",
                    line=dict(color=THEME["vector"], width=2),
                    opacity=0.50,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Current particles last so they stay visible
    fig.add_trace(
        go.Scatter(
            x=particles[:, 0],
            y=particles[:, 1],
            mode="markers",
            marker=dict(size=7, color=THEME["particle"]),
            name="Particles",
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text="Fluid Flow and Particle Transport", x=0.5, xanchor="center"),
        xaxis=common_axis_style("x", [xmin, xmax]),
        yaxis={**common_axis_style("y", [ymin, ymax]), "scaleanchor": "x", "scaleratio": 1},
        template="plotly_white",
        plot_bgcolor=THEME["bg"],
        paper_bgcolor=THEME["bg"],
        font=dict(size=14, color=THEME["text"]),
        height=620,
        margin=dict(l=70, r=20, t=70, b=55),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def line_plot(x, y, title, y_label, current_x=None):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_arr,
            y=y_arr,
            mode="lines",
            line=dict(width=3, color=THEME["trail"], shape="spline", smoothing=0.65),
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.08)",
            showlegend=False,
        )
    )

    if current_x is not None:
        fig.add_vline(x=current_x, line_width=2, line_dash="dash", line_color=THEME["particle"])

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis=common_axis_style("Time (s)", padded_range(x_arr, 0.08)),
        yaxis=common_axis_style(y_label, padded_range(y_arr, 0.15)),
        template="plotly_white",
        plot_bgcolor=THEME["bg"],
        paper_bgcolor=THEME["bg"],
        height=320,
        margin=dict(l=70, r=20, t=55, b=55),
        font=dict(size=14, color=THEME["text"]),
        showlegend=False,
    )
    return fig


# ============================================================
# Main app
# ============================================================
def run():
    init_fluid_state()

    st.title("Fluid Flow and Particle Transport Lab")

    st.markdown("### Governing equation")
    st.latex(r"\frac{d\mathbf{x}}{dt}=\mathbf{v}(x,y)")

    st.markdown("### Advection–diffusion extension")
    st.latex(r"d\mathbf{x}=\mathbf{v}(x,y)\,dt+\sqrt{2D\,dt}\,\eta")

    st.markdown(
        """
This simulation shows **passive tracer transport** in idealized 2D flow fields.

- particles are carried by a prescribed flow field
- optional diffusion adds random spreading
- density is shown using a full-domain heatmap built from particle counts
"""
    )

    # ---------------- Sidebar ----------------
    st.sidebar.header("Flow setup")

    flow_type = st.sidebar.selectbox(
        "Flow type",
        ["Uniform Flow", "Shear Flow", "Vortex Flow", "Source Flow", "Sink Flow"],
        key=f"{SECTION}_flow_type",
    )

    strength = st.sidebar.slider(
        "Flow strength",
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1,
        key=f"{SECTION}_strength",
    )

    core_size = st.sidebar.slider(
        "Core softening",
        min_value=0.02,
        max_value=1.0,
        value=0.15,
        step=0.01,
        key=f"{SECTION}_core",
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Particles")

    n_particles = st.sidebar.slider(
        "Number of particles",
        min_value=20,
        max_value=1000,
        value=120,
        step=10,
        key=f"{SECTION}_n_particles",
    )

    injection_mode = st.sidebar.selectbox(
        "Initial distribution",
        ["Random Domain", "Left Edge", "Center Blob", "Ring"],
        key=f"{SECTION}_injection_mode",
    )

    diffusion = st.sidebar.slider(
        "Diffusion coefficient D",
        min_value=0.0,
        max_value=MAX_REASONABLE_DIFFUSION,
        value=0.02,
        step=0.005,
        key=f"{SECTION}_diffusion",
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Domain and time")

    domain_half = st.sidebar.slider(
        "Domain half-size",
        min_value=2.0,
        max_value=10.0,
        value=5.0,
        step=0.5,
        key=f"{SECTION}_domain_half",
    )

    boundary_mode = st.sidebar.selectbox(
        "Boundary mode",
        ["Wrap", "Reflect", "Clamp"],
        key=f"{SECTION}_boundary_mode",
    )

    dt_user = st.sidebar.slider(
        "Internal timestep",
        min_value=0.002,
        max_value=0.05,
        value=0.01,
        step=0.002,
        key=f"{SECTION}_dt",
    )

    t_max = st.sidebar.slider(
        "Simulation time window",
        min_value=5.0,
        max_value=120.0,
        value=40.0,
        step=5.0,
        key=f"{SECTION}_tmax",
    )

    playback_speed = st.sidebar.slider(
        "Playback speed",
        min_value=0.2,
        max_value=6.0,
        value=1.2,
        step=0.1,
        key=f"{SECTION}_playback",
    )

    trail_points = st.sidebar.slider(
        "Visible trail frames",
        min_value=10,
        max_value=250,
        value=60,
        step=10,
        key=f"{SECTION}_trail_points",
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Display")

    show_vectors = st.sidebar.checkbox(
        "Show vector field",
        value=True,
        key=f"{SECTION}_show_vectors",
    )

    show_density = st.sidebar.checkbox(
        "Show density map",
        value=True,
        key=f"{SECTION}_show_density",
    )

    vector_density = st.sidebar.slider(
        "Vector grid density",
        min_value=8,
        max_value=24,
        value=14,
        step=1,
        key=f"{SECTION}_vector_density",
    )

    # ---------------- Initialization signature ----------------
    signature = current_signature(
        flow_type,
        strength,
        core_size,
        n_particles,
        injection_mode,
        diffusion,
        domain_half,
        boundary_mode,
        dt_user,
        t_max,
        trail_points,
    )

    if st.session_state[f"{SECTION}_prev_signature"] != signature:
        st.session_state[f"{SECTION}_prev_signature"] = signature
        particles0 = initialize_particles(n_particles, domain_half, injection_mode)
        reset_simulation(particles0, trail_points)

    # ---------------- Controls ----------------
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("▶ Play", key=f"{SECTION}_play_btn", width="stretch"):
            if st.session_state[f"{SECTION}_sim_time"] >= t_max:
                particles0 = initialize_particles(n_particles, domain_half, injection_mode)
                reset_simulation(particles0, trail_points)
            st.session_state[f"{SECTION}_playing"] = True
            st.session_state[f"{SECTION}_last_update"] = time.time()
            st.rerun()

    with c2:
        if st.button("⏸ Pause", key=f"{SECTION}_pause_btn", width="stretch"):
            st.session_state[f"{SECTION}_playing"] = False
            st.rerun()

    with c3:
        if st.button("⟲ Reset / Reseed", key=f"{SECTION}_reset_btn", width="stretch"):
            reseed_particles(n_particles, domain_half, injection_mode, trail_points)
            st.rerun()

    progress = min(st.session_state[f"{SECTION}_sim_time"] / t_max, 1.0) if t_max > 0 else 0.0
    st.progress(progress)
    st.caption(f"Simulation progress: {100 * progress:.1f}%")

    # ---------------- Update state ----------------
    update_simulation(
        flow_type=flow_type,
        strength=strength,
        diffusion=diffusion,
        core_size=core_size,
        domain_half=domain_half,
        boundary_mode=boundary_mode,
        dt_user=dt_user,
        playback_speed=playback_speed,
        t_max=t_max,
        trail_points=trail_points,
    )

    particles = st.session_state[f"{SECTION}_particles"].copy()
    history = st.session_state[f"{SECTION}_history"]
    history_times = np.asarray(st.session_state[f"{SECTION}_history_times"], dtype=float)

    stats = compute_stats(particles, flow_type, strength, core_size)

    hist_arr = np.array(history)
    spread_x_hist = np.std(hist_arr[:, :, 0], axis=1)
    spread_y_hist = np.std(hist_arr[:, :, 1], axis=1)

    # ---------------- Metrics ----------------
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Flow strength", f"{strength:.2f}")
    m2.metric("Mean speed", f"{stats['mean_speed']:.3f}")
    m3.metric("Particles", f"{n_particles}")
    m4.metric("Transport regime", classify_flow(flow_type, diffusion))

    st.info(
        "In shear flow the cloud stretches, in vortex flow it rotates, and with diffusion enabled "
        "the cloud broadens over time. The color map shows particle concentration across the full domain."
    )

    # ---------------- Main view ----------------
    fig_main = render_flow_scene(
        particles=particles,
        history=history,
        flow_type=flow_type,
        strength=strength,
        core_size=core_size,
        domain_half=domain_half,
        show_vectors=show_vectors,
        show_density=show_density,
        vector_density=vector_density,
    )
    st.plotly_chart(fig_main, width="stretch")

    # ---------------- Diagnostics ----------------
    left, right = st.columns([1.15, 1.0])

    with left:
        r1, r2 = st.columns(2)
        with r1:
            st.plotly_chart(
                line_plot(history_times, spread_x_hist, "Spread in x vs time", "σx", st.session_state[f"{SECTION}_sim_time"]),
                width="stretch",
            )
        with r2:
            st.plotly_chart(
                line_plot(history_times, spread_y_hist, "Spread in y vs time", "σy", st.session_state[f"{SECTION}_sim_time"]),
                width="stretch",
            )

    with right:
        st.subheader("Current state")
        st.write(f"**Time:** {st.session_state[f'{SECTION}_sim_time']:.3f} s")
        st.write(f"**Mean x:** {stats['mean_x']:.3f}")
        st.write(f"**Mean y:** {stats['mean_y']:.3f}")
        st.write(f"**Spread x:** {stats['spread_x']:.3f}")
        st.write(f"**Spread y:** {stats['spread_y']:.3f}")
        st.write(f"**Max speed:** {stats['max_speed']:.3f}")
        st.write(f"**Boundary mode:** {boundary_mode}")
        st.write(f"**Diffusion coefficient:** {diffusion:.3f}")

    with st.expander("What to observe"):
        st.markdown(
            """
- In **uniform flow**, particles mostly translate together.
- In **shear flow**, velocity depends on vertical position, so the cloud stretches.
- In **vortex flow**, particles rotate around the center.
- In **source/sink flow**, particles move radially outward/inward.
- Nonzero **diffusion** broadens the cloud even when the flow is ordered.
- The **density heatmap** shows where tracer concentration is highest.
"""
        )

    # ---------------- Autoplay ----------------
    if st.session_state[f"{SECTION}_playing"]:
        time.sleep(0.03)
        st.rerun()