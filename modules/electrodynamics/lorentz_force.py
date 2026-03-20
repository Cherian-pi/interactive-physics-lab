import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# Visual style / unit helpers
# ============================================================
THEME = {
    "bg": "#ffffff",
    "grid": "rgba(120, 140, 160, 0.22)",
    "axis": "#31435a",
    "text": "#1f2937",
    "trajectory_glow": "rgba(59, 130, 246, 0.18)",
    "particle": "#ef4444",
    "velocity": "#2563eb",
    "force": "#f59e0b",
    "efield": "#06b6d4",
    "bfield": "#10b981",
}


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



def padded_range(values: np.ndarray, pad_frac: float = 0.15):
    vmin = float(np.min(values))
    vmax = float(np.max(values))

    if np.isclose(vmin, vmax):
        delta = max(abs(vmin) * 0.15, 1.0)
        return [vmin - delta, vmax + delta]

    span = vmax - vmin
    pad = nice_number(span * pad_frac)
    return [vmin - pad, vmax + pad]



def choose_si_scale(values, quantity: str):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    max_abs = float(np.max(np.abs(finite))) if finite.size else 0.0

    presets = {
        "time": [(1e-12, "ps"), (1e-9, "ns"), (1e-6, "µs"), (1e-3, "ms"), (1.0, "s")],
        "speed": [(1.0, "m/s"), (1e3, "km/s"), (1e6, "Mm/s")],
        "energy": [(1.602176634e-19, "eV"), (1.602176634e-16, "keV"), (1.602176634e-13, "MeV"), (1.0, "J")],
        "force": [(1e-15, "fN"), (1e-12, "pN"), (1e-9, "nN"), (1e-6, "µN"), (1.0, "N")],
        "velocity_component": [(1.0, "m/s"), (1e3, "km/s"), (1e6, "Mm/s")],
        "position": [(1e-9, "nm"), (1e-6, "µm"), (1e-3, "mm"), (1.0, "m"), (1e3, "km")],
    }

    options = presets.get(quantity, [(1.0, "")])
    chosen_scale, chosen_unit = options[0]
    for scale, unit in options:
        scaled = max_abs / scale if scale != 0 else max_abs
        if 0.1 <= scaled < 1000:
            chosen_scale, chosen_unit = scale, unit
            break
        chosen_scale, chosen_unit = scale, unit
    return chosen_scale, chosen_unit



def format_axis_title(base: str, unit: str) -> str:
    return f"{base} ({unit})" if unit else base



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


# ============================================================
# Physical constants
# ============================================================
E_CHARGE = 1.602176634e-19
E_MASS = 9.1093837015e-31
P_MASS = 1.67262192369e-27
ALPHA_MASS = 6.644657230e-27

PARTICLES = {
    "Electron": {"q": -E_CHARGE, "m": E_MASS, "color": "#00d9ff"},
    "Proton": {"q": E_CHARGE, "m": P_MASS, "color": "#ff9f1c"},
    "Alpha particle": {"q": 2 * E_CHARGE, "m": ALPHA_MASS, "color": "#ff4fd8"},
    "Custom": {"q": E_CHARGE, "m": P_MASS, "color": "#7CFC00"},
}


# ============================================================
# Session state helpers
# ============================================================
def init_lorentz_state():
    defaults = {
        "lorentz_playing": False,
        "lorentz_last_update": time.time(),
        "lorentz_sim_time": 0.0,
        "lorentz_state_vec": None,
        "lorentz_history_r": [],
        "lorentz_history_v": [],
        "lorentz_history_t": [],
        "lorentz_prev_signature": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_simulation(state0: np.ndarray):
    st.session_state.lorentz_playing = False
    st.session_state.lorentz_last_update = time.time()
    st.session_state.lorentz_sim_time = 0.0
    st.session_state.lorentz_state_vec = state0.copy()
    st.session_state.lorentz_history_r = [state0[:3].copy()]
    st.session_state.lorentz_history_v = [state0[3:].copy()]
    st.session_state.lorentz_history_t = [0.0]


def current_signature(*items):
    out = []
    for item in items:
        if isinstance(item, np.ndarray):
            out.append(tuple(np.round(item.astype(float), 12)))
        else:
            out.append(item)
    return tuple(out)


# ============================================================
# Physics
# ============================================================
def safe_unit(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n < 1e-30:
        return np.zeros_like(vec)
    return vec / n


def lorentz_acceleration(v: np.ndarray, q: float, m: float, E_vec: np.ndarray, B_vec: np.ndarray) -> np.ndarray:
    return (q / m) * (E_vec + np.cross(v, B_vec))


def rk4_step(state: np.ndarray, dt: float, q: float, m: float, E_vec: np.ndarray, B_vec: np.ndarray) -> np.ndarray:
    def deriv(s):
        r = s[:3]
        v = s[3:]
        a = lorentz_acceleration(v, q, m, E_vec, B_vec)
        return np.hstack([v, a])

    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt * k1)
    k3 = deriv(state + 0.5 * dt * k2)
    k4 = deriv(state + dt * k3)

    new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    if not np.all(np.isfinite(new_state)):
        return state.copy()

    return new_state


def choose_dt(q: float, m: float, B_vec: np.ndarray, speed_scale: float) -> float:
    """
    Internal stable timestep for integration.
    """
    Bmag = np.linalg.norm(B_vec)

    if Bmag < 1e-20:
        return 2e-9

    omega_c = abs(q) * Bmag / m
    T_c = 2.0 * np.pi / omega_c

    dt = T_c / 150.0
    dt = max(min(dt, 2e-8), 1e-12)

    if speed_scale > 5e5:
        dt *= 0.5

    return dt


def update_simulation(q: float, m: float, E_vec: np.ndarray, B_vec: np.ndarray, time_scale: float, t_max: float):
    now = time.time()
    elapsed_wall = now - st.session_state.lorentz_last_update
    st.session_state.lorentz_last_update = now

    if not st.session_state.lorentz_playing:
        return

    state = st.session_state.lorentz_state_vec.copy()
    v = state[3:]
    speed_scale = np.linalg.norm(v)

    dt_internal = choose_dt(q, m, B_vec, speed_scale)
    target_sim_advance = elapsed_wall * time_scale

    if target_sim_advance <= 0:
        return

    n_substeps = max(1, int(np.ceil(target_sim_advance / dt_internal)))
    dt = target_sim_advance / n_substeps

    for _ in range(n_substeps):
        if st.session_state.lorentz_sim_time >= t_max:
            st.session_state.lorentz_playing = False
            break

        state = rk4_step(state, dt, q, m, E_vec, B_vec)
        st.session_state.lorentz_sim_time += dt
        st.session_state.lorentz_history_r.append(state[:3].copy())
        st.session_state.lorentz_history_v.append(state[3:].copy())
        st.session_state.lorentz_history_t.append(st.session_state.lorentz_sim_time)

    st.session_state.lorentz_state_vec = state.copy()

    max_history = 5000
    if len(st.session_state.lorentz_history_t) > max_history:
        st.session_state.lorentz_history_r = st.session_state.lorentz_history_r[-max_history:]
        st.session_state.lorentz_history_v = st.session_state.lorentz_history_v[-max_history:]
        st.session_state.lorentz_history_t = st.session_state.lorentz_history_t[-max_history:]


# ============================================================
# Analysis helpers
# ============================================================
def classify_motion(E_vec: np.ndarray, B_vec: np.ndarray) -> str:
    E_mag = np.linalg.norm(E_vec)
    B_mag = np.linalg.norm(B_vec)

    if E_mag < 1e-20 and B_mag < 1e-20:
        return "Uniform motion"
    if B_mag < 1e-20:
        return "Electric acceleration"
    if E_mag < 1e-20:
        return "Magnetic bending / circular motion"
    return "Combined electric–magnetic motion"


# ============================================================
# Plot helpers
# ============================================================
def make_2d_arrow(start_xy, vec_xy, scale, color, name):
    end = start_xy + vec_xy * scale
    return go.Scatter(
        x=[start_xy[0], end[0]],
        y=[start_xy[1], end[1]],
        mode="lines+markers",
        line=dict(color=color, width=5),
        marker=dict(size=[0, 9], color=color, symbol=["circle", "arrow-bar-up"]),
        name=name,
        showlegend=True,
    )


def render_xy_scene(history_r, current_r, current_v, current_f, particle_color, B_vec):
    hist = np.array(history_r)
    x = hist[:, 0]
    y = hist[:, 1]

    pos_scale, pos_unit = choose_si_scale(np.concatenate([x, y]), "position")
    x_scaled = x / pos_scale
    y_scaled = y / pos_scale
    x_now = current_r[0] / pos_scale
    y_now = current_r[1] / pos_scale

    x_range = padded_range(x_scaled, pad_frac=0.18)
    y_range = padded_range(y_scaled, pad_frac=0.18)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_scaled,
            y=y_scaled,
            mode="lines",
            line=dict(color=THEME["trajectory_glow"], width=10, shape="spline", smoothing=0.85),
            hoverinfo="skip",
            showlegend=False,
            name="Trajectory glow",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_scaled,
            y=y_scaled,
            mode="lines",
            line=dict(color=particle_color, width=4, shape="spline", smoothing=0.75),
            name="Trajectory",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[x_now],
            y=[y_now],
            mode="markers",
            marker=dict(size=14, color=THEME["particle"], line=dict(color="white", width=1.5)),
            name="Particle",
        )
    )

    scale_base = 0.12 * max(x_range[1] - x_range[0], y_range[1] - y_range[0], 1e-4)

    v_hat = safe_unit(current_v[:2])
    f_hat = safe_unit(current_f[:2])

    if np.linalg.norm(v_hat) > 0:
        fig.add_trace(make_2d_arrow(np.array([x_now, y_now]), v_hat, scale_base, THEME["velocity"], "Velocity"))

    if np.linalg.norm(f_hat) > 0:
        fig.add_trace(make_2d_arrow(np.array([x_now, y_now]), f_hat, scale_base, THEME["force"], "Force"))

    Bz = B_vec[2]
    if abs(Bz) > 0:
        b_text = f"Magnetic field: Bz = {Bz:.2e} T"
        b_symbol = "⊙ out of plane" if Bz > 0 else "⊗ into plane"
    else:
        b_text = "Magnetic field: Bz = 0"
        b_symbol = ""

    fig.update_layout(
        title=dict(text="Real-Time x–y Motion", x=0.5, xanchor="center"),
        xaxis=common_axis_style(format_axis_title("x", pos_unit), x_range),
        yaxis={**common_axis_style(format_axis_title("y", pos_unit), y_range), "scaleanchor": "x", "scaleratio": 1},
        annotations=[
            dict(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=f"{b_text}<br>{b_symbol}",
                showarrow=False,
                align="left",
                font=dict(size=12, color=THEME["text"]),
                bgcolor="rgba(255,255,255,0.82)",
                bordercolor="rgba(120,140,160,0.25)",
                borderwidth=1,
            )
        ],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        template="plotly_white",
        plot_bgcolor=THEME["bg"],
        paper_bgcolor=THEME["bg"],
        font=dict(size=14, color=THEME["text"]),
        height=560,
        margin=dict(l=70, r=30, t=70, b=55),
    )
    return fig


def add_3d_arrow(fig, origin, vec, scale, color, name):
    vec_hat = safe_unit(vec)
    if np.linalg.norm(vec_hat) == 0:
        return

    end = origin + vec_hat * scale

    fig.add_trace(
        go.Scatter3d(
            x=[origin[0], end[0]],
            y=[origin[1], end[1]],
            z=[origin[2], end[2]],
            mode="lines+markers",
            line=dict(color=color, width=8),
            marker=dict(size=[0, 5], color=color),
            name=name,
            showlegend=True,
        )
    )


def render_3d_scene(history_r, current_r, current_v, current_f, particle_color, E_vec, B_vec):
    hist = np.array(history_r)
    x = hist[:, 0]
    y = hist[:, 1]
    z = hist[:, 2]

    pos_scale, pos_unit = choose_si_scale(np.concatenate([x, y, z]), "position")
    x_scaled = x / pos_scale
    y_scaled = y / pos_scale
    z_scaled = z / pos_scale
    current_scaled = current_r / pos_scale

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=x_scaled,
            y=y_scaled,
            z=z_scaled,
            mode="lines",
            line=dict(color=particle_color, width=7),
            name="Trajectory",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[current_scaled[0]],
            y=[current_scaled[1]],
            z=[current_scaled[2]],
            mode="markers",
            marker=dict(size=6.5, color=THEME["particle"]),
            name="Particle",
        )
    )

    extent = max(
        np.ptp(x_scaled) if len(x_scaled) > 1 else 1e-4,
        np.ptp(y_scaled) if len(y_scaled) > 1 else 1e-4,
        np.ptp(z_scaled) if len(z_scaled) > 1 else 1e-4,
        1e-4,
    )
    arrow_scale = 0.2 * extent
    origin = np.array(current_scaled, dtype=float)

    add_3d_arrow(fig, origin, current_v, arrow_scale, THEME["velocity"], "Velocity")
    add_3d_arrow(fig, origin, current_f, arrow_scale, THEME["force"], "Force")
    add_3d_arrow(fig, origin, E_vec, arrow_scale, THEME["efield"], "E field")
    add_3d_arrow(fig, origin, B_vec, arrow_scale, THEME["bfield"], "B field")

    axis_title = lambda label: format_axis_title(label, pos_unit)
    fig.update_layout(
        title=dict(text="Real-Time 3D Motion", x=0.5, xanchor="center"),
        scene=dict(
            xaxis=dict(title=axis_title("x"), showgrid=True, gridcolor=THEME["grid"], backgroundcolor="white"),
            yaxis=dict(title=axis_title("y"), showgrid=True, gridcolor=THEME["grid"], backgroundcolor="white"),
            zaxis=dict(title=axis_title("z"), showgrid=True, gridcolor=THEME["grid"], backgroundcolor="white"),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.45, y=1.45, z=0.95)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        template="plotly_white",
        paper_bgcolor=THEME["bg"],
        font=dict(size=14, color=THEME["text"]),
        height=560,
        margin=dict(l=0, r=0, t=70, b=0),
    )
    return fig


def line_plot(x, y, title, y_label, quantity_type, current_x=None):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    time_scale, time_unit = choose_si_scale(x_arr, "time")
    value_scale, value_unit = choose_si_scale(y_arr, quantity_type)

    x_scaled = x_arr / time_scale if time_scale != 0 else x_arr
    y_scaled = y_arr / value_scale if value_scale != 0 else y_arr
    current_x_scaled = current_x / time_scale if current_x is not None and time_scale != 0 else current_x

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_scaled,
            y=y_scaled,
            mode="lines",
            line=dict(width=3.5, color=THEME["velocity"], shape="spline", smoothing=0.65),
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.08)",
        )
    )

    if current_x_scaled is not None:
        fig.add_vline(x=current_x_scaled, line_width=2, line_dash="dash", line_color=THEME["particle"])

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis=common_axis_style(format_axis_title("Time", time_unit), padded_range(x_scaled, pad_frac=0.08)),
        yaxis=common_axis_style(format_axis_title(y_label, value_unit), padded_range(y_scaled, pad_frac=0.15)),
        template="plotly_white",
        plot_bgcolor=THEME["bg"],
        paper_bgcolor=THEME["bg"],
        height=320,
        margin=dict(l=70, r=25, t=55, b=55),
        font=dict(size=14, color=THEME["text"]),
        showlegend=False,
    )
    return fig


# ============================================================
# Main app
# ============================================================
def run():
    init_lorentz_state()

    st.title("Lorentz Force Explorer")

    st.markdown("### Governing equation")
    st.latex(r"\mathbf{F}=q\left(\mathbf{E}+\mathbf{v}\times\mathbf{B}\right)")

    st.markdown("### Coupled equations of motion")
    st.latex(r"\frac{d\mathbf{r}}{dt}=\mathbf{v}")
    st.latex(r"\frac{d\mathbf{v}}{dt}=\frac{q}{m}\left(\mathbf{E}+\mathbf{v}\times\mathbf{B}\right)")

    st.markdown(
        """
This simulation is shown in **real time from** \(t=0\):
- the particle moves continuously,
- the **blue arrow** is the velocity,
- the **yellow arrow** is the instantaneous force,
- both the **2D** and **3D** views use the same evolving state.
"""
    )

    # ---------------- Sidebar ----------------
    st.sidebar.header("Particle")

    particle_name = st.sidebar.selectbox(
        "Choose particle",
        list(PARTICLES.keys()),
        key="lorentz_particle_type",
    )

    default_q = PARTICLES[particle_name]["q"]
    default_m = PARTICLES[particle_name]["m"]
    particle_color = PARTICLES[particle_name]["color"]

    if particle_name == "Custom":
        q_multiplier = st.sidebar.number_input(
            "Charge q / e",
            min_value=-10.0,
            max_value=10.0,
            value=1.0,
            step=0.5,
            key="lorentz_custom_q",
        )
        m_multiplier = st.sidebar.number_input(
            "Mass m / mₚ",
            min_value=0.001,
            max_value=100.0,
            value=1.0,
            step=0.1,
            key="lorentz_custom_m",
        )
        q = q_multiplier * E_CHARGE
        m = m_multiplier * P_MASS
    else:
        q = default_q
        m = default_m

    st.sidebar.markdown("---")
    st.sidebar.header("Preset")

    preset = st.sidebar.selectbox(
        "Quick preset",
        [
            "Pure magnetic field",
            "Pure electric field",
            "Crossed E and B fields",
            "Helical start",
            "Custom",
        ],
        key="lorentz_preset",
    )

    Ex0, Ey0, Ez0 = 0.0, 0.0, 0.0
    Bx0, By0, Bz0 = 0.0, 0.0, 1e-3
    vx0, vy0, vz0 = 1e5, 0.0, 0.0

    if preset == "Pure electric field":
        Ex0, Ey0, Ez0 = 80.0, 0.0, 0.0
        Bx0, By0, Bz0 = 0.0, 0.0, 0.0
        vx0, vy0, vz0 = 0.0, 0.0, 0.0
    elif preset == "Crossed E and B fields":
        Ex0, Ey0, Ez0 = 80.0, 0.0, 0.0
        Bx0, By0, Bz0 = 0.0, 0.0, 1e-3
        vx0, vy0, vz0 = 8e4, 0.0, 0.0
    elif preset == "Helical start":
        Ex0, Ey0, Ez0 = 0.0, 0.0, 0.0
        Bx0, By0, Bz0 = 0.0, 0.0, 1e-3
        vx0, vy0, vz0 = 8e4, 0.0, 5e4

    st.sidebar.markdown("---")
    st.sidebar.header("Electric field")
    Ex = st.sidebar.number_input("Eₓ (V/m)", value=float(Ex0), format="%.3e", key="lorentz_ex")
    Ey = st.sidebar.number_input("Eᵧ (V/m)", value=float(Ey0), format="%.3e", key="lorentz_ey")
    Ez = st.sidebar.number_input("E along z, Ez (V/m)", value=float(Ez0), format="%.3e", key="lorentz_ez")

    st.sidebar.markdown("---")
    st.sidebar.header("Magnetic field")
    Bx = st.sidebar.number_input("Bₓ (T)", value=float(Bx0), format="%.3e", key="lorentz_bx")
    By = st.sidebar.number_input("Bᵧ (T)", value=float(By0), format="%.3e", key="lorentz_by")
    Bz = st.sidebar.number_input("B along z, Bz (T)", value=float(Bz0), format="%.3e", key="lorentz_bz")

    st.sidebar.markdown("---")
    st.sidebar.header("Initial position")
    x0 = st.sidebar.number_input("x₀ (m)", value=0.0, format="%.3e", key="lorentz_x0")
    y0 = st.sidebar.number_input("y₀ (m)", value=0.0, format="%.3e", key="lorentz_y0")
    z0 = st.sidebar.number_input("z₀ (m)", value=0.0, format="%.3e", key="lorentz_z0")

    st.sidebar.markdown("---")
    st.sidebar.header("Initial velocity")
    vx0_user = st.sidebar.number_input("vₓ(0) (m/s)", value=float(vx0), format="%.3e", key="lorentz_vx0")
    vy0_user = st.sidebar.number_input("vᵧ(0) (m/s)", value=float(vy0), format="%.3e", key="lorentz_vy0")
    vz0_user = st.sidebar.number_input("vz(0) (m/s)", value=float(vz0), format="%.3e", key="lorentz_vz0")

    st.sidebar.markdown("---")
    st.sidebar.header("Animation")
    t_max = st.sidebar.slider(
        "Simulation time window (µs)",
        min_value=5,
        max_value=200,
        value=40,
        step=5,
        key="lorentz_tmax_us",
    ) * 1e-6

    time_scale = st.sidebar.slider(
        "Playback speed",
        min_value=0.2,
        max_value=20.0,
        value=3.0,
        step=0.2,
        key="lorentz_time_scale",
    ) * 1e-6

    trail_points = st.sidebar.slider(
        "Visible trail length",
        min_value=50,
        max_value=3000,
        value=800,
        step=50,
        key="lorentz_trail_points",
    )

    clear_trail = st.sidebar.checkbox(
        "Clear trail on reset",
        value=True,
        key="lorentz_clear_trail",
    )

    # ---------------- Inputs ----------------
    E_vec = np.array([Ex, Ey, Ez], dtype=float)
    B_vec = np.array([Bx, By, Bz], dtype=float)
    state0 = np.array([x0, y0, z0, vx0_user, vy0_user, vz0_user], dtype=float)

    signature = current_signature(particle_name, q, m, E_vec, B_vec, state0, t_max)
    if st.session_state.lorentz_prev_signature != signature:
        st.session_state.lorentz_prev_signature = signature
        reset_simulation(state0)

    # ---------------- Controls ----------------
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        if st.button("▶ Play", key="lorentz_play_btn", width="stretch"):
            if st.session_state.lorentz_sim_time >= t_max:
                reset_simulation(state0)
            st.session_state.lorentz_playing = True
            st.session_state.lorentz_last_update = time.time()
            st.rerun()

    with c2:
        if st.button("⏸ Pause", key="lorentz_pause_btn", width="stretch"):
            st.session_state.lorentz_playing = False
            st.rerun()

    with c3:
        if st.button("⟲ Reset", key="lorentz_reset_btn", width="stretch"):
            if clear_trail:
                reset_simulation(state0)
            else:
                st.session_state.lorentz_state_vec = state0.copy()
                st.session_state.lorentz_sim_time = 0.0
                st.session_state.lorentz_playing = False
                st.session_state.lorentz_last_update = time.time()
            st.rerun()

    progress = min(st.session_state.lorentz_sim_time / t_max, 1.0) if t_max > 0 else 0.0
    st.progress(progress)
    st.caption(f"Simulation progress: {100 * progress:.1f}%")

    # ---------------- Update state ----------------
    update_simulation(q=q, m=m, E_vec=E_vec, B_vec=B_vec, time_scale=time_scale, t_max=t_max)

    state = st.session_state.lorentz_state_vec.copy()
    r = state[:3]
    v = state[3:]
    F = q * (E_vec + np.cross(v, B_vec))

    history_r = st.session_state.lorentz_history_r[-trail_points:]
    history_v = st.session_state.lorentz_history_v[-trail_points:]
    history_t = st.session_state.lorentz_history_t[-trail_points:]

    history_v_arr = np.array(history_v)
    speeds = np.linalg.norm(history_v_arr, axis=1)
    kinetic_energy = 0.5 * m * speeds**2
    force_hist = q * (E_vec + np.cross(history_v_arr, B_vec))
    force_mag = np.linalg.norm(force_hist, axis=1)

    # ---------------- Metrics ----------------
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Charge q (C)", f"{q:.3e}")
    m2.metric("Mass m (kg)", f"{m:.3e}")
    m3.metric("Current speed (m/s)", f"{np.linalg.norm(v):.3e}")
    m4.metric("Motion type", classify_motion(E_vec, B_vec))

    st.info(
        "Watch the yellow force arrow. In a pure magnetic field it stays perpendicular "
        "to the velocity, so the particle bends rather than simply speeding up."
    )

    # ---------------- Views ----------------
    view2d, view3d = st.tabs(["2D real-time view", "3D real-time view"])

    with view2d:
        fig_xy = render_xy_scene(
            history_r=history_r,
            current_r=r,
            current_v=v,
            current_f=F,
            particle_color=particle_color,
            B_vec=B_vec,
        )
        st.plotly_chart(fig_xy, width="stretch")

    with view3d:
        fig_3d = render_3d_scene(
            history_r=history_r,
            current_r=r,
            current_v=v,
            current_f=F,
            particle_color=particle_color,
            E_vec=E_vec,
            B_vec=B_vec,
        )
        st.plotly_chart(fig_3d, width="stretch")

    # ---------------- Current state + graphs ----------------
    left, right = st.columns([1.1, 1.0])

    with left:
        row1, row2 = st.columns(2)
        with row1:
            st.plotly_chart(
                line_plot(history_t, speeds, "Speed vs time", "Speed", "speed", st.session_state.lorentz_sim_time),
                width="stretch",
            )
        with row2:
            st.plotly_chart(
                line_plot(history_t, kinetic_energy, "Kinetic energy vs time", "Energy", "energy", st.session_state.lorentz_sim_time),
                width="stretch",
            )

        row3, row4 = st.columns(2)
        with row3:
            vx_hist = history_v_arr[:, 0]
            st.plotly_chart(
                line_plot(history_t, vx_hist, "vₓ vs time", "vₓ", "velocity_component", st.session_state.lorentz_sim_time),
                width="stretch",
            )
        with row4:
            st.plotly_chart(
                line_plot(history_t, force_mag, "|F| vs time", "|F|", "force", st.session_state.lorentz_sim_time),
                width="stretch",
            )

    with right:
        st.subheader("Current state")
        time_scale_disp, time_unit_disp = choose_si_scale([st.session_state.lorentz_sim_time], "time")
        pos_scale_disp, pos_unit_disp = choose_si_scale(r, "position")
        vel_scale_disp, vel_unit_disp = choose_si_scale(v, "velocity_component")
        force_scale_disp, force_unit_disp = choose_si_scale(F, "force")

        st.write(f"**Time:** {st.session_state.lorentz_sim_time / time_scale_disp:.3f} {time_unit_disp}")
        st.write(f"**Position:** ({r[0] / pos_scale_disp:.3f}, {r[1] / pos_scale_disp:.3f}, {r[2] / pos_scale_disp:.3f}) {pos_unit_disp}")
        st.write(f"**Velocity:** ({v[0] / vel_scale_disp:.3f}, {v[1] / vel_scale_disp:.3f}, {v[2] / vel_scale_disp:.3f}) {vel_unit_disp}")
        st.write(f"**Speed:** {np.linalg.norm(v) / vel_scale_disp:.3f} {vel_unit_disp}")
        st.write(f"**Force:** ({F[0] / force_scale_disp:.3f}, {F[1] / force_scale_disp:.3f}, {F[2] / force_scale_disp:.3f}) {force_unit_disp}")
        st.write(f"**|Force|:** {np.linalg.norm(F) / force_scale_disp:.3f} {force_unit_disp}")

        Bmag = np.linalg.norm(B_vec)
        if Bmag > 1e-20 and abs(q) > 0:
            B_hat = safe_unit(B_vec)
            v_par = np.dot(v, B_hat) * B_hat
            v_perp = np.linalg.norm(v - v_par)
            rg = m * v_perp / (abs(q) * Bmag)
            omega_c = abs(q) * Bmag / m
            st.write(f"**Gyroradius:** {rg:.3e} m")
            st.write(f"**Cyclotron angular frequency:** {omega_c:.3e} rad/s")

    with st.expander("What to observe"):
        st.markdown(
            """
- In a **pure magnetic field**, the force is perpendicular to the velocity, so the path bends.
- In a **pure electric field**, the speed changes in the electric-field direction.
- In combined fields, the path can curve, drift, or spiral.
- A nonzero initial **vz(0)** creates visible 3D motion.
"""
        )

    # ---------------- Autoplay ----------------
    if st.session_state.lorentz_playing:
        time.sleep(0.03)
        st.rerun()