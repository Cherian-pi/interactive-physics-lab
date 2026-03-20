import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# Visual style / helpers
# ============================================================
THEME = {
    "bg": "#ffffff",
    "grid": "rgba(120, 140, 160, 0.22)",
    "axis": "#31435a",
    "text": "#1f2937",
    "trajectory_glow": "rgba(59, 130, 246, 0.18)",
    "planet": "#3b82f6",
    "star": "#f59e0b",
    "velocity": "#2563eb",
    "force": "#ef4444",
    "angular": "#10b981",
    "bound": "#ef4444",
    "marginal": "#f59e0b",
    "escape": "#10b981",
    "periapsis": "#10b981",
    "apoapsis": "#f59e0b",
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
        "time": [(1.0, "s"), (60.0, "min"), (3600.0, "hr"), (86400.0, "day"), (31557600.0, "yr")],
        "distance": [(1.0, "m"), (1e3, "km"), (1.496e11, "AU")],
        "speed": [(1.0, "m/s"), (1e3, "km/s")],
        "energy": [(1.0, "J"), (1e9, "GJ"), (1e12, "TJ")],
        "force": [(1.0, "N"), (1e3, "kN"), (1e6, "MN")],
        "acceleration": [(1.0, "m/s²"), (1e-3, "mm/s²")],
        "angmom": [(1.0, "kg·m²/s"), (1e9, "10⁹ kg·m²/s"), (1e15, "10¹⁵ kg·m²/s")],
        "ratio": [(1.0, "")],
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


def safe_unit(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n < 1e-30:
        return np.zeros_like(vec)
    return vec / n


# ============================================================
# Constants / presets
# ============================================================
G = 6.67430e-11

CENTRAL_BODIES = {
    "Earth": {
        "mass": 5.972e24,
        "radius": 6.371e6,
        "color": "#78a6f0",
    },
    "Sun": {
        "mass": 1.989e30,
        "radius": 6.9634e8,
        "color": "#f59e0b",
    },
    "Jupiter": {
        "mass": 1.898e27,
        "radius": 6.9911e7,
        "color": "#f97316",
    },
    "Moon": {
        "mass": 7.34767309e22,
        "radius": 1.7374e6,
        "color": "#94a3b8",
    },
    "Custom": {
        "mass": 5.972e24,
        "radius": 6.371e6,
        "color": "#a855f7",
    },
}


# ============================================================
# Session state helpers
# ============================================================
def init_orbital_state():
    defaults = {
        "orbital_playing": False,
        "orbital_last_update": time.time(),
        "orbital_sim_time": 0.0,
        "orbital_state_vec": None,
        "orbital_history_r": [],
        "orbital_history_v": [],
        "orbital_history_t": [],
        "orbital_prev_signature": None,
        "orbital_collision": False,
        "orbital_escape": False,
        "orbital_status": "Bound",
        "orbital_energy_positive_seen": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_simulation(state0: np.ndarray):
    st.session_state.orbital_playing = False
    st.session_state.orbital_last_update = time.time()
    st.session_state.orbital_sim_time = 0.0
    st.session_state.orbital_state_vec = state0.copy()
    st.session_state.orbital_history_r = [state0[:3].copy()]
    st.session_state.orbital_history_v = [state0[3:].copy()]
    st.session_state.orbital_history_t = [0.0]
    st.session_state.orbital_collision = False
    st.session_state.orbital_escape = False
    st.session_state.orbital_status = "Bound"
    st.session_state.orbital_energy_positive_seen = False


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
def gravitational_acceleration(r: np.ndarray, mu: float) -> np.ndarray:
    rmag = np.linalg.norm(r)
    if rmag < 1e-9:
        return np.zeros(3)
    return -(mu / rmag**3) * r


def gravitational_force(r: np.ndarray, m_obj: float, mu: float) -> np.ndarray:
    return m_obj * gravitational_acceleration(r, mu)


def rk4_step(state: np.ndarray, dt: float, mu: float) -> np.ndarray:
    def deriv(s):
        r = s[:3]
        v = s[3:]
        a = gravitational_acceleration(r, mu)
        return np.hstack([v, a])

    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt * k1)
    k3 = deriv(state + 0.5 * dt * k2)
    k4 = deriv(state + dt * k3)

    new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    if not np.all(np.isfinite(new_state)):
        return state.copy()
    return new_state


def orbital_quantities(r: np.ndarray, v: np.ndarray, mu: float):
    rmag = np.linalg.norm(r)
    vmag = np.linalg.norm(v)

    if rmag < 1e-12:
        return {
            "r": rmag,
            "v": vmag,
            "energy_specific": np.nan,
            "angmom_vec": np.zeros(3),
            "angmom_mag": 0.0,
            "ecc_vec": np.zeros(3),
            "ecc": np.nan,
            "a": np.nan,
            "escape_speed": np.nan,
            "circular_speed": np.nan,
            "speed_ratio_escape": np.nan,
            "vis_viva_speed": np.nan,
            "periapsis": np.nan,
            "apoapsis": np.nan,
            "true_anomaly_cos": np.nan,
        }

    h_vec = np.cross(r, v)
    h_mag = np.linalg.norm(h_vec)

    eps = 0.5 * vmag**2 - mu / rmag
    ecc_vec = np.cross(v, h_vec) / mu - r / rmag
    ecc = np.linalg.norm(ecc_vec)

    if np.isclose(eps, 0.0, atol=1e-12):
        a = np.inf
    else:
        a = -mu / (2.0 * eps)

    v_esc = np.sqrt(2 * mu / rmag)
    v_circ = np.sqrt(mu / rmag)

    if np.isfinite(a) and a > 0:
        vis_viva_speed = np.sqrt(max(mu * (2.0 / rmag - 1.0 / a), 0.0))
    else:
        vis_viva_speed = np.nan

    if np.isfinite(a) and np.isfinite(ecc) and ecc < 1.0:
        periapsis = a * (1.0 - ecc)
        apoapsis = a * (1.0 + ecc)
    else:
        periapsis = np.nan
        apoapsis = np.nan

    if ecc > 1e-12:
        true_anomaly_cos = np.dot(ecc_vec, r) / (ecc * rmag)
        true_anomaly_cos = np.clip(true_anomaly_cos, -1.0, 1.0)
    else:
        true_anomaly_cos = np.nan

    return {
        "r": rmag,
        "v": vmag,
        "energy_specific": eps,
        "angmom_vec": h_vec,
        "angmom_mag": h_mag,
        "ecc_vec": ecc_vec,
        "ecc": ecc,
        "a": a,
        "escape_speed": v_esc,
        "circular_speed": v_circ,
        "speed_ratio_escape": vmag / v_esc if v_esc > 0 else np.nan,
        "vis_viva_speed": vis_viva_speed,
        "periapsis": periapsis,
        "apoapsis": apoapsis,
        "true_anomaly_cos": true_anomaly_cos,
    }


def choose_dt(mu: float, r: np.ndarray, v: np.ndarray) -> float:
    rmag = max(np.linalg.norm(r), 1.0)
    vmag = max(np.linalg.norm(v), 1.0)

    dt_orbit = 0.01 * np.sqrt(rmag**3 / mu)
    dt_cross = 0.01 * rmag / vmag

    dt = min(dt_orbit, dt_cross)
    dt = max(min(dt, 3600.0), 0.01)
    return dt


def classify_energy_state(energy_specific: float, tol: float = 1e-4) -> str:
    if not np.isfinite(energy_specific):
        return "Undefined"
    threshold = tol * max(1.0, abs(energy_specific))
    if abs(energy_specific) <= threshold:
        return "Marginal escape"
    if energy_specific > 0:
        return "Escaping"
    return "Bound"


def classify_orbit(ecc: float, energy_specific: float, collided: bool, escaped: bool) -> str:
    if collided:
        return "Collision"
    if escaped:
        return "Escaping trajectory"
    if not np.isfinite(ecc) or not np.isfinite(energy_specific):
        return "Undefined"
    if np.isclose(ecc, 0.0, atol=5e-3):
        return "Nearly circular orbit"
    if ecc < 1.0 and energy_specific < 0:
        return "Elliptical orbit"
    if np.isclose(ecc, 1.0, atol=2e-2):
        return "Near-parabolic trajectory"
    if ecc > 1.0 and energy_specific > 0:
        return "Hyperbolic trajectory"
    return "Bound / transitional motion"


def update_simulation(mu: float, R_body: float, playback_speed: float, t_max: float):
    now = time.time()
    elapsed_wall = now - st.session_state.orbital_last_update
    st.session_state.orbital_last_update = now

    if not st.session_state.orbital_playing:
        return

    state = st.session_state.orbital_state_vec.copy()
    r = state[:3]
    v = state[3:]

    target_sim_advance = elapsed_wall * playback_speed
    if target_sim_advance <= 0:
        return

    dt_internal = choose_dt(mu, r, v)
    n_substeps = max(1, int(np.ceil(target_sim_advance / dt_internal)))
    dt = target_sim_advance / n_substeps

    for _ in range(n_substeps):
        if st.session_state.orbital_sim_time >= t_max:
            st.session_state.orbital_playing = False
            break

        state = rk4_step(state, dt, mu)
        r = state[:3]
        v = state[3:]
        rmag = np.linalg.norm(r)

        q_now = orbital_quantities(r, v, mu)
        if np.isfinite(q_now["energy_specific"]) and q_now["energy_specific"] > 0:
            st.session_state.orbital_energy_positive_seen = True

        st.session_state.orbital_sim_time += dt
        st.session_state.orbital_history_r.append(r.copy())
        st.session_state.orbital_history_v.append(v.copy())
        st.session_state.orbital_history_t.append(st.session_state.orbital_sim_time)

        if rmag <= R_body:
            st.session_state.orbital_collision = True
            st.session_state.orbital_playing = False
            st.session_state.orbital_status = "Collision"
            break

        if rmag >= 80 * R_body and q_now["energy_specific"] > 0:
            st.session_state.orbital_escape = True
            st.session_state.orbital_playing = False
            st.session_state.orbital_status = "Escaping"
            break

    st.session_state.orbital_state_vec = state.copy()

    q_final = orbital_quantities(state[:3], state[3:], mu)
    if not st.session_state.orbital_collision:
        st.session_state.orbital_status = classify_energy_state(q_final["energy_specific"])

    max_history = 8000
    if len(st.session_state.orbital_history_t) > max_history:
        st.session_state.orbital_history_r = st.session_state.orbital_history_r[-max_history:]
        st.session_state.orbital_history_v = st.session_state.orbital_history_v[-max_history:]
        st.session_state.orbital_history_t = st.session_state.orbital_history_t[-max_history:]


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


def render_xy_scene(history_r, current_r, current_v, current_f, body_radius, body_color, ecc_vec=None, periapsis=np.nan, apoapsis=np.nan):
    hist = np.array(history_r)
    x = hist[:, 0]
    y = hist[:, 1]

    pos_scale, pos_unit = choose_si_scale(np.concatenate([x, y, current_r[:2]]), "distance")
    x_scaled = x / pos_scale
    y_scaled = y / pos_scale
    x_now = current_r[0] / pos_scale
    y_now = current_r[1] / pos_scale
    body_radius_scaled = body_radius / pos_scale

    x_range = padded_range(np.append(x_scaled, 0.0), pad_frac=0.18)
    y_range = padded_range(np.append(y_scaled, 0.0), pad_frac=0.18)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_scaled,
            y=y_scaled,
            mode="lines",
            line=dict(color=THEME["trajectory_glow"], width=10, shape="spline", smoothing=0.85),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_scaled,
            y=y_scaled,
            mode="lines",
            line=dict(color=THEME["planet"], width=4, shape="spline", smoothing=0.75),
            name="Trajectory",
        )
    )

    theta = np.linspace(0, 2 * np.pi, 240)
    fig.add_trace(
        go.Scatter(
            x=body_radius_scaled * np.cos(theta),
            y=body_radius_scaled * np.sin(theta),
            mode="lines",
            fill="toself",
            line=dict(color=body_color, width=2),
            fillcolor=body_color,
            name="Central body",
        )
    )

    if ecc_vec is not None and np.isfinite(periapsis) and np.isfinite(apoapsis) and np.linalg.norm(ecc_vec) > 1e-8:
        ehat = safe_unit(ecc_vec)
        rp = ehat * periapsis
        ra = -ehat * apoapsis

        fig.add_trace(
            go.Scatter(
                x=[rp[0] / pos_scale],
                y=[rp[1] / pos_scale],
                mode="markers+text",
                marker=dict(size=10, color=THEME["periapsis"]),
                text=["Periapsis"],
                textposition="top center",
                name="Periapsis",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[ra[0] / pos_scale],
                y=[ra[1] / pos_scale],
                mode="markers+text",
                marker=dict(size=10, color=THEME["apoapsis"]),
                text=["Apoapsis"],
                textposition="top center",
                name="Apoapsis",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[x_now],
            y=[y_now],
            mode="markers",
            marker=dict(size=12, color=THEME["force"], line=dict(color="white", width=1.5)),
            name="Object",
        )
    )

    scale_base = 0.12 * max(x_range[1] - x_range[0], y_range[1] - y_range[0], 1e-4)

    v_hat = safe_unit(current_v[:2])
    f_hat = safe_unit(current_f[:2])

    if np.linalg.norm(v_hat) > 0:
        fig.add_trace(make_2d_arrow(np.array([x_now, y_now]), v_hat, scale_base, THEME["velocity"], "Velocity"))

    if np.linalg.norm(f_hat) > 0:
        fig.add_trace(make_2d_arrow(np.array([x_now, y_now]), f_hat, scale_base, THEME["force"], "Gravity"))

    fig.update_layout(
        title=dict(text="Real-Time x–y Motion", x=0.5, xanchor="center"),
        xaxis=common_axis_style(format_axis_title("x", pos_unit), x_range),
        yaxis={**common_axis_style(format_axis_title("y", pos_unit), y_range), "scaleanchor": "x", "scaleratio": 1},
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


def render_3d_scene(history_r, current_r, current_v, current_f, body_radius, body_color):
    hist = np.array(history_r)
    x = hist[:, 0]
    y = hist[:, 1]
    z = hist[:, 2]

    pos_scale, pos_unit = choose_si_scale(np.concatenate([x, y, z, current_r]), "distance")
    x_scaled = x / pos_scale
    y_scaled = y / pos_scale
    z_scaled = z / pos_scale
    current_scaled = current_r / pos_scale
    body_radius_scaled = body_radius / pos_scale

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=x_scaled,
            y=y_scaled,
            z=z_scaled,
            mode="lines",
            line=dict(color=THEME["planet"], width=7),
            name="Trajectory",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[current_scaled[0]],
            y=[current_scaled[1]],
            z=[current_scaled[2]],
            mode="markers",
            marker=dict(size=5.5, color=THEME["force"]),
            name="Object",
        )
    )

    # True sphere with equal scaling
    u = np.linspace(0, 2 * np.pi, 60)
    vv = np.linspace(0, np.pi, 40)

    xs = body_radius_scaled * np.outer(np.cos(u), np.sin(vv))
    ys = body_radius_scaled * np.outer(np.sin(u), np.sin(vv))
    zs = body_radius_scaled * np.outer(np.ones_like(u), np.cos(vv))

    fig.add_trace(
        go.Surface(
            x=xs,
            y=ys,
            z=zs,
            surfacecolor=np.ones_like(xs),
            colorscale=[[0, body_color], [1, body_color]],
            cmin=0,
            cmax=1,
            showscale=False,
            opacity=0.95,
            lighting=dict(ambient=0.7, diffuse=0.9, specular=0.25, roughness=0.7),
            lightposition=dict(x=100, y=100, z=200),
            name="Central body",
            hoverinfo="skip",
        )
    )

    all_points = np.vstack([hist, current_r.reshape(1, 3)])
    max_extent = np.max(np.abs(all_points / pos_scale))
    max_extent = max(max_extent, 2.5 * body_radius_scaled, 1e-6)
    axis_limit = 1.15 * max_extent

    arrow_scale = 0.18 * axis_limit
    origin = np.array(current_scaled, dtype=float)

    add_3d_arrow(fig, origin, current_v, arrow_scale, THEME["velocity"], "Velocity")
    add_3d_arrow(fig, origin, current_f, arrow_scale, THEME["force"], "Gravity")

    axis_title = lambda label: format_axis_title(label, pos_unit)
    fig.update_layout(
        title=dict(text="Real-Time 3D Motion", x=0.5, xanchor="center"),
        scene=dict(
            xaxis=dict(
                title=axis_title("x"),
                range=[-axis_limit, axis_limit],
                showgrid=True,
                gridcolor=THEME["grid"],
                backgroundcolor="white",
                zeroline=True,
            ),
            yaxis=dict(
                title=axis_title("y"),
                range=[-axis_limit, axis_limit],
                showgrid=True,
                gridcolor=THEME["grid"],
                backgroundcolor="white",
                zeroline=True,
            ),
            zaxis=dict(
                title=axis_title("z"),
                range=[-axis_limit, axis_limit],
                showgrid=True,
                gridcolor=THEME["grid"],
                backgroundcolor="white",
                zeroline=True,
            ),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.1)),
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
        fig.add_vline(x=current_x_scaled, line_width=2, line_dash="dash", line_color=THEME["force"])

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


def dual_line_plot(x, y1, y2, title, y_label, quantity_type, name1="Actual", name2="Theory", current_x=None):
    x_arr = np.asarray(x, dtype=float)
    y1_arr = np.asarray(y1, dtype=float)
    y2_arr = np.asarray(y2, dtype=float)

    time_scale, time_unit = choose_si_scale(x_arr, "time")

    finite1 = y1_arr[np.isfinite(y1_arr)]
    finite2 = y2_arr[np.isfinite(y2_arr)]
    if finite1.size == 0 and finite2.size == 0:
        combined = np.array([0.0])
    else:
        combined = np.concatenate([finite1, finite2]) if finite1.size and finite2.size else (finite1 if finite1.size else finite2)

    value_scale, value_unit = choose_si_scale(combined, quantity_type)

    x_scaled = x_arr / time_scale if time_scale != 0 else x_arr
    y1_scaled = y1_arr / value_scale if value_scale != 0 else y1_arr
    y2_scaled = y2_arr / value_scale if value_scale != 0 else y2_arr
    current_x_scaled = current_x / time_scale if current_x is not None and time_scale != 0 else current_x

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_scaled,
            y=y1_scaled,
            mode="lines",
            line=dict(width=3.5, color=THEME["velocity"]),
            name=name1,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_scaled,
            y=y2_scaled,
            mode="lines",
            line=dict(width=3.0, dash="dash", color=THEME["force"]),
            name=name2,
        )
    )

    if current_x_scaled is not None:
        fig.add_vline(x=current_x_scaled, line_width=2, line_dash="dot", line_color=THEME["axis"])

    y_combined_scaled = np.concatenate([
        y1_scaled[np.isfinite(y1_scaled)],
        y2_scaled[np.isfinite(y2_scaled)]
    ]) if (np.any(np.isfinite(y1_scaled)) or np.any(np.isfinite(y2_scaled))) else np.array([0.0])

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis=common_axis_style(format_axis_title("Time", time_unit), padded_range(x_scaled, pad_frac=0.08)),
        yaxis=common_axis_style(format_axis_title(y_label, value_unit), padded_range(y_combined_scaled, pad_frac=0.15)),
        template="plotly_white",
        plot_bgcolor=THEME["bg"],
        paper_bgcolor=THEME["bg"],
        height=320,
        margin=dict(l=70, r=25, t=55, b=55),
        font=dict(size=14, color=THEME["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def verdict_box(status: str, speed_ratio: float):
    if status == "Escaping":
        st.success(f"Escaping: specific energy is positive and v / v_escape = {speed_ratio:.3f}")
    elif status == "Marginal escape":
        st.warning(f"Marginal escape: energy is near zero and v / v_escape = {speed_ratio:.3f}")
    elif status == "Collision":
        st.error("The object collided with the central body.")
    else:
        st.error(f"Bound: specific energy is negative and v / v_escape = {speed_ratio:.3f}")


# ============================================================
# Launch setup
# ============================================================
def build_escape_initial_state(R_body, altitude, mu, launch_plane_deg, launch_dir_mode, speed_mode, manual_speed, speed_fraction):
    r0_mag = R_body + altitude

    plane_angle = np.deg2rad(launch_plane_deg)
    radial_hat = np.array([np.cos(plane_angle), np.sin(plane_angle), 0.0])
    tangential_hat = np.array([-np.sin(plane_angle), np.cos(plane_angle), 0.0])

    r0 = r0_mag * radial_hat
    v_esc_local = np.sqrt(2.0 * mu / r0_mag)

    if speed_mode == "Fraction of escape speed":
        speed = speed_fraction * v_esc_local
    else:
        speed = manual_speed

    if launch_dir_mode == "Radial outward":
        v0 = speed * radial_hat
    elif launch_dir_mode == "Tangential":
        v0 = speed * tangential_hat
    else:
        direction = safe_unit(radial_hat + tangential_hat)
        v0 = speed * direction

    return np.hstack([r0, v0]), v_esc_local


# ============================================================
# Main app
# ============================================================
def run():
    init_orbital_state()

    st.title("Orbital Motion Explorer")

    st.markdown("### Governing equation")
    st.latex(r"\mathbf{F}=-\frac{GMm}{r^3}\mathbf{r}")

    st.markdown("### Specific orbital energy")
    st.latex(r"\varepsilon = \frac{v^2}{2} - \frac{GM}{r}")

    st.markdown("### Velocity–orbit relation (vis-viva equation)")
    st.latex(r"v^2=\mu\left(\frac{2}{r}-\frac{1}{a}\right)")

    st.markdown(
        r"""
This simulation runs in **real time from** \(t=0\).

- In **General Orbit** mode, you explore circular, elliptical, and escaping trajectories.
- In **Escape Velocity Explorer** mode, you choose launch speed and direction and the app shows whether the object is **bound**, **marginal**, or **escaping**.
- In an **elliptical orbit**, the object moves **fastest near periapsis** and **slowest near apoapsis**.
"""
    )

    st.sidebar.header("Mode")
    mode = st.sidebar.selectbox(
        "Choose simulation mode",
        ["General Orbit", "Escape Velocity Explorer"],
        key="orbital_mode",
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Central body")

    body_name = st.sidebar.selectbox(
        "Choose central body",
        list(CENTRAL_BODIES.keys()),
        key="orbital_body_type",
    )

    M = CENTRAL_BODIES[body_name]["mass"]
    R_body = CENTRAL_BODIES[body_name]["radius"]
    body_color = CENTRAL_BODIES[body_name]["color"]

    if body_name == "Custom":
        M = st.sidebar.number_input(
            "Central mass M (kg)",
            min_value=1.0e10,
            max_value=1.0e32,
            value=float(M),
            format="%.3e",
            key="orbital_custom_mass",
        )
        R_body = st.sidebar.number_input(
            "Central radius R (m)",
            min_value=1.0,
            max_value=1.0e10,
            value=float(R_body),
            format="%.3e",
            key="orbital_custom_radius",
        )

    mu = G * M

    st.sidebar.markdown("---")
    st.sidebar.header("Orbiting object")

    m_obj = st.sidebar.number_input(
        "Object mass m (kg)",
        min_value=1.0,
        max_value=1.0e8,
        value=1000.0,
        step=100.0,
        key="orbital_object_mass",
    )

    if mode == "General Orbit":
        st.sidebar.markdown("---")
        st.sidebar.header("Preset")

        preset = st.sidebar.selectbox(
            "Quick preset",
            [
                "Low circular orbit",
                "Elliptical orbit",
                "Escape trajectory",
                "Polar / 3D tilt",
                "Custom",
            ],
            key="orbital_preset",
        )

        r0x, r0y, r0z = 0.0, 0.0, 0.0
        v0x, v0y, v0z = 0.0, 0.0, 0.0

        default_altitude = 4.0e5 if body_name == "Earth" else 1.5e10 if body_name == "Sun" else 1.0e6
        r0_mag = R_body + default_altitude
        v_circ = np.sqrt(mu / r0_mag)
        v_esc = np.sqrt(2 * mu / r0_mag)

        if preset == "Low circular orbit":
            r0x, r0y, r0z = r0_mag, 0.0, 0.0
            v0x, v0y, v0z = 0.0, v_circ, 0.0
        elif preset == "Elliptical orbit":
            r0x, r0y, r0z = r0_mag, 0.0, 0.0
            v0x, v0y, v0z = 0.0, 0.72 * v_circ, 0.0
        elif preset == "Escape trajectory":
            r0x, r0y, r0z = r0_mag, 0.0, 0.0
            v0x, v0y, v0z = 0.0, 1.05 * v_esc, 0.0
        elif preset == "Polar / 3D tilt":
            r0x, r0y, r0z = r0_mag, 0.0, 0.0
            v0x, v0y, v0z = 0.0, 0.75 * v_circ, 0.55 * v_circ

        st.sidebar.markdown("---")
        st.sidebar.header("Initial position")
        x0 = st.sidebar.number_input("x₀ (m)", value=float(r0x), format="%.3e", key="orbital_x0")
        y0 = st.sidebar.number_input("y₀ (m)", value=float(r0y), format="%.3e", key="orbital_y0")
        z0 = st.sidebar.number_input("z₀ (m)", value=float(r0z), format="%.3e", key="orbital_z0")

        st.sidebar.markdown("---")
        st.sidebar.header("Initial velocity")
        vx0 = st.sidebar.number_input("vₓ(0) (m/s)", value=float(v0x), format="%.3e", key="orbital_vx0")
        vy0 = st.sidebar.number_input("vᵧ(0) (m/s)", value=float(v0y), format="%.3e", key="orbital_vy0")
        vz0 = st.sidebar.number_input("$v_z(0)$ (m/s)", value=float(v0z), format="%.3e", key="orbital_vz0")

        state0 = np.array([x0, y0, z0, vx0, vy0, vz0], dtype=float)

    else:
        st.sidebar.markdown("---")
        st.sidebar.header("Launch setup")

        if body_name == "Earth":
            default_altitude = 100000.0
        elif body_name == "Sun":
            default_altitude = 1.0e9
        else:
            default_altitude = 50000.0

        altitude = st.sidebar.number_input(
            "Launch altitude above surface (m)",
            min_value=0.0,
            max_value=float(100 * R_body),
            value=float(default_altitude),
            format="%.3e",
            key="orbital_escape_altitude",
        )

        launch_plane_deg = st.sidebar.slider(
            "Launch point angle around body (deg)",
            min_value=0.0,
            max_value=360.0,
            value=0.0,
            step=1.0,
            key="orbital_launch_plane_deg",
        )

        launch_dir_mode = st.sidebar.selectbox(
            "Launch direction",
            ["Radial outward", "Tangential", "45° outward"],
            key="orbital_launch_dir_mode",
        )

        speed_mode = st.sidebar.radio(
            "Speed input mode",
            ["Fraction of escape speed", "Manual speed"],
            key="orbital_speed_mode",
        )

        r_launch = R_body + altitude
        local_escape_speed_preview = np.sqrt(2 * mu / r_launch)

        if speed_mode == "Fraction of escape speed":
            speed_fraction = st.sidebar.slider(
                "Speed / local escape speed",
                min_value=0.10,
                max_value=2.00,
                value=1.00,
                step=0.01,
                key="orbital_escape_fraction",
            )
            manual_speed = speed_fraction * local_escape_speed_preview
        else:
            manual_speed = st.sidebar.number_input(
                "Launch speed (m/s)",
                min_value=0.0,
                max_value=float(5 * local_escape_speed_preview),
                value=float(local_escape_speed_preview),
                format="%.3e",
                key="orbital_escape_manual_speed",
            )
            speed_fraction = manual_speed / local_escape_speed_preview if local_escape_speed_preview > 0 else 0.0

        state0, local_escape_speed = build_escape_initial_state(
            R_body=R_body,
            altitude=altitude,
            mu=mu,
            launch_plane_deg=launch_plane_deg,
            launch_dir_mode=launch_dir_mode,
            speed_mode=speed_mode,
            manual_speed=manual_speed,
            speed_fraction=speed_fraction,
        )

        st.sidebar.caption(f"Local escape speed: {local_escape_speed:.3e} m/s")

    st.sidebar.markdown("---")
    st.sidebar.header("Animation")

    tmax_days_default = 0.2 if body_name == "Earth" else 365.0 if body_name == "Sun" else 5.0
    if mode == "Escape Velocity Explorer":
        tmax_days_default *= 1.5

    t_max = st.sidebar.slider(
        "Simulation time window",
        min_value=1.0,
        max_value=1000.0,
        value=float(tmax_days_default),
        step=1.0,
        key="orbital_tmax_days",
    ) * 86400.0

    playback_speed = st.sidebar.slider(
        "Playback speed multiplier",
        min_value=10.0,
        max_value=50000.0,
        value=3000.0,
        step=10.0,
        key="orbital_playback_speed",
    )

    trail_points = st.sidebar.slider(
        "Visible trail length",
        min_value=50,
        max_value=5000,
        value=1200,
        step=50,
        key="orbital_trail_points",
    )

    clear_trail = st.sidebar.checkbox(
        "Clear trail on reset",
        value=True,
        key="orbital_clear_trail",
    )

    signature = current_signature(mode, body_name, M, R_body, m_obj, state0, t_max)
    if st.session_state.orbital_prev_signature != signature:
        st.session_state.orbital_prev_signature = signature
        reset_simulation(state0)

    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        if st.button("▶ Play", key="orbital_play_btn", width="stretch"):
            if st.session_state.orbital_sim_time >= t_max:
                reset_simulation(state0)
            st.session_state.orbital_playing = True
            st.session_state.orbital_last_update = time.time()
            st.rerun()

    with c2:
        if st.button("⏸ Pause", key="orbital_pause_btn", width="stretch"):
            st.session_state.orbital_playing = False
            st.rerun()

    with c3:
        if st.button("⟲ Reset", key="orbital_reset_btn", width="stretch"):
            if clear_trail:
                reset_simulation(state0)
            else:
                st.session_state.orbital_state_vec = state0.copy()
                st.session_state.orbital_sim_time = 0.0
                st.session_state.orbital_playing = False
                st.session_state.orbital_last_update = time.time()
                st.session_state.orbital_collision = False
                st.session_state.orbital_escape = False
                st.session_state.orbital_status = "Bound"
                st.session_state.orbital_energy_positive_seen = False
            st.rerun()

    progress = min(st.session_state.orbital_sim_time / t_max, 1.0) if t_max > 0 else 0.0
    st.progress(progress)
    st.caption(f"Simulation progress: {100 * progress:.1f}%")

    update_simulation(mu=mu, R_body=R_body, playback_speed=playback_speed, t_max=t_max)

    state = st.session_state.orbital_state_vec.copy()
    r = state[:3]
    v = state[3:]
    F = gravitational_force(r, m_obj, mu)
    a_vec = gravitational_acceleration(r, mu)

    history_r = st.session_state.orbital_history_r[-trail_points:]
    history_v = st.session_state.orbital_history_v[-trail_points:]
    history_t = st.session_state.orbital_history_t[-trail_points:]

    history_r_arr = np.array(history_r)
    history_v_arr = np.array(history_v)

    radii = np.linalg.norm(history_r_arr, axis=1)
    speeds = np.linalg.norm(history_v_arr, axis=1)
    kinetic_energy = 0.5 * m_obj * speeds**2
    potential_energy = -mu * m_obj / np.maximum(radii, 1.0)
    total_energy = kinetic_energy + potential_energy
    angmom_hist = np.linalg.norm(np.cross(history_r_arr, m_obj * history_v_arr), axis=1)

    q = orbital_quantities(r, v, mu)

    if np.isfinite(q["a"]) and q["a"] > 0:
        vis_viva_speeds = np.sqrt(np.maximum(mu * (2.0 / np.maximum(radii, 1.0) - 1.0 / q["a"]), 0.0))
    else:
        vis_viva_speeds = np.full_like(speeds, np.nan, dtype=float)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Distance r", f"{q['r']:.3e} m")
    m2.metric("Speed v", f"{q['v']:.3e} m/s")
    if mode == "Escape Velocity Explorer":
        m3.metric("v / v_escape", f"{q['speed_ratio_escape']:.3f}")
        m4.metric("Verdict", st.session_state.orbital_status)
    else:
        m3.metric("Eccentricity e", f"{q['ecc']:.4f}" if np.isfinite(q["ecc"]) else "—")
        m4.metric(
            "Orbit type",
            classify_orbit(
                q["ecc"],
                q["energy_specific"],
                st.session_state.orbital_collision,
                st.session_state.orbital_escape,
            ),
        )

    if mode == "Escape Velocity Explorer":
        verdict_box(st.session_state.orbital_status, q["speed_ratio_escape"])
    else:
        if st.session_state.orbital_collision:
            st.error("The orbiting object has collided with the central body.")
        elif st.session_state.orbital_escape:
            st.warning("The orbiting object is on an escaping trajectory and left the visible domain.")
        else:
            st.info(
                "Watch the velocity change around an elliptical orbit: faster near periapsis, slower near apoapsis."
            )

    view2d, view3d = st.tabs(["2D real-time view", "3D real-time view"])

    with view2d:
        fig_xy = render_xy_scene(
            history_r=history_r,
            current_r=r,
            current_v=v,
            current_f=F,
            body_radius=R_body,
            body_color=body_color,
            ecc_vec=q["ecc_vec"],
            periapsis=q["periapsis"],
            apoapsis=q["apoapsis"],
        )
        st.plotly_chart(fig_xy, width="stretch", key="orbital_xy_view")

    with view3d:
        fig_3d = render_3d_scene(
            history_r=history_r,
            current_r=r,
            current_v=v,
            current_f=F,
            body_radius=R_body,
            body_color=body_color,
        )
        st.plotly_chart(fig_3d, width="stretch", key="orbital_3d_view")

    left, right = st.columns([1.15, 1.0])

    with left:
        row1, row2 = st.columns(2)
        with row1:
            st.plotly_chart(
                line_plot(
                    history_t,
                    radii,
                    "Distance from center vs time",
                    "Distance",
                    "distance",
                    st.session_state.orbital_sim_time,
                ),
                width="stretch",key="orbital_distance_time",
            )
        with row2:
            st.plotly_chart(
                line_plot(
                    history_t,
                    speeds,
                    "Speed vs time",
                    "Speed",
                    "speed",
                    st.session_state.orbital_sim_time,
                ),
                width="stretch",key="orbital_speed_time",
            )

        row3, row4 = st.columns(2)
        with row3:
            st.plotly_chart(
                dual_line_plot(
                    history_t,
                    speeds,
                    vis_viva_speeds,
                    "Actual speed vs vis-viva prediction",
                    "Speed",
                    "speed",
                    name1="Actual speed",
                    name2="Vis-viva speed",
                    current_x=st.session_state.orbital_sim_time,
                ),
                width="stretch", key="orbital_visviva_compare",
            )
        with row4:
            if mode == "General Orbit":
                st.plotly_chart(
                    line_plot(
                        history_t,
                        angmom_hist,
                        "Angular momentum vs time",
                        "Angular momentum",
                        "angmom",
                        st.session_state.orbital_sim_time,
                    ),
                    width="stretch", key="orbital_angmom_time",
                )
            else:
                ratio_vals = np.divide(
                    speeds,
                    np.sqrt(2 * mu / np.maximum(radii, 1.0)),
                    out=np.zeros_like(speeds),
                    where=radii > 0,
                )
                st.plotly_chart(
                    line_plot(
                        history_t,
                        ratio_vals,
                        "v / v_escape vs time",
                        "v / v_escape",
                        "ratio",
                        st.session_state.orbital_sim_time,
                    ),
                    width="stretch", key="orbital_escape_ratio_time",
                )

        row5, row6 = st.columns(2)
        with row5:
            st.plotly_chart(
                line_plot(
                    history_t,
                    total_energy,
                    "Total energy vs time",
                    "Energy",
                    "energy",
                    st.session_state.orbital_sim_time,
                ),
                width="stretch",
            )
        with row6:
            if np.isfinite(q["apoapsis"]) and np.isfinite(q["periapsis"]):
                apsis_metric = np.full_like(radii, q["apoapsis"] - q["periapsis"], dtype=float)
                st.plotly_chart(
                    line_plot(
                        history_t,
                        apsis_metric,
                        "Apoapsis - periapsis span",
                        "Orbit span",
                        "distance",
                        st.session_state.orbital_sim_time,
                    ),
                    width="stretch",
                )
            else:
                st.plotly_chart(
                    line_plot(
                        history_t,
                        radii,
                        "Distance from center vs time",
                        "Distance",
                        "distance",
                        st.session_state.orbital_sim_time,
                    ),
                    width="stretch",
                )

    with right:
        st.subheader("Current state")

        time_scale_disp, time_unit_disp = choose_si_scale([st.session_state.orbital_sim_time], "time")
        dist_scale_disp, dist_unit_disp = choose_si_scale(r, "distance")
        vel_scale_disp, vel_unit_disp = choose_si_scale(v, "speed")
        force_scale_disp, force_unit_disp = choose_si_scale(F, "force")
        acc_scale_disp, acc_unit_disp = choose_si_scale(a_vec, "acceleration")

        st.write(f"**Time:** {st.session_state.orbital_sim_time / time_scale_disp:.3f} {time_unit_disp}")
        st.write(f"**Position:** ({r[0] / dist_scale_disp:.3f}, {r[1] / dist_scale_disp:.3f}, {r[2] / dist_scale_disp:.3f}) {dist_unit_disp}")
        st.write(f"**Velocity:** ({v[0] / vel_scale_disp:.3f}, {v[1] / vel_scale_disp:.3f}, {v[2] / vel_scale_disp:.3f}) {vel_unit_disp}")
        st.write(f"**Distance from center:** {q['r'] / dist_scale_disp:.3f} {dist_unit_disp}")
        st.write(f"**Speed:** {q['v'] / vel_scale_disp:.3f} {vel_unit_disp}")
        st.write(f"**Local escape speed:** {q['escape_speed']:.3e} m/s")
        st.write(f"**Circular speed here:** {q['circular_speed']:.3e} m/s")
        st.write(f"**v / v_escape:** {q['speed_ratio_escape']:.4f}")
        st.write(f"**Specific orbital energy:** {q['energy_specific']:.3e} J/kg")
        st.write(f"**Gravity force magnitude:** {np.linalg.norm(F) / force_scale_disp:.3f} {force_unit_disp}")
        st.write(f"**Acceleration magnitude:** {np.linalg.norm(a_vec) / acc_scale_disp:.3f} {acc_unit_disp}")

        if np.isfinite(q["vis_viva_speed"]):
            st.write(f"**Vis-viva predicted speed here:** {q['vis_viva_speed']:.3e} m/s")
        else:
            st.write("**Vis-viva predicted speed here:** —")

        if np.isfinite(q["periapsis"]):
            st.write(f"**Periapsis distance:** {q['periapsis']:.3e} m")
        else:
            st.write("**Periapsis distance:** —")

        if np.isfinite(q["apoapsis"]):
            st.write(f"**Apoapsis distance:** {q['apoapsis']:.3e} m")
        else:
            st.write("**Apoapsis distance:** —")

        if mode == "General Orbit":
            if np.isfinite(q["a"]):
                st.write(f"**Semi-major axis a:** {q['a']:.3e} m")
            else:
                st.write("**Semi-major axis a:** ∞")
            st.write(f"**Angular momentum |h|:** {q['angmom_mag']:.3e} m²/s")
        else:
            st.write(f"**Launch verdict:** {st.session_state.orbital_status}")
            st.write(f"**Positive-energy phase observed:** {'Yes' if st.session_state.orbital_energy_positive_seen else 'No'}")

    with st.expander("What to observe"):
        if mode == "Escape Velocity Explorer":
            st.markdown(
                r"""
- If **v / v_escape < 1**, the object is usually bound and may fall back or remain on a closed path.
- If **v / v_escape \approx 1**, the motion is near the escape threshold.
- If **v / v_escape > 1**, the trajectory is unbound.
- A **radial outward** launch shows the clearest escape/fall-back behaviour.
- A **tangential** launch can still escape, but it may first curve around the body.
"""
            )
        else:
            st.markdown(
                r"""
- If the initial speed matches the local circular speed, the orbit is close to circular.
- If the speed is lower than circular but still bound, the orbit becomes elliptical.
- In an **elliptical orbit**, the object moves **fastest near periapsis** and **slowest near apoapsis**.
- The speed change follows the **vis-viva equation**:
  \[
  v^2=\mu\left(\frac{2}{r}-\frac{1}{a}\right)
  \]
- If the speed exceeds escape speed, the path becomes unbound.
- A nonzero **vz(0)** creates a tilted 3D orbit.
- In a clean numerical orbit, total energy and angular momentum should stay nearly constant.
"""
            )

    if st.session_state.orbital_playing:
        time.sleep(0.03)
        st.rerun()