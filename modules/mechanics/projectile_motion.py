import numpy as np
import plotly.graph_objects as go
import streamlit as st

MODULE_PREFIX = "projectile"
MAX_PROJECTILES = 3

COLORS = ["#2563eb", "#dc2626", "#16a34a"]
FAINT_COLORS = [
    "rgba(37,99,235,0.20)",
    "rgba(220,38,38,0.20)",
    "rgba(22,163,74,0.20)",
]


# ============================================================
# Helpers
# ============================================================
def _k(name: str) -> str:
    return f"{MODULE_PREFIX}_{name}"


def init_projectile_state() -> None:
    defaults = {
        _k("comparison_mode"): "Compare 3",
        _k("show_velocity"): True,
        _k("show_apex"): True,
        _k("show_reference_path"): True,
        _k("animation_speed"): 1.0,
        _k("animation_frames"): 120,
        _k("speed_1"): 20.0,
        _k("angle_1"): 45.0,
        _k("gravity_1"): 9.81,
        _k("speed_2"): 28.0,
        _k("angle_2"): 35.0,
        _k("gravity_2"): 9.81,
        _k("speed_3"): 32.0,
        _k("angle_3"): 60.0,
        _k("gravity_3"): 9.81,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def axis_style(title: str, axis_range=None) -> dict:
    d = dict(
        title=title,
        showgrid=True,
        gridcolor="rgba(120,140,160,0.20)",
        zeroline=True,
        zerolinecolor="rgba(80,100,120,0.30)",
        ticks="outside",
        ticklen=6,
        tickformat=".3~g",
    )
    if axis_range is not None:
        d["range"] = axis_range
    return d


# ============================================================
# Physics
# ============================================================
def compute_trajectory(v0: float, theta_deg: float, g: float, n: int = 700) -> dict:
    g = max(float(g), 1e-6)
    v0 = max(float(v0), 0.0)
    theta = np.radians(float(theta_deg))

    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)

    t_flight = max(2.0 * vy0 / g, 1e-6)
    t = np.linspace(0.0, t_flight, n)

    x = vx0 * t
    y = vy0 * t - 0.5 * g * t**2
    y = np.maximum(y, 0.0)

    vx = np.full_like(t, vx0)
    vy = vy0 - g * t
    speed = np.sqrt(vx**2 + vy**2)

    t_apex = vy0 / g
    x_apex = vx0 * t_apex
    y_apex = (vy0**2) / (2.0 * g)
    R = vx0 * t_flight

    return {
        "t": t,
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "speed": speed,
        "v0": v0,
        "theta_deg": theta_deg,
        "g": g,
        "vx0": vx0,
        "vy0": vy0,
        "t_flight": t_flight,
        "t_apex": t_apex,
        "x_apex": x_apex,
        "y_apex": y_apex,
        "R": R,
        "H": y_apex,
    }


def sample_state(traj: dict, t_now: float) -> dict:
    t_now = float(np.clip(t_now, 0.0, traj["t_flight"]))
    g = traj["g"]

    x = traj["vx0"] * t_now
    y = max(traj["vy0"] * t_now - 0.5 * g * t_now**2, 0.0)
    vx = traj["vx0"]
    vy = traj["vy0"] - g * t_now
    v = float(np.sqrt(vx**2 + vy**2))

    return {"t": t_now, "x": x, "y": y, "vx": vx, "vy": vy, "v": v}


def get_active_count() -> int:
    mode = st.session_state[_k("comparison_mode")]
    return {"Single": 1, "Compare 2": 2, "Compare 3": 3}[mode]


def build_trajectories() -> list[dict]:
    trjs = []
    for i in range(1, MAX_PROJECTILES + 1):
        trjs.append(
            compute_trajectory(
                st.session_state[_k(f"speed_{i}")],
                st.session_state[_k(f"angle_{i}")],
                st.session_state[_k(f"gravity_{i}")],
            )
        )
    return trjs


# ============================================================
# UI controls
# ============================================================
def controls_panel() -> int:
    with st.sidebar:
        st.header("Projectile Controls")

        st.radio(
            "Comparison mode",
            ["Single", "Compare 2", "Compare 3"],
            key=_k("comparison_mode"),
        )

        st.slider(
            "Animation speed",
            min_value=0.4,
            max_value=2.5,
            step=0.1,
            key=_k("animation_speed"),
        )

        st.slider(
            "Animation smoothness",
            min_value=60,
            max_value=220,
            step=10,
            key=_k("animation_frames"),
            help="More frames gives smoother motion.",
        )

        st.checkbox("Show velocity vector", key=_k("show_velocity"))
        st.checkbox("Show apex marker", key=_k("show_apex"))
        st.checkbox("Show full parabolic path", key=_k("show_reference_path"))

        st.markdown("---")
        st.subheader("Launch parameters")

        active_count = get_active_count()

        for i in range(1, MAX_PROJECTILES + 1):
            with st.expander(f"Projectile {i}", expanded=(i <= active_count)):
                st.slider(
                    f"Initial speed v₀ {i} (m/s)",
                    min_value=1.0,
                    max_value=100.0,
                    step=1.0,
                    key=_k(f"speed_{i}"),
                )
                st.slider(
                    f"Launch angle θ {i} (°)",
                    min_value=1.0,
                    max_value=89.0,
                    step=1.0,
                    key=_k(f"angle_{i}"),
                )
                st.slider(
                    f"Gravity g {i} (m/s²)",
                    min_value=1.0,
                    max_value=20.0,
                    step=0.01,
                    key=_k(f"gravity_{i}"),
                )

    return active_count


# ============================================================
# Plotting
# ============================================================
def make_live_figure(
    trajectories: list[dict],
    active_count: int,
    show_velocity: bool,
    show_apex: bool,
    show_reference_path: bool,
    n_frames: int,
    animation_speed: float,
) -> go.Figure:
    fig = go.Figure()

    x_max = max(np.max(tr["x"]) for tr in trajectories[:active_count])
    y_max = max(np.max(tr["y"]) for tr in trajectories[:active_count])
    x_pad = max(2.0, 0.08 * x_max)
    y_pad = max(2.0, 0.12 * y_max)

    # Static background: full parabola + apex
    for i in range(active_count):
        tr = trajectories[i]
        color = COLORS[i]
        faint = FAINT_COLORS[i]

        if show_reference_path:
            fig.add_trace(
                go.Scatter(
                    x=tr["x"],
                    y=tr["y"],
                    mode="lines",
                    name=f"Projectile {i+1} path",
                    line=dict(color=faint, width=5, shape="spline", smoothing=1.0),
                    hovertemplate="x = %{x:.2f} m<br>y = %{y:.2f} m<extra></extra>",
                )
            )

        if show_apex:
            fig.add_trace(
                go.Scatter(
                    x=[tr["x_apex"]],
                    y=[tr["y_apex"]],
                    mode="markers",
                    name=f"Projectile {i+1} apex",
                    marker=dict(color=color, size=10, symbol="diamond"),
                    hovertemplate="Apex<br>x = %{x:.2f} m<br>y = %{y:.2f} m<extra></extra>",
                )
            )

    # Initial dynamic traces
    dynamic_trace_count_per_projectile = 3 if show_velocity else 2

    for i in range(active_count):
        tr = trajectories[i]
        color = COLORS[i]
        s = sample_state(tr, 0.0)

        # growing trail
        fig.add_trace(
            go.Scatter(
                x=[0.0],
                y=[0.0],
                mode="lines",
                line=dict(color=color, width=4, shape="spline", smoothing=1.0),
                name=f"Projectile {i+1} travelled",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # moving body
        fig.add_trace(
            go.Scatter(
                x=[s["x"]],
                y=[s["y"]],
                mode="markers",
                marker=dict(size=14, color=color, line=dict(color="white", width=1.5)),
                name=f"Projectile {i+1}",
                hovertemplate="x = %{x:.2f} m<br>y = %{y:.2f} m<extra></extra>",
            )
        )

        # velocity vector
        if show_velocity:
            scale = 0.10 * max(x_max, y_max, 1.0) / max(np.max(tr["speed"]), 1e-6)
            fig.add_trace(
                go.Scatter(
                    x=[s["x"], s["x"] + s["vx"] * scale],
                    y=[s["y"], s["y"] + s["vy"] * scale],
                    mode="lines",
                    line=dict(color=color, width=3),
                    name=f"v{i+1}",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    max_t = max(tr["t_flight"] for tr in trajectories[:active_count])
    frame_times = np.linspace(0.0, max_t, max(int(n_frames), 30))
    frames = []

    for t_now in frame_times:
        frame_data = []

        for i in range(active_count):
            tr = trajectories[i]
            color = COLORS[i]

            idx = np.searchsorted(tr["t"], t_now, side="right")
            idx = max(1, min(idx, len(tr["t"])))
            x_seg = tr["x"][:idx]
            y_seg = tr["y"][:idx]

            s = sample_state(tr, t_now)

            frame_data.append(
                go.Scatter(
                    x=x_seg,
                    y=y_seg,
                    mode="lines",
                    line=dict(color=color, width=4, shape="spline", smoothing=1.0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            frame_data.append(
                go.Scatter(
                    x=[s["x"]],
                    y=[s["y"]],
                    mode="markers",
                    marker=dict(size=14, color=color, line=dict(color="white", width=1.5)),
                    showlegend=False,
                    hovertemplate="x = %{x:.2f} m<br>y = %{y:.2f} m<extra></extra>",
                )
            )

            if show_velocity:
                scale = 0.10 * max(x_max, y_max, 1.0) / max(np.max(tr["speed"]), 1e-6)
                frame_data.append(
                    go.Scatter(
                        x=[s["x"], s["x"] + s["vx"] * scale],
                        y=[s["y"], s["y"] + s["vy"] * scale],
                        mode="lines",
                        line=dict(color=color, width=3),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

        frames.append(go.Frame(data=frame_data, name=f"{t_now:.4f}"))

    fig.frames = frames

    duration_ms = int(max(10, 35 / max(animation_speed, 0.1)))

    fig.update_layout(
    title=dict(text="Real-Time Projectile Motion", x=0.5),
    xaxis=axis_style("Horizontal distance x (m)", [-0.1 * x_pad, x_max + x_pad]),
    yaxis=axis_style("Vertical height y (m)", [-0.1 * y_pad, y_max + y_pad]),
    template="plotly_white",
    plot_bgcolor="white",
    paper_bgcolor="white",
    height=620,
    margin=dict(l=60, r=20, t=80, b=150),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    updatemenus=[
        {
            "type": "buttons",
            "direction": "left",
            "x": 0.5,
            "y": -0.30,
            "xanchor": "center",
            "yanchor": "top",
            "showactive": False,
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": duration_ms, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0},
                        },
                    ],
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": False},
                            "transition": {"duration": 0},
                        },
                    ],
                },
                {
                    "label": "↺ Restart",
                    "method": "animate",
                    "args": [
                        [fig.frames[0].name] if fig.frames else [None],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": True},
                            "transition": {"duration": 0},
                        },
                    ],
                },
            ],
        }
    ],
    sliders=[
        {
            "active": 0,
            "y": -0.08,
            "x": 0.08,
            "len": 0.84,
            "currentvalue": {"prefix": "Time t = ", "suffix": " s"},
            "pad": {"t": 10},
            "steps": [
                {
                    "label": f"{float(fr.name):.2f}",
                    "method": "animate",
                    "args": [
                        [fr.name],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": True},
                            "transition": {"duration": 0},
                        },
                    ],
                }
                for fr in fig.frames
            ],
        }
    ],
)

    return fig


def make_comparison_figure(trajectories: list[dict], active_count: int) -> go.Figure:
    fig = go.Figure()

    for i in range(active_count):
        tr = trajectories[i]
        fig.add_trace(
            go.Scatter(
                x=tr["x"],
                y=tr["y"],
                mode="lines",
                name=f"Projectile {i+1}",
                line=dict(color=COLORS[i], width=4, shape="spline", smoothing=1.0),
                hovertemplate="x = %{x:.2f} m<br>y = %{y:.2f} m<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text="Trajectory Comparison", x=0.5),
        xaxis=axis_style("Horizontal distance x (m)"),
        yaxis=axis_style("Vertical height y (m)"),
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=470,
        margin=dict(l=60, r=20, t=60, b=55),
    )
    return fig


def make_component_figure(
    trajectories: list[dict],
    active_count: int,
    quantity: str,
    title: str,
    y_label: str,
    t_mark: float,
) -> go.Figure:
    fig = go.Figure()

    for i in range(active_count):
        tr = trajectories[i]
        fig.add_trace(
            go.Scatter(
                x=tr["t"],
                y=tr[quantity],
                mode="lines",
                name=f"Projectile {i+1}",
                line=dict(color=COLORS[i], width=3, shape="spline", smoothing=0.8),
            )
        )

    fig.add_vline(
        x=t_mark,
        line_width=2,
        line_dash="dash",
        line_color="rgba(50,50,50,0.55)",
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=axis_style("Time t (s)"),
        yaxis=axis_style(y_label),
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=340,
        margin=dict(l=60, r=20, t=55, b=55),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1.0),
    )
    return fig


# ============================================================
# Tables and readout
# ============================================================
def comparison_table(trajectories: list[dict], active_count: int) -> None:
    rows = []
    for i in range(active_count):
        tr = trajectories[i]
        rows.append(
            {
                "Projectile": f"{i+1}",
                "v₀ (m/s)": f"{tr['v0']:.2f}",
                "θ (°)": f"{tr['theta_deg']:.2f}",
                "g (m/s²)": f"{tr['g']:.2f}",
                "vₓ₀ (m/s)": f"{tr['vx0']:.2f}",
                "vᵧ₀ (m/s)": f"{tr['vy0']:.2f}",
                "Time of flight T (s)": f"{tr['t_flight']:.2f}",
                "Maximum height Hₘₐₓ (m)": f"{tr['H']:.2f}",
                "Horizontal range R (m)": f"{tr['R']:.2f}",
            }
        )

    st.dataframe(rows, width="stretch", hide_index=True)


def manual_readout(trajectories: list[dict], active_count: int, t_inspect: float) -> None:
    st.markdown("### Manual time inspection")
    cols = st.columns(active_count)

    for i in range(active_count):
        s = sample_state(trajectories[i], t_inspect)
        with cols[i]:
            st.markdown(f"#### Projectile {i+1}")
            st.write(f"**t** = {s['t']:.2f} s")
            st.write(f"**x** = {s['x']:.2f} m")
            st.write(f"**y** = {s['y']:.2f} m")
            st.write(f"**vₓ** = {s['vx']:.2f} m/s")
            st.write(f"**vᵧ** = {s['vy']:.2f} m/s")
            st.write(f"**|v|** = {s['v']:.2f} m/s")


def teaching_notes() -> None:
    st.markdown("### Physics notes")
    st.markdown("- The trajectory is parabolic because horizontal motion is uniform while vertical motion is uniformly accelerated.")
    st.markdown("- The horizontal component **vₓ** remains constant in ideal projectile motion.")
    st.markdown("- The vertical component **vᵧ** decreases linearly with time due to gravity.")
    st.markdown("- For equal launch and landing heights with no air resistance, angles near **45°** often give the maximum range.")
    st.markdown("- Reducing **g** increases time of flight and usually increases range.")

    with st.expander("Equations used"):
        st.latex(r"x(t)=v_0\cos\theta \, t")
        st.latex(r"y(t)=v_0\sin\theta \, t-\frac{1}{2}gt^2")
        st.latex(r"v_x(t)=v_0\cos\theta")
        st.latex(r"v_y(t)=v_0\sin\theta-gt")
        st.latex(r"T=\frac{2v_0\sin\theta}{g}")
        st.latex(r"R=\frac{v_0^2\sin(2\theta)}{g}")
        st.latex(r"H_{\max}=\frac{v_0^2\sin^2\theta}{2g}")


# ============================================================
# Main
# ============================================================
def run() -> None:
    init_projectile_state()

    st.title("Projectile Motion Explorer")
    st.markdown(
        "A smooth projectile-motion module with real-time parabolic animation, side-by-side comparison, and physics analysis."
    )

    active_count = controls_panel()
    trajectories = build_trajectories()

    live_fig = make_live_figure(
        trajectories=trajectories,
        active_count=active_count,
        show_velocity=st.session_state[_k("show_velocity")],
        show_apex=st.session_state[_k("show_apex")],
        show_reference_path=st.session_state[_k("show_reference_path")],
        n_frames=int(st.session_state[_k("animation_frames")]),
        animation_speed=float(st.session_state[_k("animation_speed")]),
    )
    st.plotly_chart(live_fig, width="stretch", key="projectile_live_canvas")

    max_t = max(tr["t_flight"] for tr in trajectories[:active_count])
    t_inspect = st.slider(
        "Manual inspection time t (s)",
        min_value=0.0,
        max_value=float(max_t),
        value=min(float(max_t) * 0.35, float(max_t)),
        step=0.05,
    )

    manual_readout(trajectories, active_count, t_inspect)

    tab1, tab2, tab3 = st.tabs(
        ["Comparison", "Component graphs", "Summary"]
    )

    with tab1:
        st.plotly_chart(
            make_comparison_figure(trajectories, active_count),
            width="stretch",
            key="projectile_comparison_plot",
        )
        comparison_table(trajectories, active_count)

    with tab2:
        c1, c2 = st.columns(2)

        with c1:
            st.plotly_chart(
                make_component_figure(
                    trajectories, active_count, "x",
                    "Horizontal position x(t)", "x (m)", t_inspect
                ),
                width="stretch",
                key="projectile_xt",
            )
            st.plotly_chart(
                make_component_figure(
                    trajectories, active_count, "vy",
                    "Vertical velocity vᵧ(t)", "vᵧ (m/s)", t_inspect
                ),
                width="stretch",
                key="projectile_vyt",
            )

        with c2:
            st.plotly_chart(
                make_component_figure(
                    trajectories, active_count, "y",
                    "Vertical position y(t)", "y (m)", t_inspect
                ),
                width="stretch",
                key="projectile_yt",
            )
            st.plotly_chart(
                make_component_figure(
                    trajectories, active_count, "speed",
                    "Speed magnitude |v|(t)", "|v| (m/s)", t_inspect
                ),
                width="stretch",
                key="projectile_speedt",
            )

    with tab3:
        comparison_table(trajectories, active_count)
        teaching_notes()