import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# =========================================================
# MODULE CONFIG
# =========================================================
MODULE_PREFIX = "osc"
MAX_OSCILLATORS = 3


# =========================================================
# STATE HELPERS
# =========================================================
def _k(name: str) -> str:
    return f"{MODULE_PREFIX}_{name}"


def ensure_module_state():
    defaults = {
        _k("playing"): False,
        _k("sim_time"): 0.0,
        _k("last_update"): time.time(),
        _k("speed"): 1.0,
        _k("time_window"): 10.0,
        _k("refresh_rate"): 0.06,
        _k("mode"): "Spring–Mass",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_animation():
    st.session_state[_k("sim_time")] = 0.0
    st.session_state[_k("last_update")] = time.time()


def set_playing(value: bool):
    st.session_state[_k("playing")] = value
    st.session_state[_k("last_update")] = time.time()


def toggle_playing():
    st.session_state[_k("playing")] = not st.session_state[_k("playing")]
    st.session_state[_k("last_update")] = time.time()


def update_simulation_time():
    now = time.time()
    last = st.session_state[_k("last_update")]

    if st.session_state[_k("playing")]:
        dt = now - last
        st.session_state[_k("sim_time")] += dt * st.session_state[_k("speed")]

    st.session_state[_k("last_update")] = now


def get_sim_time() -> float:
    return st.session_state[_k("sim_time")]


# =========================================================
# NUMERICAL HELPERS
# =========================================================
def rk4_step(f, t, y, dt, *args):
    k1 = f(t, y, *args)
    k2 = f(t + dt / 2, y + dt * k1 / 2, *args)
    k3 = f(t + dt / 2, y + dt * k2 / 2, *args)
    k4 = f(t + dt, y + dt * k3, *args)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def solve_ode_system(deriv_func, y0, t_array, *args):
    y = np.zeros((len(t_array), len(y0)))
    y[0] = y0

    for i in range(1, len(t_array)):
        dt = t_array[i] - t_array[i - 1]
        y[i] = rk4_step(deriv_func, t_array[i - 1], y[i - 1], dt, *args)

    return y


# =========================================================
# PHYSICS HELPERS
# =========================================================
def spring_state(m, k, A, phi, t):
    omega = np.sqrt(k / m)
    T = 2 * np.pi / omega
    f = 1 / T

    x = A * np.cos(omega * t + phi)
    v = -A * omega * np.sin(omega * t + phi)
    a = -omega**2 * x

    ke = 0.5 * m * v**2
    pe = 0.5 * k * x**2
    total_e = ke + pe

    return {
        "omega": omega,
        "T": T,
        "f": f,
        "x": x,
        "v": v,
        "a": a,
        "ke": ke,
        "pe": pe,
        "total_e": total_e,
    }


def pendulum_derivs(t, y, L, g):
    theta, omega = y
    dtheta = omega
    domega = -(g / L) * np.sin(theta)
    return np.array([dtheta, domega])


def damped_derivs(t, y, m, c, k):
    x, v = y
    dx = v
    dv = -(c / m) * v - (k / m) * x
    return np.array([dx, dv])


# =========================================================
# DRAWING HELPERS
# =========================================================
def style_axis(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)


def draw_spring_mass(ax, x, A, title="Spring–Mass Motion"):
    wall_x = -A - 0.9
    mass_width = 0.35
    mass_height = 0.28
    mass_center = x
    mass_left = mass_center - mass_width / 2

    coil_start = wall_x + 0.06
    coil_end = mass_left
    if coil_end <= coil_start + 0.05:
        coil_end = coil_start + 0.05

    spring_x = np.linspace(coil_start, coil_end, 140)
    spring_y = 0.10 * np.sin(np.linspace(0, 10 * np.pi, 140))

    ax.plot([wall_x, wall_x], [-0.5, 0.5], linewidth=4)
    ax.plot(spring_x, spring_y, linewidth=2)
    ax.axvline(0, linestyle="--", linewidth=1, alpha=0.5)

    rect = plt.Rectangle((mass_left, -mass_height / 2), mass_width, mass_height, fill=True, alpha=0.85)
    ax.add_patch(rect)

    ax.text(0, 0.33, "Equilibrium", ha="center", fontsize=9)

    lim = max(A + 1.2, 1.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.8, 0.8)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=13)
    ax.axis("off")


def draw_pendulum(ax, theta, L, title="Pendulum Motion"):
    x = L * np.sin(theta)
    y = -L * np.cos(theta)

    ax.plot([0, x], [0, y], linewidth=2.5)
    ax.scatter([0], [0], s=60)
    ax.scatter([x], [y], s=240)

    ax.plot([0, 0], [0, -L], linestyle="--", alpha=0.4)

    arc_theta = np.linspace(0, theta, 100)
    arc_r = 0.28 * L
    ax.plot(arc_r * np.sin(arc_theta), -arc_r * np.cos(arc_theta), linestyle="--", alpha=0.7)

    lim = max(1.2 * L, 1.2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, 0.3 * lim)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.25)


def progressive_time_array(sim_time: float, time_window: float, density: int = 700):
    tau = sim_time % time_window
    if tau < 1e-4:
        tau = 1e-4
    n_points = max(150, int(density * tau / time_window))
    return tau, np.linspace(0, tau, n_points)


# =========================================================
# CONTROL PANEL
# =========================================================
def render_sidebar_controls():
    st.sidebar.title("Oscillations Lab")

    st.session_state[_k("mode")] = st.sidebar.selectbox(
        "Choose section",
        ["Spring–Mass", "Simple Pendulum", "Damped Oscillator"],
        key=_k("mode_select"),
    )

    st.sidebar.markdown("### Animation Controls")

    st.session_state[_k("speed")] = st.sidebar.slider(
        "Animation speed",
        min_value=0.1,
        max_value=3.0,
        value=float(st.session_state[_k("speed")]),
        step=0.1,
        key=_k("speed_slider"),
    )

    st.session_state[_k("time_window")] = st.sidebar.slider(
        "Time window (s)",
        min_value=2.0,
        max_value=100.0,
        value=float(st.session_state[_k("time_window")]),
        step=1.0,
        key=_k("window_slider"),
    )

    st.session_state[_k("refresh_rate")] = st.sidebar.slider(
        "Refresh interval (s)",
        min_value=0.03,
        max_value=0.20,
        value=float(st.session_state[_k("refresh_rate")]),
        step=0.01,
        key=_k("refresh_slider"),
    )


def render_top_controls():
    play_text = "Pause" if st.session_state[_k("playing")] else "Play"

    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        if st.button(play_text, key=_k("play_pause_btn"), use_container_width=True):
            toggle_playing()

    with c2:
        if st.button("Reset", key=_k("reset_btn"), use_container_width=True):
            reset_animation()

    with c3:
        st.markdown(f"**Simulation time:** {st.session_state[_k('sim_time')]:.2f} s")


# =========================================================
# SECTION: SPRING–MASS
# =========================================================
def run_spring_mass(sim_time, time_window):
    st.header("Spring–Mass Oscillators")
    st.markdown(
        """
Compare multiple spring–mass oscillators and see how mass, spring constant,
amplitude, and phase affect displacement and energy.
"""
    )

    st.sidebar.markdown("### Spring–Mass Parameters")

    n_osc = st.sidebar.slider(
        "Number of oscillators",
        min_value=1,
        max_value=MAX_OSCILLATORS,
        value=2,
        step=1,
        key=_k("spring_n_osc"),
    )

    params = []
    for i in range(n_osc):
        with st.sidebar.expander(f"Oscillator {i + 1}", expanded=(i == 0)):
            m = st.slider(
                f"Mass m{i + 1} (kg)",
                0.1, 10.0, 1.0 + 0.5 * i, 0.1,
                key=_k(f"spring_m_{i}")
            )
            k = st.slider(
                f"Spring constant k{i + 1} (N/m)",
                0.5, 50.0, 10.0 + 5.0 * i, 0.5,
                key=_k(f"spring_k_{i}")
            )
            A = st.slider(
                f"Amplitude A{i + 1} (m)",
                0.1, 5.0, 1.0, 0.1,
                key=_k(f"spring_A_{i}")
            )
            phi = st.slider(
                f"Phase φ{i + 1} (rad)",
                0.0, float(2 * np.pi), 0.0, 0.1,
                key=_k(f"spring_phi_{i}")
            )
            params.append({"m": m, "k": k, "A": A, "phi": phi})

    st.subheader("Individual Oscillator Views")

    for i, p in enumerate(params):
        state_now = spring_state(p["m"], p["k"], p["A"], p["phi"], sim_time)
        omega = state_now["omega"]

        tau, t_hist = progressive_time_array(sim_time, time_window, density=650)
        x_hist = p["A"] * np.cos(omega * t_hist + p["phi"])
        v_hist = -p["A"] * omega * np.sin(omega * t_hist + p["phi"])
        ke_hist = 0.5 * p["m"] * v_hist**2
        pe_hist = 0.5 * p["k"] * x_hist**2
        total_e_hist = ke_hist + pe_hist

        st.markdown(f"### Oscillator {i + 1}")

        q1, q2, q3 = st.columns(3)
        q1.metric("Angular frequency ω", f"{state_now['omega']:.3f} rad/s")
        q2.metric("Period T", f"{state_now['T']:.3f} s")
        q3.metric("Frequency f", f"{state_now['f']:.3f} Hz")

        r1, r2, r3 = st.columns(3)
        r1.metric("Displacement x", f"{state_now['x']:.3f} m")
        r2.metric("Velocity v", f"{state_now['v']:.3f} m/s")
        r3.metric("Acceleration a", f"{state_now['a']:.3f} m/s²")

        r4, r5, r6 = st.columns(3)
        r4.metric("Kinetic energy", f"{state_now['ke']:.3f} J")
        r5.metric("Potential energy", f"{state_now['pe']:.3f} J")
        r6.metric("Total energy", f"{state_now['total_e']:.3f} J")

        col_left, col_right = st.columns(2)

        with col_left:
            fig_snap, ax_snap = plt.subplots(figsize=(8, 2.8))
            draw_spring_mass(ax_snap, state_now["x"], p["A"], title=f"Oscillator {i + 1}: Motion")
            st.pyplot(fig_snap, width="stretch")

        with col_right:
            fig_x, ax_x = plt.subplots(figsize=(8, 3.0))
            ax_x.plot(t_hist, x_hist, linewidth=2)
            ax_x.scatter([tau], [state_now["x"]], s=50)
            ax_x.set_xlim(0, time_window)
            style_axis(ax_x, "Time (s)", "Displacement x (m)", f"Oscillator {i + 1}: Displacement")
            st.pyplot(fig_x, width="stretch")

        fig_e, ax_e = plt.subplots(figsize=(10, 3.6))
        ax_e.plot(t_hist, ke_hist, label="Kinetic Energy")
        ax_e.plot(t_hist, pe_hist, label="Potential Energy")
        ax_e.plot(t_hist, total_e_hist, label="Total Energy")
        ax_e.scatter([tau], [state_now["ke"]], s=35)
        ax_e.scatter([tau], [state_now["pe"]], s=35)
        ax_e.scatter([tau], [state_now["total_e"]], s=35)
        ax_e.set_xlim(0, time_window)
        style_axis(ax_e, "Time (s)", "Energy (J)", f"Oscillator {i + 1}: Energy")
        ax_e.legend()
        st.pyplot(fig_e, width="stretch")

        with st.expander(f"Show equations for Oscillator {i + 1}", expanded=False):
            st.latex(r"x(t)=A\cos(\omega t+\phi)")
            st.latex(r"v(t)=-A\omega\sin(\omega t+\phi)")
            st.latex(r"a(t)=-\omega^2x(t)")
            st.latex(r"\omega=\sqrt{\frac{k}{m}}")
            st.latex(r"T=\frac{2\pi}{\omega}")
            st.latex(r"KE=\frac{1}{2}mv^2,\quad PE=\frac{1}{2}kx^2")

        st.markdown("---")

    st.subheader("Comparison View")

    fig_compare, ax_compare = plt.subplots(figsize=(10, 4.5))
    tau, t_hist = progressive_time_array(sim_time, time_window, density=650)

    for i, p in enumerate(params):
        omega = np.sqrt(p["k"] / p["m"])
        x_hist = p["A"] * np.cos(omega * t_hist + p["phi"])
        ax_compare.plot(t_hist, x_hist, label=f"Oscillator {i + 1}")

    ax_compare.set_xlim(0, time_window)
    style_axis(ax_compare, "Time (s)", "Displacement x (m)", "Comparison of Oscillators")
    ax_compare.legend()
    st.pyplot(fig_compare, width="stretch")


# =========================================================
# SECTION: SIMPLE PENDULUM
# =========================================================
def run_simple_pendulum(sim_time, time_window):
    st.header("Simple Pendulum")
    st.markdown(
        r"""
This section uses the nonlinear equation
\[
\ddot{\theta}+\frac{g}{L}\sin\theta=0
\]
so the motion remains valid even for larger angles.
"""
    )

    st.sidebar.markdown("### Pendulum Parameters")

    L = st.sidebar.slider("Length L (m)", 0.2, 5.0, 1.5, 0.1, key=_k("pend_L"))
    g = st.sidebar.slider("Gravity g (m/s²)", 1.0, 20.0, 9.81, 0.1, key=_k("pend_g"))
    theta0_deg = st.sidebar.slider("Initial angle θ₀ (deg)", 1, 85, 20, 1, key=_k("pend_theta0"))
    omega0 = st.sidebar.slider("Initial angular velocity ω₀ (rad/s)", -5.0, 5.0, 0.0, 0.1, key=_k("pend_omega0"))
    bob_mass = st.sidebar.slider("Bob mass m (kg)", 0.1, 10.0, 1.0, 0.1, key=_k("pend_mass"))

    theta0 = np.deg2rad(theta0_deg)

    tau, t_hist = progressive_time_array(sim_time, time_window, density=800)

    sol = solve_ode_system(
    pendulum_derivs,
    np.array([theta0, omega0]),
    t_hist,
    L,
    g
    )
    

    theta_hist = sol[:, 0]
    omega_hist = sol[:, 1]

    theta_now = theta_hist[-1]
    omega_now = omega_hist[-1]
    alpha_now = -(g / L) * np.sin(theta_now)

    ke_hist = 0.5 * bob_mass * (L**2) * omega_hist**2
    pe_hist = bob_mass * g * L * (1 - np.cos(theta_hist))
    total_e_hist = ke_hist + pe_hist

    ke_now = ke_hist[-1]
    pe_now = pe_hist[-1]
    total_e_now = total_e_hist[-1]

    T_small = 2 * np.pi * np.sqrt(L / g)
    f_small = 1 / T_small

    q1, q2, q3 = st.columns(3)
    q1.metric("Current angle θ", f"{np.rad2deg(theta_now):.2f}°")
    q2.metric("Angular velocity ω", f"{omega_now:.3f} rad/s")
    q3.metric("Angular acceleration α", f"{alpha_now:.3f} rad/s²")

    r1, r2, r3 = st.columns(3)
    r1.metric("Small-angle period", f"{T_small:.3f} s")
    r2.metric("Small-angle frequency", f"{f_small:.3f} Hz")
    r3.metric("Length L", f"{L:.2f} m")

    r4, r5, r6 = st.columns(3)
    r4.metric("Kinetic energy", f"{ke_now:.3f} J")
    r5.metric("Potential energy", f"{pe_now:.3f} J")
    r6.metric("Total energy", f"{total_e_now:.3f} J")

    col_left, col_right = st.columns(2)

    with col_left:
        fig_p, ax_p = plt.subplots(figsize=(6, 5))
        draw_pendulum(ax_p, theta_now, L, title="Pendulum Motion")
        st.pyplot(fig_p, width="stretch")

    with col_right:
        fig_theta, ax_theta = plt.subplots(figsize=(8, 3.0))
        ax_theta.plot(t_hist, np.rad2deg(theta_hist), linewidth=2)
        ax_theta.scatter([tau], [np.rad2deg(theta_now)], s=50)
        ax_theta.set_xlim(0, time_window)
        style_axis(ax_theta, "Time (s)", "Angle θ (deg)", "Angular Displacement")
        st.pyplot(fig_theta, width="stretch")

    fig_omega, ax_omega = plt.subplots(figsize=(10, 3.2))
    ax_omega.plot(t_hist, omega_hist, linewidth=2)
    ax_omega.scatter([tau], [omega_now], s=45)
    ax_omega.set_xlim(0, time_window)
    style_axis(ax_omega, "Time (s)", "Angular velocity ω (rad/s)", "Angular Velocity")
    st.pyplot(fig_omega, width="stretch")

    fig_e, ax_e = plt.subplots(figsize=(10, 3.5))
    ax_e.plot(t_hist, ke_hist, label="Kinetic Energy")
    ax_e.plot(t_hist, pe_hist, label="Potential Energy")
    ax_e.plot(t_hist, total_e_hist, label="Total Energy")
    ax_e.scatter([tau], [ke_now], s=35)
    ax_e.scatter([tau], [pe_now], s=35)
    ax_e.scatter([tau], [total_e_now], s=35)
    ax_e.set_xlim(0, time_window)
    style_axis(ax_e, "Time (s)", "Energy (J)", "Pendulum Energy")
    ax_e.legend()
    st.pyplot(fig_e, width="stretch")

    with st.expander("Show equations", expanded=False):
        st.latex(r"\ddot{\theta}+\frac{g}{L}\sin\theta=0")
        st.latex(r"KE=\frac{1}{2}mL^2\omega^2")
        st.latex(r"PE=mgL(1-\cos\theta)")
        st.latex(r"T_{\mathrm{small\ angle}}=2\pi\sqrt{\frac{L}{g}}")


# =========================================================
# SECTION: DAMPED OSCILLATOR
# =========================================================
def run_damped_oscillator(sim_time, time_window):
    st.header("Damped Oscillator")
    st.markdown(
        r"""
This section models a damped spring–mass system:
\[
m\ddot{x}+c\dot{x}+kx=0
\]
"""
    )

    st.sidebar.markdown("### Damped Oscillator Parameters")

    m = st.sidebar.slider("Mass m (kg)", 0.1, 10.0, 1.0, 0.1, key=_k("damp_m"))
    k = st.sidebar.slider("Spring constant k (N/m)", 0.5, 50.0, 10.0, 0.5, key=_k("damp_k"))
    c = st.sidebar.slider("Damping coefficient c (kg/s)", 0.0, 20.0, 1.0, 0.1, key=_k("damp_c"))
    x0 = st.sidebar.slider("Initial displacement x₀ (m)", -5.0, 5.0, 1.0, 0.1, key=_k("damp_x0"))
    v0 = st.sidebar.slider("Initial velocity v₀ (m/s)", -10.0, 10.0, 0.0, 0.1, key=_k("damp_v0"))

    tau, t_hist = progressive_time_array(sim_time, time_window, density=800)

    sol = solve_ode_system(
    damped_derivs,
    np.array([x0, v0]),
    t_hist,
    m,
    c,
    k
    ) 

    x_hist = sol[:, 0]
    v_hist = sol[:, 1]
    a_hist = -(c / m) * v_hist - (k / m) * x_hist

    x_now = x_hist[-1]
    v_now = v_hist[-1]
    a_now = a_hist[-1]

    ke_hist = 0.5 * m * v_hist**2
    pe_hist = 0.5 * k * x_hist**2
    total_e_hist = ke_hist + pe_hist

    ke_now = ke_hist[-1]
    pe_now = pe_hist[-1]
    total_e_now = total_e_hist[-1]

    omega0 = np.sqrt(k / m)
    gamma = c / (2 * m)

    if np.isclose(gamma, omega0, atol=1e-3):
        regime = "Critically damped"
    elif gamma < omega0:
        regime = "Underdamped"
    else:
        regime = "Overdamped"

    q1, q2, q3 = st.columns(3)
    q1.metric("Natural frequency ω₀", f"{omega0:.3f} rad/s")
    q2.metric("Damping factor γ", f"{gamma:.3f} s⁻¹")
    q3.metric("Regime", regime)

    r1, r2, r3 = st.columns(3)
    r1.metric("Displacement x", f"{x_now:.3f} m")
    r2.metric("Velocity v", f"{v_now:.3f} m/s")
    r3.metric("Acceleration a", f"{a_now:.3f} m/s²")

    r4, r5, r6 = st.columns(3)
    r4.metric("Kinetic energy", f"{ke_now:.3f} J")
    r5.metric("Potential energy", f"{pe_now:.3f} J")
    r6.metric("Total energy", f"{total_e_now:.3f} J")

    A_plot = max(abs(x0), np.max(np.abs(x_hist)), 1.0)

    col_left, col_right = st.columns(2)

    with col_left:
        fig_snap, ax_snap = plt.subplots(figsize=(8, 2.8))
        draw_spring_mass(ax_snap, x_now, A_plot, title="Damped Spring–Mass Motion")
        st.pyplot(fig_snap, width="stretch")

    with col_right:
        fig_x, ax_x = plt.subplots(figsize=(8, 3.0))
        ax_x.plot(t_hist, x_hist, linewidth=2)
        ax_x.scatter([tau], [x_now], s=50)
        ax_x.set_xlim(0, time_window)
        style_axis(ax_x, "Time (s)", "Displacement x (m)", "Displacement")
        st.pyplot(fig_x, width="stretch")

    fig_v, ax_v = plt.subplots(figsize=(10, 3.2))
    ax_v.plot(t_hist, v_hist, linewidth=2)
    ax_v.scatter([tau], [v_now], s=45)
    ax_v.set_xlim(0, time_window)
    style_axis(ax_v, "Time (s)", "Velocity v (m/s)", "Velocity")
    st.pyplot(fig_v, width="stretch")

    fig_e, ax_e = plt.subplots(figsize=(10, 3.5))
    ax_e.plot(t_hist, ke_hist, label="Kinetic Energy")
    ax_e.plot(t_hist, pe_hist, label="Potential Energy")
    ax_e.plot(t_hist, total_e_hist, label="Total Energy")
    ax_e.scatter([tau], [ke_now], s=35)
    ax_e.scatter([tau], [pe_now], s=35)
    ax_e.scatter([tau], [total_e_now], s=35)
    ax_e.set_xlim(0, time_window)
    style_axis(ax_e, "Time (s)", "Energy (J)", "Energy Decay")
    ax_e.legend()
    st.pyplot(fig_e, width="stretch")

    with st.expander("Show equations", expanded=False):
        st.latex(r"m\ddot{x}+c\dot{x}+kx=0")
        st.latex(r"\omega_0=\sqrt{\frac{k}{m}},\quad \gamma=\frac{c}{2m}")
        st.latex(r"KE=\frac{1}{2}mv^2,\quad PE=\frac{1}{2}kx^2")
        st.latex(r"\gamma<\omega_0 \Rightarrow \text{underdamped}")
        st.latex(r"\gamma=\omega_0 \Rightarrow \text{critically damped}")
        st.latex(r"\gamma>\omega_0 \Rightarrow \text{overdamped}")


# =========================================================
# MAIN ENTRY
# =========================================================
def run():
    ensure_module_state()
    render_sidebar_controls()
    update_simulation_time()

    st.title("Oscillations Lab")
    st.markdown("Explore oscillatory motion with clean, section-based animation controls.")

    render_top_controls()

    sim_time = get_sim_time()
    time_window = st.session_state[_k("time_window")]
    refresh_rate = st.session_state[_k("refresh_rate")]
    mode = st.session_state[_k("mode")]

    if mode == "Spring–Mass":
        run_spring_mass(sim_time, time_window)
    elif mode == "Simple Pendulum":
        run_simple_pendulum(sim_time, time_window)
    elif mode == "Damped Oscillator":
        run_damped_oscillator(sim_time, time_window)

    if st.session_state[_k("playing")]:
        time.sleep(refresh_rate)
        st.rerun()