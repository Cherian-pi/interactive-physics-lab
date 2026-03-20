import time
import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# ============================================================
# Double Slit Experiment Module
# ============================================================

MODULE_PREFIX = "ds"


# ============================================================
# State helpers
# ============================================================

def _k(name: str) -> str:
    return f"{MODULE_PREFIX}_{name}"


def init_state():
    defaults = {
        _k("playing"): False,
        _k("sim_time"): 0.0,
        _k("last_update"): time.time(),
        _k("hits"): [],
        _k("rng_seed"): 42,
        _k("last_params_signature"): None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_simulation():
    st.session_state[_k("sim_time")] = 0.0
    st.session_state[_k("hits")] = []
    st.session_state[_k("last_update")] = time.time()


def update_simulation_time():
    now = time.time()
    last = st.session_state[_k("last_update")]
    dt = max(0.0, now - last)

    if st.session_state[_k("playing")]:
        st.session_state[_k("sim_time")] += dt

    st.session_state[_k("last_update")] = now
    return dt


# ============================================================
# Physics helpers
# ============================================================

def nm_to_m(x_nm: float) -> float:
    return x_nm * 1e-9


def um_to_m(x_um: float) -> float:
    return x_um * 1e-6


def mm_to_m(x_mm: float) -> float:
    return x_mm * 1e-3


def safe_sinc(z):
    # np.sinc(y) = sin(pi y)/(pi y)
    return np.sinc(z / np.pi)


def screen_grid(screen_half_width_mm: float, n_points: int = 1400):
    x_m = np.linspace(
        -mm_to_m(screen_half_width_mm),
        mm_to_m(screen_half_width_mm),
        n_points,
    )
    return x_m


def single_slit_envelope(x, wavelength_m, slit_width_m, screen_distance_m):
    beta = np.pi * slit_width_m * x / (wavelength_m * screen_distance_m)
    return safe_sinc(beta) ** 2


def two_slit_intensity(x, wavelength_m, slit_sep_m, slit_width_m, screen_distance_m):
    alpha = np.pi * slit_sep_m * x / (wavelength_m * screen_distance_m)
    envelope = single_slit_envelope(
        x=x,
        wavelength_m=wavelength_m,
        slit_width_m=slit_width_m,
        screen_distance_m=screen_distance_m,
    )
    interference = np.cos(alpha) ** 2
    return envelope * interference


def one_slit_intensity(x, wavelength_m, slit_width_m, screen_distance_m):
    return single_slit_envelope(
        x=x,
        wavelength_m=wavelength_m,
        slit_width_m=slit_width_m,
        screen_distance_m=screen_distance_m,
    )


def normalize_distribution(y):
    y = np.clip(y, 0.0, None)
    total = np.sum(y)
    if total <= 0:
        return np.ones_like(y) / len(y)
    return y / total


def fringe_spacing_mm(wavelength_m, slit_sep_m, screen_distance_m):
    # delta x ≈ λL/d
    if slit_sep_m <= 0:
        return None
    return (wavelength_m * screen_distance_m / slit_sep_m) * 1e3


def parameter_signature(*args):
    return tuple(args)


# ============================================================
# Sampling helpers
# ============================================================

def generate_hits(x_grid_m, prob, n_new, seed):
    if n_new <= 0:
        return []
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(x_grid_m), size=n_new, p=prob)
    return x_grid_m[indices].tolist()


def maybe_regenerate_if_parameters_changed(signature):
    if st.session_state[_k("last_params_signature")] != signature:
        st.session_state[_k("last_params_signature")] = signature
        reset_simulation()


# ============================================================
# Plot helpers
# ============================================================

def build_geometry_figure(
    wavelength_nm,
    slit_sep_um,
    slit_width_um,
    screen_distance_m,
    screen_half_width_mm,
    mode_label,
):
    fig, ax = plt.subplots(figsize=(10, 2.8))

    source_x = 0.08
    barrier_x = 0.42
    screen_x = 0.92

    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.axis("off")

    # Source
    ax.scatter([source_x], [0], s=90)
    ax.text(source_x, -0.18, "Source", ha="center", fontsize=10)

    # Barrier
    ax.plot([barrier_x, barrier_x], [-0.9, -0.25], linewidth=4)
    ax.plot([barrier_x, barrier_x], [0.25, 0.9], linewidth=4)

    slit_sep_vis = 0.30
    slit_width_vis = 0.10
    y1 = +slit_sep_vis / 2
    y2 = -slit_sep_vis / 2

    if mode_label == "One slit":
        # Only upper slit shown as open
        ax.plot([barrier_x, barrier_x], [-0.9, y1 - slit_width_vis / 2], linewidth=4)
        ax.plot([barrier_x, barrier_x], [y1 + slit_width_vis / 2, 0.9], linewidth=4)
        ax.plot([barrier_x, barrier_x], [y1 - slit_width_vis / 2, y1 + slit_width_vis / 2],
                linewidth=8, solid_capstyle="butt")
        slit_centers = [y1]
    else:
        # Two slits
        ax.plot([barrier_x, barrier_x], [-0.9, y2 - slit_width_vis / 2], linewidth=4)
        ax.plot([barrier_x, barrier_x], [y2 + slit_width_vis / 2, y1 - slit_width_vis / 2], linewidth=4)
        ax.plot([barrier_x, barrier_x], [y1 + slit_width_vis / 2, 0.9], linewidth=4)

        ax.plot([barrier_x, barrier_x], [y1 - slit_width_vis / 2, y1 + slit_width_vis / 2],
                linewidth=8, solid_capstyle="butt")
        ax.plot([barrier_x, barrier_x], [y2 - slit_width_vis / 2, y2 + slit_width_vis / 2],
                linewidth=8, solid_capstyle="butt")
        slit_centers = [y1, y2]

    # Screen
    ax.plot([screen_x, screen_x], [-0.85, 0.85], linewidth=5)
    ax.text(screen_x, -0.18, "Screen", ha="center", fontsize=10)

    # Rays
    for sy in slit_centers:
        ax.plot([source_x, barrier_x], [0, sy], linestyle="--", linewidth=1.2)
        ax.plot([barrier_x, screen_x], [sy, 0], linestyle="--", linewidth=1.2)

    ax.text(
        0.5,
        1.05,
        (
            f"{mode_label} • λ = {wavelength_nm:.0f} nm   "
            f"d = {slit_sep_um:.1f} µm   "
            f"a = {slit_width_um:.1f} µm   "
            f"L = {screen_distance_m:.2f} m   "
            f"screen ±{screen_half_width_mm:.1f} mm"
        ),
        ha="center",
        va="bottom",
        fontsize=11,
        transform=ax.transAxes,
    )

    return fig


def build_intensity_figure(
    x_grid_m,
    intensity,
    envelope,
    show_envelope,
    mode_label,
):
    fig, ax = plt.subplots(figsize=(10, 3.4))

    x_mm = x_grid_m * 1e3
    ax.plot(x_mm, intensity, linewidth=2.2, label="Intensity")
    if show_envelope and mode_label == "Two slits":
        ax.plot(x_mm, envelope, linestyle="--", linewidth=1.5, label="Single-slit envelope")

    ax.set_xlabel("Screen position x (mm)")
    ax.set_ylabel("Relative intensity")
    ax.set_title("Interference Pattern")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def build_buildup_figure(
    hits_m,
    screen_half_width_mm,
    expected_curve_x_m,
    expected_prob,
    show_expected_curve=True,
):
    fig, ax = plt.subplots(figsize=(10, 4.0))

    ax.set_xlim(-screen_half_width_mm, screen_half_width_mm)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Screen position x (mm)")
    ax.set_ylabel("Detection buildup")
    ax.set_title("Quantum Detection Buildup")

    if len(hits_m) > 0:
        hits_mm = np.array(hits_m) * 1e3
        rng = np.random.default_rng(12345)
        y_jitter = rng.uniform(0.03, 0.95, size=len(hits_mm))
        ax.scatter(hits_mm, y_jitter, s=10, alpha=0.65, label="Detected hits")

    if show_expected_curve:
        xp = expected_curve_x_m * 1e3
        yp = expected_prob / np.max(expected_prob)
        ax.plot(xp, yp, linewidth=2.0, label="Expected probability profile")

    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig


# ============================================================
# UI helpers
# ============================================================

def compact_formula_box(mode_label):
    if mode_label == "Two slits":
        st.latex(
            r"I(x) \propto \cos^2\left(\frac{\pi d x}{\lambda L}\right)\,\mathrm{sin}^2\left(\frac{\pi a x}{\lambda L}\right)"
        )
    else:
        st.latex(
            r"I(x) \propto \mathrm{sin}^2\left(\frac{\pi a x}{\lambda L}\right)"
        )


def interpretation_box(wavelength_nm, slit_sep_um, slit_width_um, screen_distance_m, mode_label):
    lam = nm_to_m(wavelength_nm)
    d = um_to_m(slit_sep_um)
    a = um_to_m(slit_width_um)
    L = screen_distance_m

    dx = fringe_spacing_mm(lam, d, L) if mode_label == "Two slits" else None

    notes = []
    if mode_label == "Two slits" and dx is not None:
        notes.append(f"Estimated fringe spacing ≈ **{dx:.3f} mm**.")
    notes.append("Increasing **wavelength** spreads the pattern outward.")
    if mode_label == "Two slits":
        notes.append("Increasing **slit separation** makes fringes closer together.")
    notes.append("Increasing **slit width** narrows the overall diffraction envelope.")
    notes.append("Increasing **screen distance** spreads the pattern across the screen.")

    st.markdown("### Physical interpretation")
    for line in notes:
        st.markdown(f"- {line}")


# ============================================================
# Main module
# ============================================================

def run():
    init_state()
    dt = update_simulation_time()

    st.title("Double Slit Experiment")
    st.caption(
        "Explore how slit geometry shapes the interference pattern and how discrete detections build up the quantum distribution."
    )

    left_col, right_col = st.columns([1.05, 2.2], gap="large")

    with left_col:
        st.subheader("Controls")

        mode_label = st.radio(
            "Experiment mode",
            ["Two slits", "One slit"],
            key=_k("mode"),
            horizontal=True,
        )

        st.markdown("#### Source and slit parameters")
        wavelength_nm = st.slider(
            "Wavelength λ (nm)",
            min_value=350,
            max_value=800,
            value=550,
            step=5,
            key=_k("wavelength_nm"),
        )

        slit_sep_um = st.slider(
            "Slit separation d (µm)",
            min_value=10.0,
            max_value=200.0,
            value=60.0,
            step=1.0,
            key=_k("slit_sep_um"),
        )

        slit_width_um = st.slider(
            "Slit width a (µm)",
            min_value=2.0,
            max_value=60.0,
            value=12.0,
            step=0.5,
            key=_k("slit_width_um"),
        )

        screen_distance_m = st.slider(
            "Screen distance L (m)",
            min_value=0.2,
            max_value=5.0,
            value=1.5,
            step=0.1,
            key=_k("screen_distance_m"),
        )

        screen_half_width_mm = st.slider(
            "Screen half-width (mm)",
            min_value=2.0,
            max_value=30.0,
            value=10.0,
            step=0.5,
            key=_k("screen_half_width_mm"),
        )

        st.markdown("#### Detection controls")
        arrival_rate = st.slider(
            "Arrival rate (hits/s)",
            min_value=1,
            max_value=200,
            value=30,
            step=1,
            key=_k("arrival_rate"),
        )

        max_hits = st.slider(
            "Maximum recorded hits",
            min_value=50,
            max_value=5000,
            value=800,
            step=50,
            key=_k("max_hits"),
        )

        st.markdown("#### Display options")
        show_envelope = st.checkbox(
            "Show envelope",
            value=True,
            key=_k("show_envelope"),
        )
        show_expected_curve = st.checkbox(
            "Show expected probability profile",
            value=True,
            key=_k("show_expected_curve"),
        )
        show_theory = st.checkbox(
            "Show theory panel",
            value=True,
            key=_k("show_theory"),
        )

        st.markdown("#### Simulation")
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("Play", use_container_width=True, key=_k("play_btn")):
                st.session_state[_k("playing")] = True
                st.session_state[_k("last_update")] = time.time()
        with b2:
            if st.button("Pause", use_container_width=True, key=_k("pause_btn")):
                st.session_state[_k("playing")] = False
        with b3:
            if st.button("Reset", use_container_width=True, key=_k("reset_btn")):
                st.session_state[_k("playing")] = False
                reset_simulation()

        compact_formula_box(mode_label)

    wavelength_m = nm_to_m(wavelength_nm)
    slit_sep_m = um_to_m(slit_sep_um)
    slit_width_m = um_to_m(slit_width_um)

    signature = parameter_signature(
        mode_label,
        wavelength_nm,
        slit_sep_um,
        slit_width_um,
        screen_distance_m,
        screen_half_width_mm,
        max_hits,
        arrival_rate,
    )
    maybe_regenerate_if_parameters_changed(signature)

    x_grid_m = screen_grid(screen_half_width_mm=screen_half_width_mm, n_points=1400)

    envelope = single_slit_envelope(
        x=x_grid_m,
        wavelength_m=wavelength_m,
        slit_width_m=slit_width_m,
        screen_distance_m=screen_distance_m,
    )

    if mode_label == "Two slits":
        intensity = two_slit_intensity(
            x=x_grid_m,
            wavelength_m=wavelength_m,
            slit_sep_m=slit_sep_m,
            slit_width_m=slit_width_m,
            screen_distance_m=screen_distance_m,
        )
    else:
        intensity = one_slit_intensity(
            x=x_grid_m,
            wavelength_m=wavelength_m,
            slit_width_m=slit_width_m,
            screen_distance_m=screen_distance_m,
        )

    prob = normalize_distribution(intensity)

    # Add new hits while playing
    if st.session_state[_k("playing")]:
        n_new = int(arrival_rate * dt)
        n_remaining = max_hits - len(st.session_state[_k("hits")])
        n_new = max(0, min(n_new, n_remaining))

        if n_new > 0:
            seed = st.session_state[_k("rng_seed")] + len(st.session_state[_k("hits")])
            new_hits = generate_hits(
                x_grid_m=x_grid_m,
                prob=prob,
                n_new=n_new,
                seed=seed,
            )
            st.session_state[_k("hits")].extend(new_hits)

        if len(st.session_state[_k("hits")]) >= max_hits:
            st.session_state[_k("playing")] = False

    with right_col:
        top_a, top_b, top_c = st.columns(3)
        with top_a:
            st.metric("Recorded hits", f"{len(st.session_state[_k('hits')])}")
        with top_b:
            st.metric("Mode", mode_label)
        with top_c:
            if mode_label == "Two slits":
                dx = fringe_spacing_mm(wavelength_m, slit_sep_m, screen_distance_m)
                st.metric("Fringe spacing", f"{dx:.3f} mm" if dx is not None else "—")
            else:
                st.metric("Fringe spacing", "Not applicable")

        geom_fig = build_geometry_figure(
            wavelength_nm=wavelength_nm,
            slit_sep_um=slit_sep_um,
            slit_width_um=slit_width_um,
            screen_distance_m=screen_distance_m,
            screen_half_width_mm=screen_half_width_mm,
            mode_label=mode_label,
        )
        st.pyplot(geom_fig, clear_figure=True)

        intensity_fig = build_intensity_figure(
            x_grid_m=x_grid_m,
            intensity=intensity,
            envelope=envelope,
            show_envelope=show_envelope,
            mode_label=mode_label,
        )
        st.pyplot(intensity_fig, clear_figure=True)

        buildup_fig = build_buildup_figure(
            hits_m=st.session_state[_k("hits")],
            screen_half_width_mm=screen_half_width_mm,
            expected_curve_x_m=x_grid_m,
            expected_prob=prob,
            show_expected_curve=show_expected_curve,
        )
        st.pyplot(buildup_fig, clear_figure=True)

        if show_theory:
            interpretation_box(
                wavelength_nm=wavelength_nm,
                slit_sep_um=slit_sep_um,
                slit_width_um=slit_width_um,
                screen_distance_m=screen_distance_m,
                mode_label=mode_label,
            )

    # Controlled rerun for animation
    if st.session_state[_k("playing")]:
        time.sleep(0.05)
        st.rerun()