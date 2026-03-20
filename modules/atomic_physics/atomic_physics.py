import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st


RYDBERG_EV = 13.6
PLANCK_EV_S = 4.135667696e-15
HC_EV_NM = 1239.841984
MAX_N = 12
ANIMATION_DT = 0.05


# -----------------------------
# Core physics
# -----------------------------
def hydrogen_energy(n: int, z: int = 1) -> float:
    return -(RYDBERG_EV * z * z) / (n ** 2)


def transition_energy(n_initial: int, n_final: int, z: int = 1) -> float:
    return abs(hydrogen_energy(n_final, z) - hydrogen_energy(n_initial, z))


def wavelength_nm_from_ev(energy_ev: float) -> float:
    if energy_ev <= 0:
        return np.inf
    return HC_EV_NM / energy_ev


def frequency_hz_from_ev(energy_ev: float) -> float:
    return energy_ev / PLANCK_EV_S


def classify_series(n_final: int) -> str:
    mapping = {
        1: "Lyman",
        2: "Balmer",
        3: "Paschen",
        4: "Brackett",
        5: "Pfund",
    }
    return mapping.get(n_final, f"Series ending at n={n_final}")


def region_from_wavelength(wavelength_nm: float) -> str:
    if wavelength_nm < 380:
        return "Ultraviolet"
    if wavelength_nm < 450:
        return "Visible (violet-blue)"
    if wavelength_nm < 495:
        return "Visible (blue-cyan)"
    if wavelength_nm < 570:
        return "Visible (green)"
    if wavelength_nm < 590:
        return "Visible (yellow)"
    if wavelength_nm < 620:
        return "Visible (orange)"
    if wavelength_nm <= 750:
        return "Visible (red)"
    return "Infrared"


def rgba_from_wavelength(wavelength_nm: float, alpha: float = 1.0):
    # Approximate visible-spectrum mapping for educational display.
    if wavelength_nm < 380:
        return (0.55, 0.45, 0.8, 0.35 * alpha)
    if wavelength_nm > 750:
        return (0.6, 0.25, 0.25, 0.35 * alpha)

    gamma = 0.8
    if 380 <= wavelength_nm < 440:
        r = -(wavelength_nm - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif 440 <= wavelength_nm < 490:
        r = 0.0
        g = (wavelength_nm - 440) / (490 - 440)
        b = 1.0
    elif 490 <= wavelength_nm < 510:
        r = 0.0
        g = 1.0
        b = -(wavelength_nm - 510) / (510 - 490)
    elif 510 <= wavelength_nm < 580:
        r = (wavelength_nm - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif 580 <= wavelength_nm < 645:
        r = 1.0
        g = -(wavelength_nm - 645) / (645 - 580)
        b = 0.0
    else:
        r = 1.0
        g = 0.0
        b = 0.0

    if 380 <= wavelength_nm < 420:
        factor = 0.3 + 0.7 * (wavelength_nm - 380) / (420 - 380)
    elif 420 <= wavelength_nm <= 700:
        factor = 1.0
    else:
        factor = 0.3 + 0.7 * (750 - wavelength_nm) / (750 - 700)

    r = (r * factor) ** gamma
    g = (g * factor) ** gamma
    b = (b * factor) ** gamma
    return (r, g, b, alpha)


# -----------------------------
# Session state helpers
# -----------------------------
def init_atomic_state():
    defaults = {
        "atomic_anim_playing": False,
        "atomic_anim_progress": 0.0,
        "atomic_last_update": time.time(),
        "atomic_excited_level": 1,
        "atomic_flash_strength": 0.0,
        "atomic_last_emitted_wavelength": None,
        "atomic_last_emitted_energy": None,
        "atomic_last_emitted_path": "Ground state",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def atomic_play():
    st.session_state.atomic_anim_playing = True
    st.session_state.atomic_last_update = time.time()


def atomic_pause():
    st.session_state.atomic_anim_playing = False


def atomic_reset():
    st.session_state.atomic_anim_playing = False
    st.session_state.atomic_anim_progress = 0.0
    st.session_state.atomic_last_update = time.time()
    st.session_state.atomic_flash_strength = 0.0


def excite_atom(max_level: int):
    st.session_state.atomic_excited_level = min(max_level, st.session_state.atomic_excited_level + 1)


def drop_one_level():
    current = st.session_state.atomic_excited_level
    if current > 1:
        next_level = current - 1
        dE = transition_energy(current, next_level)
        st.session_state.atomic_last_emitted_energy = dE
        st.session_state.atomic_last_emitted_wavelength = wavelength_nm_from_ev(dE)
        st.session_state.atomic_last_emitted_path = f"{current} → {next_level}"
        st.session_state.atomic_excited_level = next_level
        st.session_state.atomic_flash_strength = 1.0


def relax_to_ground():
    current = st.session_state.atomic_excited_level
    if current <= 1:
        return
    path = []
    first_wavelength = None
    total_energy = 0.0
    while current > 1:
        next_level = current - 1
        dE = transition_energy(current, next_level)
        wl = wavelength_nm_from_ev(dE)
        if first_wavelength is None:
            first_wavelength = wl
        total_energy += dE
        path.append(f"{current} → {next_level}")
        current = next_level
    st.session_state.atomic_excited_level = 1
    st.session_state.atomic_last_emitted_wavelength = first_wavelength
    st.session_state.atomic_last_emitted_energy = total_energy
    st.session_state.atomic_last_emitted_path = ", ".join(path)
    st.session_state.atomic_flash_strength = 1.0


# -----------------------------
# Plot helpers
# -----------------------------
def plot_bohr_orbits(max_n: int, electron_n: int, z: int = 1, flash_strength: float = 0.0, photon_wavelength: float | None = None):
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    ax.set_aspect("equal")
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")

    nucleus = patches.Circle((0, 0), 0.18, color="#f8fafc", ec="#f59e0b", lw=2)
    ax.add_patch(nucleus)
    ax.text(0, 0, f"+{z}", color="#f59e0b", ha="center", va="center", fontsize=12, fontweight="bold")

    radii = np.linspace(0.55, 2.6, max_n)
    theta = np.linspace(0, 2 * np.pi, 400)
    for idx, radius in enumerate(radii, start=1):
        lw = 2.3 if idx == electron_n else 1.2
        alpha = 0.9 if idx == electron_n else 0.3
        ax.plot(radius * np.cos(theta), radius * np.sin(theta), color="#93c5fd", lw=lw, alpha=alpha)
        ax.text(radius / math.sqrt(2), radius / math.sqrt(2), f"n={idx}", color="#e2e8f0", fontsize=9)

    e_radius = radii[electron_n - 1]
    electron = patches.Circle((e_radius, 0), 0.09, color="#22d3ee", ec="white", lw=1.5)
    ax.add_patch(electron)

    if flash_strength > 0 and photon_wavelength is not None:
        rgba = rgba_from_wavelength(photon_wavelength, alpha=min(1.0, 0.35 + 0.55 * flash_strength))
        for i in range(3):
            y = 0.4 - i * 0.22
            x = np.linspace(1.0, 3.25, 250)
            amp = 0.06 + 0.02 * i
            y_wave = y + amp * np.sin(10 * x)
            ax.plot(x, y_wave, color=rgba, lw=2.0 + i * 0.6)
        ax.text(2.2, 0.78, f"Photon: {photon_wavelength:.1f} nm", color=rgba[:3], fontsize=10, ha="center")

    ax.set_xlim(-3.2, 3.6)
    ax.set_ylim(-3.0, 3.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Bohr-Orbit View", color="white", pad=12)
    return fig


def plot_energy_levels(max_n: int, n_initial: int, n_final: int, progress: float, mode: str, z: int = 1):
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    energies = [hydrogen_energy(n, z) for n in range(1, max_n + 1)]

    for n, energy in enumerate(energies, start=1):
        alpha = 1.0 if n in (n_initial, n_final) else 0.45
        lw = 3 if n in (n_initial, n_final) else 1.8
        ax.hlines(energy, 0.12, 0.88, lw=lw, alpha=alpha)
        ax.text(0.9, energy, f"n={n}  ({energy:.2f} eV)", va="center", fontsize=9)

    e_start = hydrogen_energy(n_initial, z)
    e_end = hydrogen_energy(n_final, z)
    current_energy = e_start + progress * (e_end - e_start)

    if mode == "Emission":
        transition_color = "tab:red"
        electron_color = "gold"
        label = f"Emission: n={n_initial} → n={n_final}"
    else:
        transition_color = "tab:blue"
        electron_color = "deepskyblue"
        label = f"Absorption: n={n_initial} → n={n_final}"

    ax.annotate(
        "",
        xy=(0.5, e_end),
        xytext=(0.5, e_start),
        arrowprops=dict(arrowstyle="->", lw=2.8, color=transition_color, alpha=0.85),
    )
    ax.scatter([0.5], [current_energy], s=180, color=electron_color, edgecolors="black", zorder=5)
    ax.text(0.54, (e_start + e_end) / 2, label, color=transition_color, fontsize=10, va="center")

    if mode == "Emission" and progress > 0.75:
        ax.text(0.18, 0.15, "photon emitted", color="tab:red", fontsize=10, transform=ax.transAxes)
    if mode == "Absorption" and progress > 0.75:
        ax.text(0.18, 0.15, "photon absorbed", color="tab:blue", fontsize=10, transform=ax.transAxes)

    ax.set_xlim(0, 1.08)
    ax.set_ylim(min(energies) - 1.0, 0.8)
    ax.set_xticks([])
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Hydrogen Energy-Level Transition")
    ax.grid(alpha=0.25)
    return fig


def plot_spectrum_lines(n_final: int, max_n: int, highlight_n_initial: int | None = None, z: int = 1):
    fig, ax = plt.subplots(figsize=(10.0, 3.6))

    visible_gradient = np.linspace(380, 750, 700)
    band = np.array([[rgba_from_wavelength(w)[:3] for w in visible_gradient]])
    ax.imshow(band, extent=(380, 750, 0, 0.22), aspect="auto", origin="lower")
    ax.text(565, 0.11, "Visible band", color="black", ha="center", va="center", fontsize=9)

    finite_wavelengths = []
    for n_initial in range(n_final + 1, max_n + 1):
        dE = transition_energy(n_initial, n_final, z)
        wl = wavelength_nm_from_ev(dE)
        finite_wavelengths.append(wl)
        is_highlight = n_initial == highlight_n_initial
        color = rgba_from_wavelength(wl, alpha=1.0)
        lw = 3.3 if is_highlight else 2.1
        ymax = 0.96 if is_highlight else 0.8
        ax.vlines(wl, 0.28, ymax, linewidth=lw, color=color)
        ax.text(wl, ymax + 0.03, f"{n_initial}→{n_final}", rotation=90, ha="center", va="bottom", fontsize=8)

    if finite_wavelengths:
        xmin = max(50, min(finite_wavelengths) * 0.9)
        xmax = min(2600, max(finite_wavelengths) * 1.1)
        ax.set_xlim(xmin, xmax)

    ax.axvspan(0, 380, alpha=0.06)
    ax.axvspan(750, 2600, alpha=0.06)
    ax.text(190, 0.12, "UV", ha="center", fontsize=9)
    ax.text(1450, 0.12, "IR", ha="center", fontsize=9)

    ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    ax.set_xlabel("Wavelength (nm)")
    ax.set_title(f"{classify_series(n_final)} Series")
    ax.grid(alpha=0.2, axis="x")
    return fig


def build_series_rows(n_final: int, max_n: int, z: int = 1):
    rows = []
    for n_initial in range(n_final + 1, max_n + 1):
        dE = transition_energy(n_initial, n_final, z)
        wl = wavelength_nm_from_ev(dE)
        rows.append(
            {
                "Transition": f"{n_initial} → {n_final}",
                "Energy (eV)": round(dE, 4),
                "Wavelength (nm)": round(wl, 2),
                "Region": region_from_wavelength(wl),
            }
        )
    return rows


def render_transition_summary(n_initial: int, n_final: int, mode: str, z: int = 1):
    dE = transition_energy(n_initial, n_final, z)
    wl = wavelength_nm_from_ev(dE)
    freq = frequency_hz_from_ev(dE)
    transition_name = classify_series(min(n_initial, n_final) if mode == "Absorption" else n_final)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Photon energy", f"{dE:.3f} eV")
    c2.metric("Wavelength", f"{wl:.2f} nm")
    c3.metric("Frequency", f"{freq:.3e} Hz")
    c4.metric("Region", region_from_wavelength(wl))

    st.caption(
        f"{mode} transition from n={n_initial} to n={n_final}. "
        f"Series label: {transition_name}."
    )


# -----------------------------
# Main module
# -----------------------------
def run():
    init_atomic_state()

    st.title("⚛️ Atomic Physics Explorer")
    st.markdown(
        """
Explore the **hydrogen atom** through the Bohr model with a more interactive lab-style UI:
- discrete energy levels
- animated absorption and emission transitions
- orbit visualization with photon release
- visible, ultraviolet, and infrared spectral lines
"""
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Bohr Levels",
        "Animated Transition",
        "Excite the Atom",
        "Hydrogen Spectrum",
    ])

    with tab1:
        st.subheader("Bohr Energy Levels")
        c1, c2 = st.columns([1, 1])
        max_n_levels = c1.slider(
            "Maximum principal quantum number",
            min_value=3,
            max_value=MAX_N,
            value=6,
            key="atomic_bohr_max_n",
        )
        selected_orbit = c2.slider(
            "Highlighted orbit",
            min_value=1,
            max_value=max_n_levels,
            value=min(2, max_n_levels),
            key="atomic_bohr_selected_n",
        )

        left, right = st.columns([1.05, 1.0])
        with left:
            fig = plot_bohr_orbits(max_n_levels, selected_orbit)
            st.pyplot(fig, clear_figure=True)
        with right:
            fig, ax = plt.subplots(figsize=(7.8, 5.8))
            for n in range(1, max_n_levels + 1):
                energy = hydrogen_energy(n)
                lw = 3 if n == selected_orbit else 1.8
                alpha = 1.0 if n == selected_orbit else 0.45
                ax.hlines(energy, 0.18, 0.82, linewidth=lw, alpha=alpha)
                ax.text(0.84, energy, f"n={n} ({energy:.2f} eV)", va="center", fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_ylim(-14.5, 0.8)
            ax.set_xticks([])
            ax.set_ylabel("Energy (eV)")
            ax.set_title("Bohr Energy Levels for Hydrogen")
            ax.grid(alpha=0.25)
            st.pyplot(fig, clear_figure=True)

        st.info(
            "The Bohr model treats the hydrogen electron as occupying only discrete allowed levels, "
            r"with $E_n = -13.6/n^2$ eV. Higher n means a less tightly bound electron."
        )

    with tab2:
        st.subheader("Animated Single Transition")
        controls_left, controls_mid, controls_right = st.columns([1.15, 1.0, 1.1])

        mode = controls_left.radio(
            "Process",
            ["Emission", "Absorption"],
            key="atomic_transition_mode",
        )
        max_n_transition = controls_mid.slider(
            "Maximum level available",
            min_value=3,
            max_value=MAX_N,
            value=6,
            key="atomic_transition_max_n",
        )
        speed = controls_right.slider(
            "Animation speed",
            min_value=0.25,
            max_value=3.0,
            value=1.0,
            step=0.25,
            key="atomic_transition_speed",
        )

        pick_left, pick_mid, pick_right, pick_far = st.columns([1.0, 1.0, 0.8, 1.2])
        if mode == "Emission":
            n_initial = pick_left.selectbox(
                "Initial level",
                options=list(range(2, max_n_transition + 1)),
                index=min(3, max_n_transition - 2),
                key="atomic_emission_n_initial",
            )
            n_final = pick_mid.selectbox(
                "Final level",
                options=list(range(1, n_initial)),
                index=0,
                key="atomic_emission_n_final",
            )
        else:
            n_initial = pick_left.selectbox(
                "Initial level",
                options=list(range(1, max_n_transition)),
                index=0,
                key="atomic_absorption_n_initial",
            )
            n_final = pick_mid.selectbox(
                "Final level",
                options=list(range(n_initial + 1, max_n_transition + 1)),
                index=0,
                key="atomic_absorption_n_final",
            )

        pick_right.button("▶ Play", key="atomic_btn_play", on_click=atomic_play)
        pick_far.button("⏸ Pause", key="atomic_btn_pause", on_click=atomic_pause)
        reset_col, spacer_col = st.columns([0.24, 0.76])
        reset_col.button("↺ Reset", key="atomic_btn_reset", on_click=atomic_reset)

        if not st.session_state.atomic_anim_playing:
            manual_progress = st.slider(
                "Manual scrub",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.atomic_anim_progress),
                step=0.01,
                key="atomic_transition_scrub",
            )
            st.session_state.atomic_anim_progress = manual_progress

        if st.session_state.atomic_anim_playing:
            now = time.time()
            dt = now - st.session_state.atomic_last_update
            st.session_state.atomic_last_update = now
            st.session_state.atomic_anim_progress = min(
                1.0,
                st.session_state.atomic_anim_progress + dt * 0.35 * speed,
            )
            if st.session_state.atomic_anim_progress >= 1.0:
                st.session_state.atomic_anim_playing = False

        render_transition_summary(n_initial, n_final, mode)

        left, right = st.columns([1.15, 1.0])
        with left:
            fig = plot_energy_levels(
                max_n_transition,
                n_initial,
                n_final,
                st.session_state.atomic_anim_progress,
                mode,
            )
            st.pyplot(fig, clear_figure=True)
        with right:
            orbit_n = max(1, int(round(n_initial + st.session_state.atomic_anim_progress * (n_final - n_initial))))
            show_flash = mode == "Emission" and st.session_state.atomic_anim_progress > 0.78
            photon_wl = wavelength_nm_from_ev(transition_energy(n_initial, n_final))
            fig = plot_bohr_orbits(
                max_n=max_n_transition,
                electron_n=orbit_n,
                flash_strength=1.0 if show_flash else 0.0,
                photon_wavelength=photon_wl if show_flash else None,
            )
            st.pyplot(fig, clear_figure=True)

        if mode == "Emission":
            st.success("The electron falls to a lower level and releases a photon carrying the energy difference.")
        else:
            st.success("The electron absorbs a photon with the right energy and jumps upward to a higher allowed level.")

    with tab3:
        st.subheader("Excite the Atom Step by Step")
        c1, c2 = st.columns([1.0, 1.2])
        excite_max_n = c1.slider(
            "Highest orbit allowed in this demo",
            min_value=3,
            max_value=9,
            value=6,
            key="atomic_excite_max_n",
        )
        current_level = min(st.session_state.atomic_excited_level, excite_max_n)
        st.session_state.atomic_excited_level = current_level

        b1, b2, b3 = c2.columns(3)
        b1.button("Add energy", key="atomic_excite_button", on_click=excite_atom, args=(excite_max_n,))
        b2.button("Drop one level", key="atomic_drop_button", on_click=drop_one_level)
        b3.button("Relax to ground", key="atomic_relax_button", on_click=relax_to_ground)

        left, right = st.columns([1.0, 1.0])
        with left:
            fig = plot_bohr_orbits(
                max_n=excite_max_n,
                electron_n=st.session_state.atomic_excited_level,
                flash_strength=st.session_state.atomic_flash_strength,
                photon_wavelength=st.session_state.atomic_last_emitted_wavelength,
            )
            st.pyplot(fig, clear_figure=True)
        with right:
            st.metric("Current electron level", f"n = {st.session_state.atomic_excited_level}")
            st.metric(
                "Current level energy",
                f"{hydrogen_energy(st.session_state.atomic_excited_level):.3f} eV",
            )
            if st.session_state.atomic_last_emitted_wavelength is not None:
                st.metric("Last emitted photon", f"{st.session_state.atomic_last_emitted_wavelength:.2f} nm")
                st.metric("Released energy", f"{st.session_state.atomic_last_emitted_energy:.3f} eV")
                st.write(f"**Latest path:** {st.session_state.atomic_last_emitted_path}")
                st.write(f"**Region:** {region_from_wavelength(st.session_state.atomic_last_emitted_wavelength)}")
            else:
                st.write("No photon has been emitted yet. Raise the electron and then let it fall.")

        st.info(
            "This panel makes the energy-level idea more physical: adding energy pushes the electron upward, "
            "while falling back down releases one or more photons."
        )

        if st.session_state.atomic_flash_strength > 0:
            st.session_state.atomic_flash_strength = max(0.0, st.session_state.atomic_flash_strength - 0.12)

    with tab4:
        st.subheader("Hydrogen Spectral Series")
        c1, c2, c3 = st.columns(3)
        n_final_series = c1.selectbox(
            "Final level of the series",
            options=[1, 2, 3, 4, 5],
            index=1,
            key="atomic_series_n_final",
        )
        max_n_series = c2.slider(
            "Highest starting level",
            min_value=n_final_series + 1,
            max_value=MAX_N,
            value=max(n_final_series + 4, n_final_series + 1),
            key="atomic_series_max_n",
        )
        highlight = c3.selectbox(
            "Highlight transition",
            options=[None] + list(range(n_final_series + 1, max_n_series + 1)),
            format_func=lambda x: "None" if x is None else f"{x} → {n_final_series}",
            key="atomic_series_highlight",
        )

        st.write(f"**Series:** {classify_series(n_final_series)}")
        fig = plot_spectrum_lines(n_final_series, max_n_series, highlight_n_initial=highlight)
        st.pyplot(fig, clear_figure=True)
        st.dataframe(build_series_rows(n_final_series, max_n_series), width="stretch")
        st.info(
            "Balmer lines fall in or near the visible range, whereas Lyman lies mainly in the ultraviolet and "
            "Paschen shifts into the infrared."
        )

    if st.session_state.atomic_anim_playing:
        time.sleep(ANIMATION_DT)
        st.rerun()
