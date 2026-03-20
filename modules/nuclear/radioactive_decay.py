import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


ISOTOPES = {
    "Carbon-14": {
        "symbol": "C-14",
        "parent_protons": 6,
        "parent_neutrons": 8,
        "daughter_symbol": "N-14",
        "daughter_protons": 7,
        "daughter_neutrons": 7,
        "half_life": 8.0,
        "decay_mode": "Beta",
    },
    "Uranium-238": {
        "symbol": "U-238",
        "parent_protons": 92,
        "parent_neutrons": 146,
        "daughter_symbol": "Th-234",
        "daughter_protons": 90,
        "daughter_neutrons": 144,
        "half_life": 12.0,
        "decay_mode": "Alpha",
    },
    "Cobalt-60": {
        "symbol": "Co-60",
        "parent_protons": 27,
        "parent_neutrons": 33,
        "daughter_symbol": "Ni-60",
        "daughter_protons": 28,
        "daughter_neutrons": 32,
        "half_life": 6.0,
        "decay_mode": "Beta + Gamma",
    },
    "Radon-222": {
        "symbol": "Rn-222",
        "parent_protons": 86,
        "parent_neutrons": 136,
        "daughter_symbol": "Po-218",
        "daughter_protons": 84,
        "daughter_neutrons": 134,
        "half_life": 5.0,
        "decay_mode": "Alpha",
    },
}

STATE_KEY = "decay_state"
STYLE_KEY = "decay_style_loaded"
ISOTOPE_KEY = "decay_isotope"
HALF_LIFE_KEY = "decay_half_life"
COUNT_KEY = "decay_display_count"
SPEED_KEY = "decay_speed"


def inject_module_style() -> None:
    if st.session_state.get(STYLE_KEY):
        return
    st.session_state[STYLE_KEY] = True
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(15,23,42,0.96), rgba(30,41,59,0.92));
            border: 1px solid rgba(148,163,184,0.22);
            border-radius: 18px;
            padding: 0.55rem 0.7rem;
            box-shadow: 0 10px 26px rgba(2, 6, 23, 0.18);
        }
        .decay-card {
            background: linear-gradient(180deg, rgba(15,23,42,0.96), rgba(17,24,39,0.94));
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 28px rgba(2,6,23,0.15);
            margin-bottom: 0.85rem;
        }
        .decay-card h4 {
            margin: 0 0 0.55rem 0;
            color: #f8fafc;
            font-size: 1rem;
        }
        .decay-note {
            color: #cbd5e1;
            font-size: 0.94rem;
            line-height: 1.45;
            margin: 0;
        }
        .decay-mini {
            color: #94a3b8;
            font-size: 0.86rem;
            margin-top: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def card(title: str, body: str) -> None:
    st.markdown(
        f"<div class='decay-card'><h4>{title}</h4><p class='decay-note'>{body}</p></div>",
        unsafe_allow_html=True,
    )


def _default_state() -> Dict:
    return {
        "sim_time": 0.0,
        "running": False,
        "last_wall_time": time.time(),
        "time_accumulator": 0.0,
        "history_time": [0.0],
        "history_parent": [],
        "history_daughter": [],
        "alive": None,
        "parent_positions": None,
        "daughter_positions": None,
        "display_positions": None,
        "events": [],
        "glow_timer": 0.0,
        "recoil_offset": np.zeros(2, dtype=float),
        "recoil_velocity": np.zeros(2, dtype=float),
        "selected_isotope": None,
        "display_count": None,
        "half_life": None,
        "theory_n0": None,
        "seed": 42,
    }


def decay_state() -> Dict:
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = _default_state()
    return st.session_state[STATE_KEY]


def generate_cloud_positions(n_points: int, seed: int, radius: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.random(n_points)) * radius
    theta = 2.0 * np.pi * rng.random(n_points)
    return np.column_stack((r * np.cos(theta), r * np.sin(theta)))


def create_emission_package(mode: str, origin: np.ndarray, seed: int) -> List[Dict]:
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    direction = np.array([np.cos(angle), np.sin(angle)], dtype=float)
    packages: List[Dict] = []

    def make_event(label: str, speed: float, size: float, life: float, trail: int, jitter: float, glow: str):
        local_dir = direction.copy()
        if jitter > 0:
            rot = rng.normal(0.0, jitter)
            c, s = np.cos(rot), np.sin(rot)
            local_dir[:] = (c * local_dir[0] - s * local_dir[1], s * local_dir[0] + c * local_dir[1])
        packages.append(
            {
                "x": float(origin[0]),
                "y": float(origin[1]),
                "vx": float(speed * local_dir[0]),
                "vy": float(speed * local_dir[1]),
                "age": 0.0,
                "life": float(life),
                "label": label,
                "size": float(size),
                "trail": int(trail),
                "glow": glow,
                "path_x": [float(origin[0])],
                "path_y": [float(origin[1])],
            }
        )

    if "Alpha" in mode:
        make_event("α", speed=1.05, size=230, life=1.4, trail=10, jitter=0.0, glow="#ff4fd8")
    if "Beta" in mode:
        make_event("β", speed=1.55, size=135, life=1.0, trail=14, jitter=0.08, glow="#70ff70")
    if "Gamma" in mode:
        make_event("γ", speed=1.9, size=90, life=0.8, trail=18, jitter=0.12, glow="#ffffff")

    return packages


def recoil_strength(mode: str) -> float:
    if "Alpha" in mode:
        return 0.13
    if "Beta" in mode:
        return 0.07
    if "Gamma" in mode:
        return 0.03
    return 0.05


def reset_decay_state(isotope_name: str, display_count: int, half_life: float) -> None:
    state = decay_state()
    parent_positions = generate_cloud_positions(display_count, seed=state["seed"], radius=0.92)
    daughter_positions = generate_cloud_positions(display_count, seed=state["seed"] + 1, radius=0.70)

    state.update(
        {
            "sim_time": 0.0,
            "running": False,
            "last_wall_time": time.time(),
            "time_accumulator": 0.0,
            "history_time": [0.0],
            "history_parent": [display_count],
            "history_daughter": [0],
            "alive": np.ones(display_count, dtype=bool),
            "parent_positions": parent_positions,
            "daughter_positions": daughter_positions,
            "display_positions": parent_positions.copy(),
            "events": [],
            "glow_timer": 0.0,
            "recoil_offset": np.zeros(2, dtype=float),
            "recoil_velocity": np.zeros(2, dtype=float),
            "selected_isotope": isotope_name,
            "display_count": display_count,
            "half_life": float(half_life),
            "theory_n0": display_count,
        }
    )


def needs_reset(isotope_name: str, display_count: int, half_life: float) -> bool:
    state = decay_state()
    return (
        state["alive"] is None
        or state["selected_isotope"] != isotope_name
        or state["display_count"] != display_count
        or state["half_life"] != float(half_life)
    )


def apply_decay_step(dt: float, isotope_name: str) -> None:
    state = decay_state()
    data = ISOTOPES[isotope_name]
    alive = state["alive"]
    if alive is None or len(alive) == 0:
        return

    decay_constant = np.log(2.0) / state["half_life"]
    alive_idx = np.flatnonzero(alive)

    if len(alive_idx) > 0:
        p_decay = 1.0 - np.exp(-decay_constant * dt)
        rng = np.random.default_rng(int((state["sim_time"] * 1000) + len(alive_idx) + state["seed"]))
        new_decay_mask = rng.random(len(alive_idx)) < p_decay
        decayed_idx = alive_idx[new_decay_mask]

        if len(decayed_idx) > 0:
            state["alive"][decayed_idx] = False
            origin = np.mean(state["display_positions"][decayed_idx], axis=0)
            state["events"].extend(
                create_emission_package(data["decay_mode"], origin, seed=int(state["sim_time"] * 10000) + len(decayed_idx))
            )
            state["glow_timer"] = min(0.45, state["glow_timer"] + 0.30)
            kick = recoil_strength(data["decay_mode"])
            angle = np.arctan2(origin[1] + 1e-9, origin[0] + 1e-9)
            recoil_dir = -np.array([np.cos(angle), np.sin(angle)])
            state["recoil_velocity"] += kick * recoil_dir

    alive_mask = state["alive"]
    daughter_mask = ~alive_mask

    if np.any(alive_mask):
        state["display_positions"][alive_mask] += 0.11 * (
            state["parent_positions"][alive_mask] - state["display_positions"][alive_mask]
        )
    if np.any(daughter_mask):
        state["display_positions"][daughter_mask] += 0.16 * (
            state["daughter_positions"][daughter_mask] - state["display_positions"][daughter_mask]
        )

    state["recoil_offset"] += state["recoil_velocity"] * dt
    state["recoil_velocity"] *= np.exp(-4.0 * dt)
    state["recoil_offset"] *= np.exp(-2.8 * dt)
    state["glow_timer"] = max(0.0, state["glow_timer"] - dt)

    updated_events: List[Dict] = []
    for event in state["events"]:
        event["x"] += event["vx"] * dt
        event["y"] += event["vy"] * dt
        event["age"] += dt
        event["path_x"].append(event["x"])
        event["path_y"].append(event["y"])
        if len(event["path_x"]) > event["trail"]:
            event["path_x"] = event["path_x"][-event["trail"] :]
            event["path_y"] = event["path_y"][-event["trail"] :]
        if event["age"] <= event["life"]:
            updated_events.append(event)
    state["events"] = updated_events

    state["sim_time"] += dt
    parent_count = int(np.sum(state["alive"]))
    daughter_count = len(state["alive"]) - parent_count
    state["history_time"].append(state["sim_time"])
    state["history_parent"].append(parent_count)
    state["history_daughter"].append(daughter_count)


def advance_simulation(isotope_name: str, speed: float) -> None:
    state = decay_state()
    now = time.time()
    elapsed = max(0.0, now - state["last_wall_time"])
    state["last_wall_time"] = now

    if not state["running"]:
        return

    state["time_accumulator"] += min(elapsed * speed, 0.30)
    fixed_dt = 0.02
    max_steps = 12
    steps = 0
    while state["time_accumulator"] >= fixed_dt and steps < max_steps:
        apply_decay_step(fixed_dt, isotope_name)
        state["time_accumulator"] -= fixed_dt
        steps += 1


def theoretical_parent_curve(times: np.ndarray, n0: float, half_life: float) -> np.ndarray:
    lam = np.log(2.0) / half_life
    return n0 * np.exp(-lam * times)


def style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor("#121826")
    ax.set_title(title, color="white", fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color="white")
    ax.set_ylabel(ylabel, color="white")
    ax.tick_params(colors="#d7dde8")
    for spine in ax.spines.values():
        spine.set_color("#6b7280")
    ax.grid(True, alpha=0.18)


def draw_decay_scene(ax, isotope_name: str) -> None:
    state = decay_state()
    data = ISOTOPES[isotope_name]
    alive = state["alive"]
    if alive is None:
        return

    positions = state["display_positions"]
    offset = state["recoil_offset"]
    shifted = positions + offset

    ax.clear()
    ax.set_facecolor("#0b1020")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.05, 2.05)
    ax.set_aspect("equal")
    ax.axis("off")

    glow_alpha = min(0.35, 0.75 * state["glow_timer"])
    for r, a in [(1.30, 0.05 + glow_alpha * 0.6), (1.12, 0.10 + glow_alpha), (0.93, 0.08 + glow_alpha * 0.7)]:
        ax.add_patch(plt.Circle(tuple(offset), r, color="#ffd166", alpha=a, fill=False, linewidth=2.5))

    boundary_radius = 1.0 if np.any(alive) else 0.76
    ax.add_patch(
        plt.Circle(
            tuple(offset),
            boundary_radius,
            fill=False,
            linestyle="--",
            linewidth=1.5,
            color="#dbeafe",
            alpha=0.65,
        )
    )

    alive_mask = alive
    daughter_mask = ~alive

    if np.any(alive_mask):
        ax.scatter(
            shifted[alive_mask, 0],
            shifted[alive_mask, 1],
            s=74,
            c="#ff7a59",
            edgecolors="#111827",
            linewidths=0.5,
            alpha=0.95,
            zorder=5,
        )
    if np.any(daughter_mask):
        ax.scatter(
            shifted[daughter_mask, 0],
            shifted[daughter_mask, 1],
            s=60,
            c="#4fd1c5",
            edgecolors="#111827",
            linewidths=0.4,
            alpha=0.90,
            zorder=6,
        )

    for event in state["events"]:
        alpha = max(0.0, 1.0 - event["age"] / event["life"])
        for i in range(1, len(event["path_x"])):
            frac = i / max(1, len(event["path_x"]) - 1)
            ax.plot(
                event["path_x"][i - 1 : i + 1],
                event["path_y"][i - 1 : i + 1],
                linewidth=1.8,
                color=event["glow"],
                alpha=alpha * frac * 0.5,
                zorder=8,
            )
        ax.scatter(
            event["x"],
            event["y"],
            s=event["size"],
            c=event["glow"],
            alpha=alpha * 0.9,
            edgecolors="white",
            linewidths=0.6,
            zorder=9,
        )
        ax.text(
            event["x"],
            event["y"],
            event["label"],
            color="black" if event["glow"] != "#ffffff" else "white",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            zorder=10,
        )

    ax.text(
        0.0,
        -1.70,
        f"{data['symbol']} → {data['daughter_symbol']}   |   {data['decay_mode']}",
        color="#e5e7eb",
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.text(
        0.0,
        -1.90,
        "Dots represent an ensemble sample, not literal nucleons.",
        color="#cbd5e1",
        ha="center",
        va="center",
        fontsize=9.5,
    )


def draw_population_graph(ax, half_life: float) -> None:
    state = decay_state()
    t = np.array(state["history_time"], dtype=float)
    parent = np.array(state["history_parent"], dtype=float)
    daughter = np.array(state["history_daughter"], dtype=float)
    if len(t) == 0:
        return

    n0 = float(state["theory_n0"])
    theory_parent = theoretical_parent_curve(t, n0, half_life)

    ax.clear()
    style_axes(ax, "Decay population", "Time", "Count")
    ax.plot(t, parent, linewidth=2.2, label="Simulated parent")
    ax.plot(t, daughter, linewidth=2.0, label="Simulated daughter")
    ax.plot(t, theory_parent, "--", linewidth=2.0, label="Theoretical parent")
    ax.axvline(half_life, linestyle="--", linewidth=1.4, color="#ef4444", label="Half-life")
    ax.axhline(n0 / 2.0, linestyle=":", linewidth=1.5, color="#f59e0b", label="N₀ / 2")
    ax.legend(facecolor="#111827", edgecolor="#374151", labelcolor="white")


def render_header(isotope_name: str, half_life: float) -> None:
    data = ISOTOPES[isotope_name]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Parent isotope", data["symbol"])
    c2.metric("Daughter isotope", data["daughter_symbol"])
    c3.metric("Decay channel", data["decay_mode"])
    c4.metric("Simulation half-life", f"{half_life:.1f}")


def render_info_panels(isotope_name: str) -> None:
    data = ISOTOPES[isotope_name]
    c1, c2 = st.columns(2)
    with c1:
        card(
            "Physics model",
            "This module visualises radioactive decay as an ensemble of sample points. The count follows stochastic decay, while the comparison curve shows the ideal exponential law.",
        )
        card(
            "Nuclear data",
            f"Parent: Z={data['parent_protons']}, N={data['parent_neutrons']}<br>"
            f"Daughter: Z={data['daughter_protons']}, N={data['daughter_neutrons']}",
        )
    with c2:
        card(
            "Visual legend",
            "Orange = undecayed sample points<br>"
            "Cyan = decayed sample points<br>"
            "Magenta = alpha emission<br>"
            "Green = beta emission<br>"
            "White = gamma emission",
        )
        # card(
        #     "Why this layout works better",
        #     "Instead of forcing a cramped second graph, the lower section now shows live numerical diagnostics and half-life milestones. That adds new information without fighting the Streamlit layout.",
        # )


def render_metrics() -> None:
    state = decay_state()
    alive = state["alive"]
    if alive is None:
        return

    parent_count = int(np.sum(alive))
    daughter_count = len(alive) - parent_count

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Undecayed count", parent_count)
    m2.metric("Decayed count", daughter_count)
    m3.metric("Simulation time", f"{state['sim_time']:.2f}")
    m4.metric("Active emissions", len(state["events"]))


def sync_half_life_with_isotope(isotope_name: str) -> None:
    previous = st.session_state.get("decay_last_isotope")
    if previous != isotope_name:
        st.session_state[HALF_LIFE_KEY] = float(ISOTOPES[isotope_name]["half_life"])
        st.session_state["decay_last_isotope"] = isotope_name


def render_controls() -> Tuple[str, int, float, float]:
    st.sidebar.header("☢️ Radioactive Decay Controls")

    if ISOTOPE_KEY not in st.session_state:
        st.session_state[ISOTOPE_KEY] = list(ISOTOPES.keys())[0]
    isotope_name = st.sidebar.selectbox("Isotope", list(ISOTOPES.keys()), key=ISOTOPE_KEY)
    sync_half_life_with_isotope(isotope_name)

    if COUNT_KEY not in st.session_state:
        st.session_state[COUNT_KEY] = 100
    if SPEED_KEY not in st.session_state:
        st.session_state[SPEED_KEY] = 1.0

    display_count = st.sidebar.slider("Ensemble sample size", 30, 220, step=10, key=COUNT_KEY)
    half_life = st.sidebar.slider(
        "Simulation half-life",
        min_value=1.0,
        max_value=20.0,
        step=0.5,
        key=HALF_LIFE_KEY,
    )
    speed = st.sidebar.slider("Playback speed", 0.2, 4.0, 1.0, 0.1, key=SPEED_KEY)

    c1, c2, c3 = st.sidebar.columns(3)
    play = c1.button("▶ Play", width="stretch", key="decay_play")
    pause = c2.button("⏸ Pause", width="stretch", key="decay_pause")
    reset = c3.button("↺ Reset", width="stretch", key="decay_reset")

    state = decay_state()
    if play:
        state["running"] = True
        state["last_wall_time"] = time.time()
    if pause:
        state["running"] = False
    if reset or needs_reset(isotope_name, display_count, half_life):
        reset_decay_state(isotope_name, display_count, half_life)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "This lab uses a stylised ensemble model. The points are not literal nucleons inside one nucleus."
    )

    return isotope_name, display_count, float(half_life), float(speed)


def get_live_statistics(half_life: float) -> Dict[str, float]:
    state = decay_state()
    n0 = max(1, int(state["theory_n0"] or 1))
    parent_count = int(state["history_parent"][-1])
    daughter_count = int(state["history_daughter"][-1])
    sim_time = float(state["sim_time"])
    theory_parent = float(theoretical_parent_curve(np.array([sim_time]), n0, half_life)[0])
    survival_fraction = parent_count / n0
    theory_fraction = theory_parent / n0
    abs_error = abs(parent_count - theory_parent)
    return {
        "time": sim_time,
        "n0": n0,
        "parent": parent_count,
        "daughter": daughter_count,
        "survival": survival_fraction,
        "theory_survival": theory_fraction,
        "abs_error": abs_error,
    }


def render_half_life_panel(half_life: float) -> None:
    stats = get_live_statistics(half_life)
    n0 = stats["n0"]
    sim_time = stats["time"]

    card(
        "Half-life milestones",
        "Each half-life reduces the expected parent population by a factor of two.",
    )

    rows = []
    for k in range(4):
        time_mark = k * half_life
        expected = n0 / (2 ** k)
        reached = sim_time >= time_mark
        rows.append(
            {
                "Step": f"{k} × t₁⁄₂",
                "Time": f"{time_mark:.2f}",
                "Expected parent": f"{expected:.1f}",
                "Expected fraction": f"{expected / n0:.3f}",
                "Reached?": "Yes" if reached else "No",
            }
        )

    left, right = st.columns([1.05, 1.15])
    with left:
        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch", hide_index=True)
    with right:
        labels = ["Now", "1 × t₁⁄₂", "2 × t₁⁄₂", "3 × t₁⁄₂"]
        fractions = [
            stats["survival"],
            0.5,
            0.25,
            0.125,
        ]
        for label, frac in zip(labels, fractions):
            st.markdown(f"**{label}**")
            st.progress(float(max(0.0, min(1.0, frac))))
            st.markdown(
                f"<div class='decay-mini'>Fraction remaining: {frac:.3f}</div>",
                unsafe_allow_html=True,
            )


def render_statistics_panel(half_life: float) -> None:
    stats = get_live_statistics(half_life)

    card(
        "Live numerical diagnostics",
        "This panel shows what the simulation is doing right now and how far it is from the ideal exponential expectation.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Initial sample N₀", stats["n0"])
    c2.metric("Survival fraction", f"{stats['survival']:.3f}")
    c3.metric("Theory fraction", f"{stats['theory_survival']:.3f}")
    c4.metric("|Sim − Theory|", f"{stats['abs_error']:.2f}")

    details = pd.DataFrame(
        [
            ("Elapsed time", f"{stats['time']:.2f}"),
            ("Undecayed count", f"{stats['parent']}"),
            ("Decayed count", f"{stats['daughter']}"),
            ("Simulation survival", f"{stats['survival']:.4f}"),
            ("Theoretical survival", f"{stats['theory_survival']:.4f}"),
            ("Absolute count error", f"{stats['abs_error']:.4f}"),
            ("Reference half-life", f"{half_life:.2f}"),
        ],
        columns=["Quantity", "Value"],
    )
    st.dataframe(details, width="stretch", hide_index=True)


@st.fragment(run_every="80ms")
def render_live_view(isotope_name: str, half_life: float, speed: float) -> None:
    advance_simulation(isotope_name, speed)

    top_left, top_right = st.columns([1.28, 1.0])

    with top_left:
        card("Live decay scene", "Recoil, glow pulse, and emitted radiation are shown in a stylised microscopic view.")
        fig_scene, ax_scene = plt.subplots(figsize=(8.4, 6.6))
        draw_decay_scene(ax_scene, isotope_name)
        fig_scene.tight_layout(pad=1.1)
        st.pyplot(fig_scene, width="stretch")
        plt.close(fig_scene)

    with top_right:
        card("Population comparison", "This graph compares simulated parent and daughter counts with the theoretical decay law.")
        fig_pop, ax_pop = plt.subplots(figsize=(7.2, 4.1))
        draw_population_graph(ax_pop, half_life)
        fig_pop.tight_layout(pad=1.1)
        st.pyplot(fig_pop, width="stretch")
        plt.close(fig_pop)

    lower_left, lower_right = st.columns([1.0, 1.0])
    with lower_left:
        render_statistics_panel(half_life)
    with lower_right:
        render_half_life_panel(half_life)

    render_metrics()


def run() -> None:
    inject_module_style()
    st.title("☢️ Radioactive Decay")
    st.caption(
        "An interactive physics lab for Nuclear Decay"
    )

    isotope_name, _display_count, half_life, speed = render_controls()
    render_header(isotope_name, half_life)
    render_info_panels(isotope_name)
    render_live_view(isotope_name, half_life, speed)
