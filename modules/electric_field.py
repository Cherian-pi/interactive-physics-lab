import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-9


def compute_field_and_potential(X, Y, charges):
    Ex = np.zeros_like(X, dtype=float)
    Ey = np.zeros_like(Y, dtype=float)
    V = np.zeros_like(X, dtype=float)

    for q, cx, cy in charges:
        dx = X - cx
        dy = Y - cy
        r2 = dx**2 + dy**2 + EPS
        r = np.sqrt(r2)
        r3 = r2 * r

        Ex += q * dx / r3
        Ey += q * dy / r3
        V += q / r

    Ex = np.nan_to_num(Ex)
    Ey = np.nan_to_num(Ey)
    V = np.nan_to_num(V)

    return Ex, Ey, V


def render():
    st.title("Electric Field & Equipotential Visualizer")
    st.write(
        "Explore how electric field lines and equipotential contours change when you move point charges."
    )

    st.sidebar.header("Grid Settings")
    grid_points = st.sidebar.slider("Grid resolution", 50, 250, 140, key="ef_grid")
    plot_range = st.sidebar.slider("Axis range", 2, 10, 5, key="ef_range")

    st.sidebar.header("Charge Setup")
    num_charges = st.sidebar.slider("Number of charges", 1, 4, 2, key="ef_num_charges")

    subscript_digits = ["₁", "₂", "₃", "₄"]

    charges = []
    for i in range(num_charges):
        sub = subscript_digits[i]
        st.sidebar.subheader(f"Charge {i+1}")

        default_q = 1.0 if i % 2 == 0 else -1.0
        default_x = -1.0 + i * 2.0

        q = st.sidebar.slider(
            f"q{sub} (charge)",
            -5.0, 5.0, default_q, 0.1,
            key=f"ef_q_{i}"
        )
        x0 = st.sidebar.slider(
            f"x{sub} position",
            -5.0, 5.0, default_x, 0.1,
            key=f"ef_x_{i}"
        )
        y0 = st.sidebar.slider(
            f"y{sub} position",
            -5.0, 5.0, 0.0, 0.1,
            key=f"ef_y_{i}"
        )
        charges.append((q, x0, y0))

    show_vectors = st.checkbox(
        "Show electric field vectors", value=True, key="ef_vectors"
    )
    show_stream = st.checkbox(
        "Show field lines", value=True, key="ef_stream"
    )
    show_potential = st.checkbox(
        "Show equipotential contours", value=True, key="ef_potential"
    )

    x = np.linspace(-plot_range, plot_range, grid_points)
    y = np.linspace(-plot_range, plot_range, grid_points)
    X, Y = np.meshgrid(x, y)

    Ex, Ey, V = compute_field_and_potential(X, Y, charges)
    Emag = np.sqrt(Ex**2 + Ey**2)

    fig, ax = plt.subplots(figsize=(8, 8))

    if show_potential:
        contour = ax.contour(X, Y, V, levels=20)
        ax.clabel(contour, inline=True, fontsize=8)

    if show_stream:
        max_emag = np.nanmax(Emag)
        lw = 1 if max_emag == 0 else 1 + 2 * Emag / max_emag
        ax.streamplot(X, Y, Ex, Ey, density=1.2, linewidth=lw)

    if show_vectors:
        skip = max(grid_points // 20, 1)
        ax.quiver(
            X[::skip, ::skip],
            Y[::skip, ::skip],
            Ex[::skip, ::skip],
            Ey[::skip, ::skip]
        )

    for i, (q, cx, cy) in enumerate(charges):
        sub = subscript_digits[i]
        marker = "ro" if q > 0 else "bo"
        ax.plot(cx, cy, marker, markersize=12)
        ax.text(cx + 0.12, cy + 0.12, f"q{sub} = {q:+.1f}", fontsize=10)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Electric Field and Equipotential Map")
    ax.set_xlim(-plot_range, plot_range)
    ax.set_ylim(-plot_range, plot_range)
    ax.set_aspect("equal")
    ax.grid(True)

    st.pyplot(fig)

    st.markdown(r"""
    ## Key observations
    - Field lines emerge from positive charges and terminate on negative charges.
    - Equipotential lines connect points that share the same electric potential.
    - Electric field is a vector quantity, whereas electric potential is a scalar quantity.
    - When multiple charges are present, their contributions combine through superposition.

    ## Equations used
    For a collection of point charges:

    - Electric potential: $V = \sum_i \frac{q_i}{r_i}$
    - Electric field: obtained from the vector sum of the contribution from each charge

    This version uses arbitrary units so the focus remains on physical behaviour and geometric intuition.
    """)