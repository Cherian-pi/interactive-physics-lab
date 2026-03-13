import numpy as np
import plotly.graph_objects as go
import streamlit as st


def compute_trajectory(speed, angle_deg, gravity):
    angle_rad = np.radians(angle_deg)

    flight_time = (2 * speed * np.sin(angle_rad)) / gravity
    t = np.linspace(0, flight_time, 300)

    x = speed * np.cos(angle_rad) * t
    y = speed * np.sin(angle_rad) * t - 0.5 * gravity * t**2

    max_height = (speed**2 * np.sin(angle_rad)**2) / (2 * gravity)
    horizontal_range = (speed**2 * np.sin(2 * angle_rad)) / gravity

    return {
        "x": x,
        "y": y,
        "flight_time": flight_time,
        "max_height": max_height,
        "horizontal_range": horizontal_range,
    }


def run():
    st.header("Projectile Motion Explorer")
    st.write(
        "Compare three projectile trajectories by changing the speed, angle, and gravity for each one."
    )

    st.subheader("Controls")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Projectile 1")
        speed1 = st.slider("Speed 1 (m/s)", 1, 100, 20)
        angle1 = st.slider("Angle 1 (degrees)", 1, 89, 45)
        gravity1 = st.slider("Gravity 1 (m/s²)", 1.0, 20.0, 9.81)

    with col2:
        st.markdown("### Projectile 2")
        speed2 = st.slider("Speed 2 (m/s)", 1, 100, 30)
        angle2 = st.slider("Angle 2 (degrees)", 1, 89, 35)
        gravity2 = st.slider("Gravity 2 (m/s²)", 1.0, 20.0, 9.81)

    with col3:
        st.markdown("### Projectile 3")
        speed3 = st.slider("Speed 3 (m/s)", 1, 100, 40)
        angle3 = st.slider("Angle 3 (degrees)", 1, 89, 60)
        gravity3 = st.slider("Gravity 3 (m/s²)", 1.0, 20.0, 9.81)

    traj1 = compute_trajectory(speed1, angle1, gravity1)
    traj2 = compute_trajectory(speed2, angle2, gravity2)
    traj3 = compute_trajectory(speed3, angle3, gravity3)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=traj1["x"],
            y=traj1["y"],
            mode="lines",
            name="Projectile 1"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=traj2["x"],
            y=traj2["y"],
            mode="lines",
            name="Projectile 2"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=traj3["x"],
            y=traj3["y"],
            mode="lines",
            name="Projectile 3"
        )
    )

    fig.update_layout(
        title="Comparative Projectile Trajectories",
        xaxis_title="Horizontal Distance (m)",
        yaxis_title="Vertical Height (m)",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key Results")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.markdown("### Projectile 1")
        st.write(f"**Time of flight:** {traj1['flight_time']:.2f} s")
        st.write(f"**Maximum height:** {traj1['max_height']:.2f} m")
        st.write(f"**Horizontal range:** {traj1['horizontal_range']:.2f} m")

    with result_col2:
        st.markdown("### Projectile 2")
        st.write(f"**Time of flight:** {traj2['flight_time']:.2f} s")
        st.write(f"**Maximum height:** {traj2['max_height']:.2f} m")
        st.write(f"**Horizontal range:** {traj2['horizontal_range']:.2f} m")

    with result_col3:
        st.markdown("### Projectile 3")
        st.write(f"**Time of flight:** {traj3['flight_time']:.2f} s")
        st.write(f"**Maximum height:** {traj3['max_height']:.2f} m")
        st.write(f"**Horizontal range:** {traj3['horizontal_range']:.2f} m")

    st.subheader("What to observe")
    st.markdown(
        """
        - Increasing **speed** generally increases both range and height.
        - The **launch angle** changes the trade-off between height and horizontal distance.
        - Lower **gravity** keeps the projectile in the air longer.
        - Comparing multiple trajectories helps reveal which parameter is driving the difference.
        """
    )