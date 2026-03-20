import streamlit as st

from modules.mechanics import projectile_motion, oscillations, orbital_motion, fluid_flow
from modules.electrodynamics import electric_field, lorentz_force
from modules.nuclear import radioactive_decay, binding_energy
from modules.atomic_physics import atomic_physics
from modules.modern_physics import double_slit


st.set_page_config(
    page_title="Interactive Physics Lab",
    page_icon="🧪",
    layout="wide"
)


def show_home():
    st.title("Interactive Physics Lab")
    st.markdown(
        """
        Welcome to the **Interactive Physics Lab** — an interactive simulation platform
        built to help students understand physics visually, intuitively, and in real time.

        Instead of only reading equations, users can explore how physical systems behave
        by adjusting parameters, observing animations, and connecting theory with motion,
        fields, and measurable quantities.
        """
    )

    st.markdown("## Explore Physics by Category")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            ### Mechanics
            - Projectile Motion  
            - Oscillations  
            - Orbital Motion  
            - Fluid Flow
            """
        )

    with col2:
        st.markdown(
            """
            ### Electrodynamics
            - Electric Field  
            - Lorentz Force
            """
        )

    with col3:
        st.markdown(
            """
            ### Modern and Atomic Physics
            - Atomic Transitions  
            - Double Slit  
            - Nuclear Simulations
            """
        )

    st.markdown("---")

    st.markdown("## What this project offers")
    st.markdown(
        """
        - Real-time interactive simulations  
        - Physics-based visualisation of equations  
        - Adjustable parameters and live plots  
        - A clean learning environment for school and undergraduate students
        """
    )

    st.markdown("## How to use")
    st.markdown(
        """
        Use the sidebar to move from the **Home** page into the simulation sections.
        Select a physics category, then choose a simulation to interact with.
        """
    )

    st.markdown("---")
    st.info(
        "This lab is designed as a visual learning space where equations, graphs, and motion work together."
    )

    st.markdown("---")

    st.markdown(
        """
        ### About the developer

        **Cherian P Ittyipe**  
        Physics & Astrophysics Graduate  

        This project reflects my broader interest in computational physics, scientific
        visualisation, and building interactive tools for understanding physical systems.
        While my primary academic focus is cosmology, this lab reflects a
        broader approach to connecting theory, simulation, and physical understanding through
        interactive tools.


        [GitHub](https://github.com/Cherian-pi) •  
        [LinkedIn](https://www.linkedin.com/in/cherianpittyipe/) •  
        [Email](mailto:cherianpittyipe@outlook.com)
        """
    )

st.sidebar.title("Interactive Physics Lab")

main_page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Simulations"
    ]
)

if main_page == "Home":
    show_home()

elif main_page == "Simulations":
    category = st.sidebar.selectbox(
        "Choose a physics category",
        [
            "Mechanics",
            "Electrodynamics",
            "Nuclear Physics",
            "Atomic Physics",
            "Modern Physics"
        ]
    )

    if category == "Mechanics":
        simulation = st.sidebar.radio(
            "Choose a simulation",
            [
                "Projectile Motion",
                "Oscillations",
                "Orbital Motion",
                "Fluid Flow"
            ]
        )

        if simulation == "Projectile Motion":
            projectile_motion.run()
        elif simulation == "Oscillations":
            oscillations.run()
        elif simulation == "Orbital Motion":
            orbital_motion.run()
        elif simulation == "Fluid Flow":
            fluid_flow.run()

    elif category == "Electrodynamics":
        simulation = st.sidebar.radio(
            "Choose a simulation",
            [
                "Electric Field",
                "Lorentz Force",
        
            ]
        )

        if simulation == "Electric Field":
            electric_field.render()
        elif simulation == "Lorentz Force":
            lorentz_force.run()
        # elif simulation == "Circuits":
        #     circuits.run()

    elif category == "Nuclear Physics":
        simulation = st.sidebar.radio(
            "Choose a simulation",
            [
                "Radioactive Decay",
                "Binding Energy"
            ]
        )

        if simulation == "Radioactive Decay":
            radioactive_decay.run()
        elif simulation == "Binding Energy":
            binding_energy.run()

    elif category == "Atomic Physics":
        simulation = st.sidebar.radio(
            "Choose a simulation",
            [
                "Atomic Transitions"
            ]
        )

        if simulation == "Atomic Transitions":
            atomic_physics.run()

    elif category == "Modern Physics":
        simulation = st.sidebar.radio(
            "Choose a simulation",
            [
                "Double Slit"
            ]
        )

        if simulation == "Double Slit":
            double_slit.run()