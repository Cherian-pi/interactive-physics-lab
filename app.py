import streamlit as st
from modules import projectile_motion, electric_field

st.set_page_config(page_title="Interactive Physics Lab", layout="wide")

st.sidebar.title("Interactive Physics Lab")
project = st.sidebar.radio(
    "Choose a project",
    ["Projectile Motion", "Electric Field"]
)

if project == "Projectile Motion":
    projectile_motion.run()
elif project == "Electric Field":
    electric_field.render()