import streamlit as st
from modules import projectile_motion
from utils.layout import set_page_layout

set_page_layout()

st.title("Interactive Physics Lab")
st.write("A collection of interactive physics simulations built with Python and Streamlit.")

topic = st.sidebar.selectbox(
    "Choose a topic",
    ["Projectile Motion"]
)

if topic == "Projectile Motion":
    projectile_motion.run()

# import streamlit as st

# st.set_page_config(page_title="Test App", layout="wide")
# st.title("Test App")
# st.write("If you can see this text, Streamlit is working correctly.")