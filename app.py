import streamlit as st
from support.helper import SupportMethods
from multiapp import MultiApp
# import your app modules here
from apps import text_creation
from PIL import Image

sm  = SupportMethods()

favicon = Image.open('favicon.jpg')

st.set_page_config(
    page_title="TextCreation",
    page_icon=favicon,
    layout="wide",
)

app = MultiApp()

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
            
#MainMenu {visibility: hidden;}

sm.set_page_title("TextCreation")
st.sidebar.image(favicon, width=100)
st.sidebar.markdown("*Version 1.1*")

# Add all your application here
app.add_app("Generate Text", text_creation.app)


# The main app
app.run()
