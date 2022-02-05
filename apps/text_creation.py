from support.helper import SupportMethods
import streamlit as st

sm = SupportMethods()

def app():
    st.title('Generate Text')
    st.markdown(unsafe_allow_html=True, body="""<pre>Will generate text by providing sample input text
    and count of next words 
    to generate and using Greedy/Beam</pre>""")
    sm.generate_text()