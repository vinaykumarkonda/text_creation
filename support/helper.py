import streamlit as st
from datetime import datetime
from support.text_generator import TextGenerator

class SupportMethods():

    def sample_text_input(self):
        sample_text = st.sidebar.text_input('Provide Sample Input Text')
        return sample_text

    def using_beam_radio_button(self):
        using_beam_flag = st.sidebar.radio('with_Beam', (False, True))
        return using_beam_flag

    def get_count_of_next_set_of_words(self):
        count_of_next_words = st.sidebar.number_input('Generate Next Set of Words', min_value=20)
        return int(count_of_next_words)

    def generate_text(self):
        sample_text = self.sample_text_input()
        using_beam_flag = self.using_beam_radio_button()
        count_of_next_words = self.get_count_of_next_set_of_words()
        if (sample_text !=""):
            if st.sidebar.button('Submit'):
                st.empty()
                start = datetime.now()
                generated_text = TextGenerator(sample_input_text=sample_text, predict_next_words=count_of_next_words, with_beam=using_beam_flag).generate_text()
                end = datetime.now()
                st.write("Request got completed in ", int((end-start).total_seconds() * 1000),  " milliseconds")
                st.subheader(generated_text)

    def set_page_title(self, title):
        st.sidebar.markdown(unsafe_allow_html=True, body=f"""
            <iframe height=0 srcdoc="<script>
                const title = window.parent.document.querySelector('title') \

                const oldObserver = window.parent.titleObserver
                if (oldObserver) {{
                    oldObserver.disconnect()
                }} \

                const newObserver = new MutationObserver(function(mutations) {{
                    const target = mutations[0].target
                    if (target.text !== '{title}') {{
                        target.text = '{title}'
                    }}
                }}) \

                newObserver.observe(title, {{ childList: true }})
                window.parent.titleObserver = newObserver \

                title.text = '{title}'
            </script>" />
        """)
