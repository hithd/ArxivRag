#!/bin/env python3
import streamlit as st

st.set_page_config(page_title="ChatPDF and ArXiv", layout="wide")

from frontend.st_pages import initialize_session_state, pdf_chat_page, arxiv_chat_page, api_settings_page

def main():
    initialize_session_state()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["PDF Chat", "ArXiv Chat", "API Settings"])
    
    if page == "PDF Chat":
        pdf_chat_page()
    elif page == "ArXiv Chat":
        arxiv_chat_page()
    else:
        api_settings_page()

if __name__ == "__main__":
    main()
