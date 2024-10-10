#!/bin/env python3
import os
import time
import tempfile
import streamlit as st
from streamlit_chat import message
from rag import ChatPDF


st.set_page_config(page_title="ChatPDF and ArXiv", layout="wide")

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "assistant" not in st.session_state:
        st.session_state["assistant"] = None
    if "pdf_files" not in st.session_state:
        st.session_state["pdf_files"] = {}
    if "arxiv_results" not in st.session_state:
        st.session_state["arxiv_results"] = []
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "PDF Chat"
    if "api_configured" not in st.session_state:
        st.session_state["api_configured"] = False

def display_messages():
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))

def process_input(input_type):
    if not st.session_state["api_configured"]:
        st.error("Please configure the API settings first.")
        return

    if st.session_state[f"{input_type}_input"] and len(st.session_state[f"{input_type}_input"].strip()) > 0:
        user_text = st.session_state[f"{input_type}_input"].strip()
        with st.spinner("Thinking..."):
            if input_type == "pdf":
                selected_pdf = st.session_state.get("selected_pdf")
                if selected_pdf:
                    agent_text = st.session_state["assistant"].ask_pdf(user_text, selected_pdf)
                else:
                    agent_text = "Please select a PDF document first."
            else:  # arxiv
                max_docs = st.session_state.get("max_docs", 2)
                arxiv_metadata = st.session_state["assistant"].get_arxiv_metadata(user_text, max_docs)
                st.session_state["arxiv_results"] = arxiv_metadata
                agent_text = st.session_state["assistant"].ask_arxiv(user_text, max_docs)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

def read_and_save_files():
    if not st.session_state["api_configured"]:
        st.error("Please configure the API settings first.")
        return

    if st.session_state["assistant"] is None:
        st.error("Assistant is not initialized. Please configure the API settings first.")
        return

    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["pdf_input"] = ""
    st.session_state["pdf_files"] = {}

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name
            st.session_state["pdf_files"][file.name] = file_path

        with st.spinner(f"Ingesting {file.name}"):
            try:
                t0 = time.time()
                st.session_state["assistant"].ingest(file_path, file.name)
                t1 = time.time()
                st.session_state["messages"].append(
                    (f"Ingested {file.name} in {t1 - t0:.2f} seconds", False)
                )
            except Exception as e:
                st.error(f"Error ingesting {file.name}: {str(e)}")

def pdf_chat_page():
    st.header("PDF Chat")
    
    st.subheader("Upload documents")
    st.file_uploader(
        "Upload documents",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_files,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    if st.session_state["pdf_files"]:
        st.session_state["selected_pdf"] = st.selectbox("Select PDF to interact with:", list(st.session_state["pdf_files"].keys()))

    st.subheader("Chat")
    display_messages()
    st.text_input("Message", key="pdf_input", on_change=process_input, args=("pdf",))

def arxiv_chat_page():
    st.header("ArXiv Chat")
    
    st.subheader("ArXiv Search Settings")
    st.session_state["max_docs"] = st.slider("Number of ArXiv documents to retrieve", 1, 10, 2)

    st.subheader("Chat")
    display_messages()
    st.text_input("Message", key="arxiv_input", on_change=process_input, args=("arxiv",))

    if st.session_state.get("arxiv_results"):
        st.subheader("ArXiv Search Results")
        for result in st.session_state["arxiv_results"]:
            st.write(f"Title: {result['title']}")
            st.write(f"Authors: {result['authors']}")
            st.write(f"Summary: {result['summary']}")
            st.write("---")

def api_settings_page():
    st.header("API Settings")

    api_choice = st.radio("Select API to use:", ("OpenAI", "Local Ollama"))
    api_key = st.text_input("OpenAI API Key", type="password")
    api_base = st.text_input("OpenAI API Base URL", value="https://api.openai.com/v1")
    model = "qwen2.5"

    if st.button("Set API Configuration"):
        use_openai = (api_choice == "OpenAI")
        try:
            st.session_state["assistant"] = ChatPDF(llm_model=model, api_key=api_key, api_base=api_base, use_openai=use_openai)
            if use_openai:
                os.environ["OPENAI_API_KEY"] = api_key
                os.environ["OPENAI_API_BASE"] = api_base
            st.session_state["api_configured"] = True
            st.success("API Configuration set successfully!")
        except Exception as e:
            st.error(f"Error setting API configuration: {str(e)}")
            st.session_state["api_configured"] = False

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