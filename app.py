#!/bin/env python3
import os
import time
import tempfile
import streamlit as st
from streamlit_chat import message
from rag import ChatPDF

st.set_page_config(page_title="ChatPDF")


def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if (
        st.session_state["user_input"]
        and len(st.session_state["user_input"].strip()) > 0
    ):
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner("Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text, st.session_state["selected_pdf"])

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def read_and_save_files():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""
    st.session_state["pdf_files"] = {}

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name
            st.session_state["pdf_files"][file.name] = file_path

        with st.session_state["ingestion_spinner"], st.spinner(
            f"Ingesting {file.name}"
        ):
            t0 = time.time()
            st.session_state["assistant"].ingest(file_path, file.name)
            t1 = time.time()

        st.session_state["messages"].append(
            (
                f"Ingested {file.name} in {t1 - t0:.2f} seconds",
                False,
            )
        )


def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatPDF()
        st.session_state["pdf_files"] = {}

    st.header("ChatPDF")

    st.subheader("API Configuration")
    api_choice = st.radio("Select API to use:", ("OpenAI", "Local Ollama"))
    api_key = ""
    api_base = ""
    model = "qwen2.5"

    if api_choice == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password")
        api_base = st.text_input("OpenAI API Base URL")
        model = st.selectbox("Select Model", ["gpt-3.5-turbo", "gpt-4"])

    if st.button("Set API Configuration"):
        use_openai = (api_choice == "OpenAI")
        st.session_state["assistant"] = ChatPDF(llm_model=model, api_key=api_key, api_base=api_base, use_openai=use_openai)

    st.subheader("Upload documents")
    st.file_uploader(
        "Upload documents",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_files,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.session_state["ingestion_spinner"] = st.empty()

    if st.session_state["pdf_files"]:
        st.session_state["selected_pdf"] = st.selectbox("Select PDF to interact with:", list(st.session_state["pdf_files"].keys()))

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)


if __name__ == "__main__":
    page()

