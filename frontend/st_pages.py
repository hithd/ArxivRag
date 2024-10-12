import tempfile
import time
import traceback
import streamlit as st
from streamlit_chat import message
from api.rag import ChatPDF

# CSS 样式
st.markdown(
    """
    <style>
    .expander .streamlit-expanderHeader {
        font-size: 1.5rem;  /* 增大字体大小 */
        font-weight: bold;  /* 加粗 */
    }
    .expander .streamlit-expanderContent {
        padding: 20px;  /* 增加内容区域的填充 */
    }
    .expander {
        border: 2px solid black;  /* 添加边框 */
        padding: 10px;  /* 增加外部填充 */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    if "stored_arxiv_docs" not in st.session_state:
        st.session_state["stored_arxiv_docs"] = set()

def display_messages():
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
        st.markdown("<br>", unsafe_allow_html=True)

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
                sources = []  # No sources for PDF queries
            elif input_type == "arxiv":
                max_docs = st.session_state.get("max_docs", 3)
                agent_text, sources = st.session_state["assistant"].ask_arxiv(user_text, max_docs)
            else:  # local_arxiv
                agent_text = st.session_state["assistant"].ask_local_arxiv(user_text)
                sources = []  # No sources for local_arxiv queries

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))
        
        formatted_response = agent_text
        if input_type == "arxiv" and sources:
            formatted_response += "\n\n---\n\n"  # Add a separator
            formatted_response += "**Sources:**\n\n"
            for source in sources:
                formatted_response += f"* **Title:** {source['title']}\n"
                formatted_response += f"  **Authors:** {source['authors']}\n"
                formatted_response += f"  **Published:** {source['published']}\n"
                formatted_response += f"  **URL:** {source['pdf_url']}\n\n"

        st.session_state["messages"].append((formatted_response, False))

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
    st.markdown("# 📑 PDF Chat")
    st.header("")
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
    if "added_articles" not in st.session_state:
        st.session_state.added_articles = set()
    st.markdown("# 📚 ArXiv Chat")
    
    if not st.session_state["api_configured"]:
        st.error("Please configure the API settings first.")
        return

    if st.session_state["assistant"] is None:
        st.error("Assistant is not initialized. Please configure the API settings first.")
        return
    
    st.subheader("ArXiv Search")
    search_query = st.text_input("Enter search query:")
    time_period = st.selectbox("Select time period:", ["week", "month"])
    max_results = st.slider("Number of results to retrieve:", 1, 50, 10)
    
    if st.button("Search ArXiv"):
        if search_query:
            with st.spinner("Searching ArXiv..."):
                try:
                    results = st.session_state["assistant"].search_recent_arxiv(search_query, time_period, max_results)
                    st.session_state["arxiv_results"] = results
                except Exception as e:
                    st.error(f"An error occurred while searching ArXiv: {str(e)}")
                    st.write(f"Exception type: {type(e)}")
                    st.write(f"Exception args: {e.args}")
                    st.write("Traceback:")
                    st.code(traceback.format_exc())
        else:
            st.warning("Please enter a search query.")

    if st.session_state.get("arxiv_results"):
        st.header("ArXiv Search Results", divider='rainbow')
        
        for i, result in enumerate(st.session_state["arxiv_results"]):
            
            with st.expander(f"{i+1}. {result['title']}", expanded=False):
                    # Enlarged title inside the expander
                st.markdown(f"<h2 style='color:#4B9CD3;'>{result['title']}</h2>", unsafe_allow_html=True)
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Authors:** {result['authors']}")
                    st.markdown("**Summary:**")
                    st.markdown(f"<p style='text-align: justify;'>{result['summary']}</p>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"**Published:** {result['published'].strftime('%Y-%m-%d')}")
                    st.markdown(f"[View PDF]({result['pdf_url']})")
                
                if st.button(f"Add to Knowledge Base", key=f"add_{i}"):
                    doc_metadata = {
                        "id": result['id'],
                        "title": result['title'],
                        "authors": result['authors'],
                        "pdf_url": result['pdf_url']
                    }
                    with st.spinner("Loading and processing full text..."):
                        try:
                            st.session_state["assistant"].ingest_arxiv(result['id'], doc_metadata)
                            st.session_state["stored_arxiv_docs"].add(result['id'])
                            st.success(f"Added '{result['title']}' to the knowledge base.")
                        except Exception as e:
                            st.error(f"Error adding document to knowledge base: {str(e)}")

    # 强调显示添加文档区域
    # Remove the className argument from st.expander
    st.markdown("# 📄 Added Documents")
    with st.expander("Tap to Find", expanded=True):
        if st.session_state["stored_arxiv_docs"]:
            st.markdown("### **Documents in Knowledge Base:**")
            for doc in st.session_state["stored_arxiv_docs"]:
                st.markdown(f"- {doc}")
        else:
            st.write("No documents added yet.")



    st.markdown("# 💬 Chat")
    chat_mode = st.radio("Select chat mode:", ["Direct ArXiv", "Stored Knowledge Base"])
    
    if chat_mode == "Direct ArXiv":
        st.session_state["max_docs"] = st.slider("Number of ArXiv documents to retrieve", 1, 10, 2)
        st.text_input("Message", key="arxiv_input", on_change=process_input, args=("arxiv",))
    else:
        if not st.session_state["stored_arxiv_docs"]:
            st.warning("No documents stored in the knowledge base. Please add some documents first.")
        else:
            st.text_input("Message", key="local_arxiv_input", on_change=process_input, args=("local_arxiv",))

    display_messages()

def api_settings_page():
    st.header("API Settings")

    api_choice = st.radio("Select API to use:", ("OpenAI", "Local Ollama"))
    api_key = st.text_input("OpenAI API Key", type="password")
    api_base = st.text_input("OpenAI API Base URL", value="https://api.bianxie.ai/v1")
    model = "qwen2.5"

    if st.button("Set API Configuration"):
        use_openai = (api_choice == "OpenAI")
        try:
            st.session_state["assistant"] = ChatPDF(llm_model=model, api_key=api_key, api_base=api_base, use_openai=use_openai)
            st.session_state["api_configured"] = True
            st.success("API Configuration set successfully!")
        except Exception as e:
            st.error(f"Error setting API configuration: {str(e)}")
            st.session_state["api_configured"] = False
