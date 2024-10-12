import os
from langchain_core.globals import set_verbose, set_debug
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def initialize_env(api_key=None, api_base=None, use_openai=False, llm_model="qwen2.5:3b"):
    set_debug(True)
    set_verbose(True)
    if use_openai:
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_BASE"] = api_base
        model = OpenAI(temperature=0.9)
    else:
        model = ChatOllama(model=llm_model)
    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-zh-v1.5", model_kwargs={'device': 'cpu'})
    return model, embeddings
