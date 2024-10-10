from langchain_core.globals import set_verbose, set_debug
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
import openai
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.retrievers import BM25Retriever
set_debug(True)
set_verbose(True)


class ChatPDF:
    vector_stores = {}
    retriever = None
    chain = None

    def __init__(self, llm_model: str = "qwen2.5:3b", api_key: str = None, api_base: str = None, use_openai: bool = False):
        self.storage_directory = 'chroma_db'
        if use_openai:
            openai.api_key = api_key
            openai.api_base = api_base
            self.model = openai.ChatCompletion.create
        else:
            self.model = ChatOllama(model=llm_model)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=100
        )
        self.prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    "You are a helpful assistant that can answer questions about the PDF document that uploaded by the user. ",
                ),
                (
                    "human",
                    "Here is the document pieces: {context}\nQuestion: {question}",
                ),
            ]
        )
        
        self.embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-zh-v1.5", model_kwargs={'device': 'cpu'})
        #self.embeddings = OllamaEmbeddings(model="shaw/dmeta-embedding-zh")
        # self.embeddings = JinaEmbeddings( jina_api_key="jina_b7d2d2b8513a422a99cc34e0751371c1wzzk_vZELQhzZ780rvsM8cR9ObY9", model_name="jina-embeddings-v3")
        self.retriever = None
    def ingest(self, pdf_file_path: str, pdf_name: str):
        # Check if the PDF name is already in vector stores
        if pdf_name in self.vector_stores:
            print(f"{pdf_name} is already ingested in the vector store.")
            return
        pdf_directory = os.path.join(self.storage_directory, pdf_name)
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)

        self.vector_stores[pdf_name] = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory= pdf_directory,
        )

        
    def ask(self, query: str, pdf_name: str):
        if pdf_name not in self.vector_stores:
            return "Please, add the specified PDF document first."

        self.retriever = self.vector_stores[pdf_name].as_retriever(
            search_type="mmr",search_kwargs={"k": 1, "fetch_k": 5}
        )
        

        self.retriever.invoke(query)

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

        return self.chain.invoke(query)

    def clear(self):
        
        self.retriever = None
        self.chain = None
