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
from langchain_community.llms import OpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import ArxivRetriever
from langchain.embeddings import OpenAIEmbeddings
set_debug(True)
set_verbose(True)



class ChatPDF:
    vector_stores = {}
    retriever = None
    chain = None

    def __init__(self, llm_model: str = "qwen2.5:3b", api_key: str = None, api_base: str = None, use_openai: bool = False):
        self.storage_directory = 'chroma_db'
        os.environ["OPENAI_API_KEY"] = 'sk-gKrFRQIZQWfNyAy949C7B08180C2453086Cc61C8355fB64f'
        os.environ["OPENAI_API_BASE"] = 'https://api.bianxie.ai/v1'
        if use_openai:
            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["OPENAI_API_BASE"] = api_base
            self.model = OpenAI(temperature=0.9)
        else:
            self.model = ChatOllama(model=llm_model)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100
        )
        self.prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the context provided.

            Context: {context}

            Question: {question}"""
        )
        #choose different embedding for different capability

        self.embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-zh-v1.5", model_kwargs={'device': 'cpu'})
        # self.embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
        # self.embeddings = OllamaEmbeddings(model="shaw/dmeta-embedding-zh") 
        self.retriever = None
        self.arxiv_retriever = ArxivRetriever(load_max_docs=2, get_full_documents=True)

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


    def ask_pdf(self, query: str, pdf_name: str):
        if pdf_name not in self.vector_stores:
            return "Please add the specified PDF document first."

        self.retriever = self.vector_stores[pdf_name].as_retriever(
            search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}
        )
        context = self.retriever.invoke(query)

        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

        return chain.invoke({"context": context, "question": query})
    def ask_arxiv(self, query: str, max_docs: int = 2):
        self.arxiv_retriever.load_max_docs = max_docs
        docs = self.arxiv_retriever.get_relevant_documents(query)
        context = "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

        return chain.invoke({"context": context, "question": query})

    def clear(self):
        self.retriever = None
        self.chain = None

    def get_arxiv_metadata(self, query: str, max_docs: int = 2):
        self.arxiv_retriever.load_max_docs = max_docs
        docs = self.arxiv_retriever.get_relevant_documents(query)
        return [{"title": doc.metadata.get("Title"), "authors": doc.metadata.get("Authors"), "summary": doc.metadata.get("Summary")} for doc in docs]