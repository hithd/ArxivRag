from datetime import datetime, timedelta
import arxiv
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
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
set_debug(True)
set_verbose(True)



class ChatPDF:
    vector_stores = {}
    retriever = None
    chain = None
    arxiv_knowledge_base = []
    arxiv_vector_store = None
    bm25_retriever = None
    ensemble_retriever = None
    

    def __init__(self, llm_model: str = "qwen2.5:3b", api_key: str = None, api_base: str = None, use_openai: bool = False):
        #chatpdf存储路径
        self.storage_directory = 'chroma_db'
        os.environ["OPENAI_API_KEY"] = 'sk-gKrFRQIZQWfNyAy949C7B08180C2453086Cc61C8355fB64f'
        os.environ["OPENAI_API_BASE"] = 'https://api.bianxie.ai/v1'
        if use_openai:
            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["OPENAI_API_BASE"] = api_base
            self.model = OpenAI(temperature=0.9)
        else:
            self.model = ChatOllama(model=llm_model)

        #增加的多PDFbm25检索
        self.all_docs = [] 
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
        self.arxiv_vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
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

    def search_recent_arxiv(self, query: str, time_period: str = "month", max_results: int = 10):
        current_date = datetime.now().replace(tzinfo=None)
        if time_period == "week":
            start_date = current_date - timedelta(weeks=1)
        else:
            start_date = current_date - timedelta(days=30)
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        results = []
        for result in search.results():
            published_date = result.published.replace(tzinfo=None)
            if published_date > start_date:
                # 获取论文标识符
                paper_id = result.entry_id.split('/')[-1]
               
                
                results.append({
                    "id": paper_id,
                    "title": result.title,
                    "authors": ", ".join(author.name for author in result.authors),
                    "summary": result.summary,
                    "published": published_date,
                    
                    "pdf_url": result.pdf_url,
                })
            if len(results) >= max_results:
                break

        return results

    def ingest_arxiv(self, paper_id: str, doc_metadata: dict):
        try:
            full_text_docs = self.arxiv_retriever.get_relevant_documents(paper_id)
            if full_text_docs:
                full_text = full_text_docs[0].page_content
            else:
                full_text = "Full text not available."
            
            chunks = self.text_splitter.split_text(full_text)
            new_docs = [Document(page_content=chunk, metadata=doc_metadata) for chunk in chunks]
            

            if self.arxiv_vector_store is None:
                self.arxiv_vector_store = FAISS.from_documents(
                    documents=new_docs,
                    embedding=self.embeddings
                )
            else:
                self.arxiv_vector_store.add_documents(new_docs)
            #多PDF的bm25检索
            self.all_docs.extend(new_docs)
            self.bm25_retriever = BM25Retriever.from_documents(self.all_docs)
            
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.arxiv_vector_store.as_retriever(), self.bm25_retriever],
                weights=[0.5, 0.5]
            )
            print(f"Successfully ingested paper: {paper_id}")
            return True
        except Exception as e:
            print(f"Error ingesting paper {paper_id}: {str(e)}")
            return False

    def ask_local_arxiv(self, query: str):
        if self.ensemble_retriever is None:
            return "No documents in the local knowledge base. Please add some documents first."
        
        retrieved_docs = self.ensemble_retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
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
        self.arxiv_knowledge_base = []
        self.arxiv_vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None

    def get_arxiv_metadata(self, query: str, max_docs: int = 2):
        self.arxiv_retriever.load_max_docs = max_docs
        docs = self.arxiv_retriever.get_relevant_documents(query)
        return [{"title": doc.metadata.get("Title"), "authors": doc.metadata.get("Authors"), "summary": doc.metadata.get("Summary")} for doc in docs]