from utils.utils import initialize_env
from handlers.pdf_handler import PDFHandler
from handlers.arxiv_handler import ArxivHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

class ChatPDF:
    def __init__(self, llm_model: str = "qwen2.5:3b", api_key: str = None, api_base: str = None, use_openai: bool = False):
        self.model, self.embeddings = initialize_env(api_key, api_base, use_openai, llm_model)
        self.pdf_handler = PDFHandler(self.embeddings)
        self.arxiv_handler = ArxivHandler(self.embeddings)

        self.prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the context provided.

            Context: {context}

            Question: {question}"""
        )

    def ingest(self, pdf_file_path: str, pdf_name: str):
        self.pdf_handler.ingest(pdf_file_path, pdf_name)

    def ask_pdf(self, query: str, pdf_name: str):
        return self.pdf_handler.ask_pdf(query, pdf_name, self.model, self.prompt)

    def ask_arxiv(self, query: str, max_docs: int = 2):
        return self.arxiv_handler.ask_arxiv(query, self.model, self.prompt, max_docs)

    def search_recent_arxiv(self, query: str, time_period: str = "month", max_results: int = 10):
        return self.arxiv_handler.search_recent_arxiv(query, time_period, max_results)

    def ingest_arxiv(self, paper_id: str, doc_metadata: dict):
        return self.arxiv_handler.ingest_arxiv(paper_id, doc_metadata)

    def ask_local_arxiv(self, query: str):
        return self.arxiv_handler.ask_local_arxiv(query, self.model, self.prompt)

    def clear(self):
        self.pdf_handler = PDFHandler(self.embeddings)
        self.arxiv_handler = ArxivHandler(self.embeddings)

    def get_arxiv_metadata(self, query: str, max_docs: int = 2):
        return self.arxiv_handler.get_arxiv_metadata(query, max_docs)
