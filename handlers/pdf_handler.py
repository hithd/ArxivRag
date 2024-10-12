import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
class PDFHandler:
    def __init__(self, embeddings, storage_directory='chroma_db'):
        self.vector_stores = {}
        self.embeddings = embeddings
        self.storage_directory = storage_directory
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)

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
            persist_directory=pdf_directory,
        )

    def ask_pdf(self, query: str, pdf_name: str, model, prompt):
        if pdf_name not in self.vector_stores:
            return "Please add the specified PDF document first."
        retriever = self.vector_stores[pdf_name].as_retriever(
            search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}
        )
        context = retriever.invoke(query)
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
        return chain.invoke({"context": context, "question": query})
