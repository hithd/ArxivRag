from datetime import datetime, timedelta
import arxiv
from langchain_community.retrievers import BM25Retriever, ArxivRetriever
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
class ArxivHandler:
    def __init__(self, embeddings, bm25_retriever=None, ensemble_retriever=None):
        self.arxiv_retriever = ArxivRetriever(load_max_docs=2, get_full_documents=True)
        self.arxiv_vector_store = None
        self.bm25_retriever = bm25_retriever
        self.ensemble_retriever = ensemble_retriever
        self.embeddings = embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.all_docs = []

    def ask_arxiv(self, query: str, model, prompt, max_docs: int = 2):
        self.arxiv_retriever.load_max_docs = max_docs
        docs = self.arxiv_retriever.get_relevant_documents(query)
        context = "\n\n".join(doc.page_content for doc in docs)
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
        answer = chain.invoke({"context": context, "question": query})
    
        sources = []
        for doc in docs:
            metadata = doc.metadata
            print(metadata)
            source = {
                "title": metadata.get('Title', 'Unknown Title'),
                "authors": metadata.get('Authors', 'Unknown Authors'),
                "published": metadata.get('Published', 'Unknown Date'),
                
            }
            sources.append(source)

        return answer, sources

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

    def ask_local_arxiv(self, query: str, model, prompt):
        if self.ensemble_retriever is None:
            return "No documents in the local knowledge base. Please add some documents first."
        retrieved_docs = self.ensemble_retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
        return chain.invoke({"context": context, "question": query})

    def get_arxiv_metadata(self, query: str, max_docs: int = 2):
        self.arxiv_retriever.load_max_docs = max_docs
        docs = self.arxiv_retriever.get_relevant_documents(query)
        return [{"title": doc.metadata.get("Title"), "authors": doc.metadata.get("Authors"), "summary": doc.metadata.get("Summary")} for doc in docs]
