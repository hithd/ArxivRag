import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_community.retrievers import ArxivRetriever
from langchain_community.retrievers import ArxivRetriever
from langchain_community.document_loaders import ArxivLoader
loader = ArxivLoader(
    query="2305.08386",
    load_max_docs=2,
    # doc_content_chars_max=1000,
    # load_all_available_meta=False,
    # ...
)
docs = loader.load()
print(docs[0].page_content)
os.environ["OPENAI_API_KEY"] = 'sk-gKrFRQIZQWfNyAy949C7B08180C2453086Cc61C8355fB64f'
os.environ["OPENAI_API_BASE"] = 'https://api.bianxie.ai/v1'