import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Helper function for printing docs


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = WebBaseLoader(
    "https://jina.ai/reranker/"
).load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

embedding = JinaEmbeddings(model_name="jina-embeddings-v2-base-en")
retriever = FAISS.from_documents(texts, embedding).as_retriever(search_kwargs={"k": 20})

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import time

t1 = time.time_ns() / 1_000_000

from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import JinaRerank

compressor = JinaRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.get_relevant_documents(
    "what is reranker?"
)
pretty_print_docs(compressed_docs)
t2 = time.time_ns() / 1_000_000
print("used", t2-t1, "ms")