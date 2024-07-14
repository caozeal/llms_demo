import os
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# Split
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX", "rag-conversation")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Add to vectorDB
vectorstore = PineconeVectorStore.from_documents(
    documents=all_splits, embedding = HuggingFaceEmbeddings(), index_name=PINECONE_INDEX_NAME
)
retriever = vectorstore.as_retriever()