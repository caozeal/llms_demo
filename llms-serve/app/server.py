import os
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes

from app.reranker_server import compression_retriever
from rag_conversation.chain import chain

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, compression_retriever, path="/retriever")
add_routes(app, chain, path="/rag-conversation")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
