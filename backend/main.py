from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

import uuid
import os
from utils import document_processing

app = FastAPI()

# Allow CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "../data"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    with open(file_path, "wb") as f:
        f.write(await file.read())
    # Process document and store in vector DB
    try:
        page_contents = document_processing.extract_and_analyze_pages(file_path)
        index_path = os.path.join(UPLOAD_DIR, f"{file_id}_faiss_index")
        document_processing.store_in_vector_db(page_contents, index_path=index_path)
        return {"file_id": file_id, "filename": file.filename, "status": "processed"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/ask")
async def ask_question(file_id: str = Form(...), query: str = Form(...)):
    # Load vector DB and answer question
    try:
        # Find the index path for this file_id
        index_path = None
        for fname in os.listdir(UPLOAD_DIR):
            if fname.startswith(f"{file_id}_faiss_index"):
                index_path = os.path.join(UPLOAD_DIR, fname)
                break
        if not index_path:
            raise FileNotFoundError("Vector DB for this file_id not found.")
        vectorstore = document_processing.load_vector_db(index_path=index_path)
        # Use langchain RAG pipeline to answer
        from langchain.chains import RetrievalQA
        from langchain_google_genai import ChatGoogleGenerativeAI
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatGoogleGenerativeAI(
                model="models/gemini-1.5-flash",
                api_key=os.getenv("GEMINI_API_KEY"),
            ),
            retriever=retriever,
            return_source_documents=True,
        )
        result = qa_chain.invoke({"query": query})
        sources = [doc.metadata for doc in result["source_documents"]]
        return {"answer": result["result"], "sources": sources}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
def health():
    return {"status": "ok"}
