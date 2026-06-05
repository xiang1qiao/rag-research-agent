import os
import shutil

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import UPLOAD_DIR
from app.document_loader import load_document
from app.rag import add_document_to_vector_db, answer_question, clear_vector_db, clear_memory, get_memory
from app.agent import run_agent
from app.graph_agent import run_graph_agent

app = FastAPI(title="RAG 文档助手")



@app.get("/")
def home():
    return FileResponse("app/index.html")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = load_document(file_path)

    chunk_count = add_document_to_vector_db(text, file.filename)

    return {
        "message": "文件上传并入库成功",
        "filename": file.filename,
        "chunk_count": chunk_count
    }


@app.post("/ask")
async def ask_question(question: str = Form(...)):
    result = answer_question(question)

    return result

@app.post("/clear")
async def clear_db():
    return clear_vector_db()

@app.post("/clear-memory")
async def clear_chat_memory():
    return clear_memory()


@app.get("/memory")
async def read_memory():
    return get_memory()

@app.post("/agent")
async def agent(question: str = Form(...)):
    return run_agent(question)


@app.post("/graph-agent")
async def graph_agent(question: str = Form(...)):
    return run_graph_agent(question)