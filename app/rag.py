import requests
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.config import DEEPSEEK_API_KEY, DEEPSEEK_API_URL, VECTOR_DB_DIR


embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
rerank_model = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

collection = client.get_or_create_collection(
    name="document_collection"
)

conversation_history = []


def split_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


def add_document_to_vector_db(text: str, filename: str) -> int:
    chunks = split_text(text)

    existing = collection.get(where={"filename": filename})

    if existing and existing.get("ids"):
        collection.delete(where={"filename": filename})

    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()

        collection.add(
            ids=[f"{filename}_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"filename": filename, "chunk_index": i}]
        )

    return len(chunks)


def format_history(max_rounds: int = 5) -> str:
    recent_history = conversation_history[-max_rounds:]

    if not recent_history:
        return "暂无历史对话。"

    history_text = ""

    for item in recent_history:
        history_text += f"用户：{item['question']}\n"
        history_text += f"助手：{item['answer']}\n\n"

    return history_text.strip()


def call_llm(messages: list[dict], temperature: float = 0.2) -> str:
    if not DEEPSEEK_API_KEY:
        return "错误：没有配置 DEEPSEEK_API_KEY，请检查 .env 文件。"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": temperature
    }

    response = requests.post(
        DEEPSEEK_API_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        return f"调用 DeepSeek 失败：{response.text}"

    data = response.json()

    return data["choices"][0]["message"]["content"]


def rewrite_question(question: str) -> str:
    history_text = format_history()

    messages = [
        {
            "role": "system",
            "content": "你是一个搜索查询改写助手。你的任务是根据历史对话，把用户当前问题改写成一个独立、完整、适合检索的问题。"
        },
        {
            "role": "user",
            "content": f"""
历史对话：
{history_text}

当前问题：
{question}

请把当前问题改写成一个独立完整的问题。

要求：
1. 如果当前问题已经完整，直接返回原问题。
2. 如果当前问题包含“他、她、它、这个、那个、上面、刚才”等指代词，请结合历史对话补全。
3. 只输出改写后的问题，不要解释。
"""
        }
    ]

    rewritten = call_llm(messages, temperature=0.0)

    if rewritten.startswith("调用 DeepSeek 失败") or rewritten.startswith("错误："):
        return question

    return rewritten.strip()


def search_relevant_chunks(question: str, top_k: int = 4, recall_k: int = 10) -> list[dict]:
    question_embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=recall_k
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    candidates = []

    for doc, metadata in zip(documents, metadatas):
        candidates.append({
            "content": doc,
            "filename": metadata.get("filename", "未知文件"),
            "chunk_index": metadata.get("chunk_index", -1)
        })

    if not candidates:
        return []

    pairs = []

    for item in candidates:
        pairs.append([question, item["content"]])

    scores = rerank_model.predict(pairs)

    for item, score in zip(candidates, scores):
        item["rerank_score"] = float(score)

    candidates.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return candidates[:top_k]


def generate_answer(original_question: str, rewritten_question: str, sources: list[dict]) -> str:
    history_text = format_history()

    context = ""

    for source in sources:
        context += f"""
来源文件：{source["filename"]}
片段编号：{source["chunk_index"]}
片段内容：
{source["content"]}

"""

    messages = [
        {
            "role": "system",
            "content": "你是一个严谨、专业、不会编造的 RAG 文档问答助手。"
        },
        {
            "role": "user",
            "content": f"""
请根据文档内容回答用户问题。

要求：
1. 只能根据文档内容回答，不能编造。
2. 如果文档中没有答案，请回答：根据已上传文档，我没有找到相关信息。
3. 回答要清晰、分点、有逻辑。
4. 可以参考历史对话理解用户意图，但最终答案必须基于文档内容。
5. 回答最后说明：以上信息来自上传文档的相关片段。

历史对话：
{history_text}

用户原始问题：
{original_question}

检索改写后的问题：
{rewritten_question}

文档内容：
{context}
"""
        }
    ]

    return call_llm(messages, temperature=0.2)


def answer_question(question: str) -> dict:
    rewritten_question = rewrite_question(question)

    sources = search_relevant_chunks(rewritten_question)

    if not sources:
        answer = "当前知识库为空或没有检索到相关内容。"

        conversation_history.append({
            "question": question,
            "answer": answer
        })

        return {
            "answer": answer,
            "rewritten_question": rewritten_question,
            "sources": []
        }

    answer = generate_answer(question, rewritten_question, sources)

    conversation_history.append({
        "question": question,
        "answer": answer
    })

    if len(conversation_history) > 10:
        conversation_history.pop(0)

    return {
        "answer": answer,
        "rewritten_question": rewritten_question,
        "sources": sources
    }


def clear_vector_db() -> dict:
    all_data = collection.get()

    if all_data and all_data.get("ids"):
        collection.delete(ids=all_data["ids"])

    conversation_history.clear()

    return {
        "message": "知识库和对话记忆已清空"
    }


def clear_memory() -> dict:
    conversation_history.clear()

    return {
        "message": "对话记忆已清空"
    }


def get_memory() -> dict:
    return {
        "history": conversation_history
    }