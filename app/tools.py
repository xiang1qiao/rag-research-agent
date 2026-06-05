import requests

from app.rag import search_relevant_chunks


def search_docs(query: str):
    results = search_relevant_chunks(
        question=query,
        top_k=3,
        recall_k=10
    )

    return results


def calculator(expression: str):
    try:
        result = eval(expression)

        return {
            "result": result
        }

    except Exception as e:
        return {
            "error": str(e)
        }


def web_search(query: str):
    return {
        "message": f"模拟搜索结果：{query}",
        "result": "这里后续接入真实搜索API"
    }