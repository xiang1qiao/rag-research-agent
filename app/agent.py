from app.rag import call_llm
from app.tools import (
    search_docs,
    calculator,
    web_search
)
def select_tool(question: str):
    messages = [
        {
            "role": "system",
            "content": """
你是Agent工具规划器。

你只能返回以下三个工具名字之一：

search_docs
calculator
web_search

规则：

知识库问题：
search_docs

数学计算：
calculator

互联网信息：
web_search

只能返回工具名字。
"""
        },
        {
            "role": "user",
            "content": question
        }
    ]

    result = call_llm(messages, temperature=0)

    return result.strip()

def run_agent(question: str):
    tool = select_tool(question)

    if "calculator" in tool:
        tool_result = calculator(question)

        return {
            "tool": "calculator",
            "tool_result": tool_result,
            "answer": str(tool_result)
        }

    elif "web_search" in tool:
        tool_result = web_search(question)

        return {
            "tool": "web_search",
            "tool_result": tool_result,
            "answer": str(tool_result)
        }

    else:
        docs = search_docs(question)

        context = ""

        for item in docs:
            context += item["content"] + "\n\n"

        messages = [
            {
                "role": "system",
                "content": "根据检索结果回答用户问题。"
            },
            {
                "role": "user",
                "content": f"""
用户问题：

{question}

检索结果：

{context}
"""
            }
        ]

        answer = call_llm(messages)

        return {
            "tool": "search_docs",
            "tool_result": docs,
            "answer": answer
        }