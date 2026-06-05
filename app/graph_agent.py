from typing import TypedDict, Literal

from langgraph.graph import StateGraph, START, END

from app.rag import call_llm
from app.tools import search_docs, calculator, web_search


class AgentState(TypedDict):
    question: str
    tool: str
    tool_result: object
    answer: str
    review: str


def planner_node(state: AgentState) -> AgentState:
    question = state["question"]

    messages = [
        {
            "role": "system",
            "content": """
你是一个Agent规划器。你需要根据用户问题选择一个工具。

只能选择以下工具之一：
search_docs：适合回答上传文档、知识库、简历、论文、资料相关问题
calculator：适合数学计算、表达式计算
web_search：适合最新新闻、实时信息、外部互联网信息

只返回工具名，不要解释。
"""
        },
        {
            "role": "user",
            "content": question
        }
    ]

    tool = call_llm(messages, temperature=0).strip()

    if "calculator" in tool:
        selected_tool = "calculator"
    elif "web_search" in tool:
        selected_tool = "web_search"
    else:
        selected_tool = "search_docs"

    return {
        **state,
        "tool": selected_tool
    }


def route_tool(state: AgentState) -> Literal["search_docs_node", "calculator_node", "web_search_node"]:
    if state["tool"] == "calculator":
        return "calculator_node"

    if state["tool"] == "web_search":
        return "web_search_node"

    return "search_docs_node"


def search_docs_node(state: AgentState) -> AgentState:
    question = state["question"]

    docs = search_docs(question)

    return {
        **state,
        "tool_result": docs
    }


def calculator_node(state: AgentState) -> AgentState:
    question = state["question"]

    result = calculator(question)

    return {
        **state,
        "tool_result": result
    }


def web_search_node(state: AgentState) -> AgentState:
    question = state["question"]

    result = web_search(question)

    return {
        **state,
        "tool_result": result
    }


def answer_node(state: AgentState) -> AgentState:
    question = state["question"]
    tool = state["tool"]
    tool_result = state["tool_result"]

    messages = [
        {
            "role": "system",
            "content": """
你是一个Research Agent的回答生成节点。
你需要根据工具调用结果回答用户问题。

要求：
1. 回答要清晰、有逻辑。
2. 如果工具结果为空，要明确说明没有找到可靠信息。
3. 不要编造工具结果中不存在的信息。
"""
        },
        {
            "role": "user",
            "content": f"""
用户问题：
{question}

使用工具：
{tool}

工具返回结果：
{tool_result}

请生成最终回答。
"""
        }
    ]

    answer = call_llm(messages, temperature=0.2)

    return {
        **state,
        "answer": answer
    }


def reviewer_node(state: AgentState) -> AgentState:
    question = state["question"]
    answer = state["answer"]
    tool = state["tool"]

    messages = [
        {
            "role": "system",
            "content": """
你是一个Agent回答检查节点。
你需要检查回答是否基本回答了用户问题。

只输出一句简短检查结果。
"""
        },
        {
            "role": "user",
            "content": f"""
用户问题：
{question}

Agent使用工具：
{tool}

Agent回答：
{answer}

请给出检查结果。
"""
        }
    ]

    review = call_llm(messages, temperature=0)

    return {
        **state,
        "review": review
    }


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner_node", planner_node)
    graph.add_node("search_docs_node", search_docs_node)
    graph.add_node("calculator_node", calculator_node)
    graph.add_node("web_search_node", web_search_node)
    graph.add_node("answer_node", answer_node)
    graph.add_node("reviewer_node", reviewer_node)

    graph.add_edge(START, "planner_node")

    graph.add_conditional_edges(
        "planner_node",
        route_tool,
        {
            "search_docs_node": "search_docs_node",
            "calculator_node": "calculator_node",
            "web_search_node": "web_search_node"
        }
    )

    graph.add_edge("search_docs_node", "answer_node")
    graph.add_edge("calculator_node", "answer_node")
    graph.add_edge("web_search_node", "answer_node")

    graph.add_edge("answer_node", "reviewer_node")
    graph.add_edge("reviewer_node", END)

    return graph.compile()


graph_agent = build_graph()


def run_graph_agent(question: str) -> dict:
    initial_state = {
        "question": question,
        "tool": "",
        "tool_result": None,
        "answer": "",
        "review": ""
    }

    result = graph_agent.invoke(initial_state)

    return result