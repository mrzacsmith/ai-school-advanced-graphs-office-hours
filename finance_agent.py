import os
import json
import operator
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, List

# memory = SqliteSaver.from_conn_string(":memory:")
memory = MemorySaver()


load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY2")
tavily_key = os.getenv("TAVILY_API_KEY")

llm_name = "gpt-4o-mini"
llm_model = ChatOpenAI(
    openai_api_key=openai_key,
    model_name=llm_name,
    temperature=0,
)
tavily = TavilyClient(api_key=tavily_key)
# search = TavilySearchResults(k=10, tavily_api_key=tavily_key)


class AgentState(BaseModel):
    task: str
    competitors: List[str]
    csv_file: str
    financial_data: str = ""
    analysis: str = ""
    competitor_data: str = ""
    comparison: str = ""
    feedback: str = ""
    report: str = ""
    content: List[str]
    revision_number: int
    max_revision_number: int


class Queries(BaseModel):
    queries: List[str]


GATHER_FINANCIALS_PROMPT = """
You are a financial analyst. You are given a csv file with financial data for a company.
Your task is to extract the financial data and provide a report.
"""
ANALYZE_DATA_PROMPT = """
You are a financial analyst. You are given a csv file with financial data for a company.
Your task is to analyze the financial data and provide a report.
"""
RESEARCH_COMPETITORS_PROMPT = """
You are a financial analyst. You are given a csv file with financial data for a company.
Your task is to research the competitors of the company and provide a report.
"""
COMPETE_PERFORMANCE_PROMPT = """
You are a financial analyst. You are given a csv file with financial data for a company.
Your task is to complete the performance report.
"""
FEEDBACK_PROMPT = """
You are a financial analyst. You are given a csv file with financial data for a company.
Your task is to provide feedback on the performance report.
"""
WRITE_REPORT_PROMPT = """
You are a financial analyst. You are given a csv file with financial data for a company.
Your task is to write the financial report.
"""
RESEARCH_CRITIQUE_PROMPT = """
You are a financial analyst. You are given a csv file with financial data for a company.
Your task is to research and critique the financial report. Generate a max of 3 queries to search for more information.
"""


def gather_financials_node(state: AgentState):
    csv_file = state.csv_file
    df = pd.read_csv(StringIO(csv_file))

    financial_data_to_string = df.to_string(index=False)
    combined_content = (
        f"{state.task}\nHere is the financial data:\n{financial_data_to_string}"
    )

    messages = [
        SystemMessage(content=GATHER_FINANCIALS_PROMPT),
        HumanMessage(content=combined_content),
    ]

    response = llm_model.invoke(messages)
    financial_data = response.content
    return AgentState(**{**state.dict(), "financial_data": financial_data})


def analyze_data_node(state: AgentState):
    financial_data = state.financial_data
    messages = [
        SystemMessage(content=ANALYZE_DATA_PROMPT),
        HumanMessage(content=financial_data),
    ]
    response = llm_model.invoke(messages)
    analysis = response.content
    return AgentState(**{**state.dict(), "analysis": analysis})


def research_competitors_node(state: AgentState):
    content = state.content or []
    for competitor in state.competitors:
        # Create a prompt for the LLM to generate queries
        prompt = f"Generate search queries to research financial information about {competitor}. The queries should help gather data for comparison with our company (Awesome Software Inc.)."

        # Invoke the LLM with the prompt
        response = llm_model.invoke(
            [
                SystemMessage(content=RESEARCH_COMPETITORS_PROMPT),
                HumanMessage(content=prompt),
            ]
        )

        # Parse the response into a list of queries
        queries = [q.strip() for q in response.content.split("\n") if q.strip()]

        for query in queries:
            search_results = tavily.search(query=query, max_results=1)
            for r in search_results["results"]:
                content.append(r["content"])
    return AgentState(**{**state.dict(), "content": content})


def compare_performance_node(state: AgentState):
    content = "\n".join(state.content or [])
    user_message = HumanMessage(
        content=f"{state.task}\nHere is the financial analysis:\n{state.analysis}"
    )

    messages = [
        SystemMessage(content=COMPETE_PERFORMANCE_PROMPT.format(content=content)),
        user_message,
    ]
    response = llm_model.invoke(messages)
    return AgentState(
        **{
            **state.dict(),
            "comparison": response.content,
            "revision_number": state.revision_number + 1,
        }
    )


def collect_feedback_node(state: AgentState):
    messages = [
        SystemMessage(content=FEEDBACK_PROMPT),
        HumanMessage(content=state.comparison),
    ]
    response = llm_model.invoke(messages)
    return AgentState(**{**state.dict(), "feedback": response.content})


def research_critique_node(state: AgentState):
    queries = llm_model.with_structured_output(Queries).invoke(
        [
            SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=state.feedback),
        ]
    )
    content = state.content or []
    for q in queries.queries:
        search_results = tavily.search(query=q, max_results=2)
        for r in search_results["results"]:
            content.append(r["content"])
    return AgentState(**{**state.dict(), "content": content})


def write_report_node(state: AgentState):
    messages = [
        SystemMessage(content=WRITE_REPORT_PROMPT),
        HumanMessage(content=state.comparison),
    ]
    response = llm_model.invoke(messages)
    return AgentState(**{**state.dict(), "report": response.content})


def should_continue(state: AgentState):
    if state.revision_number > state.max_revision_number:
        return END
    return "collect_feedback"


flow = StateGraph(AgentState)
flow.add_node("gather_financials", gather_financials_node)
flow.add_node("analyze_data", analyze_data_node)
flow.add_node("research_competitors", research_competitors_node)
flow.add_node("compare_performance", compare_performance_node)
flow.add_node("collect_feedback", collect_feedback_node)
flow.add_node("research_critique", research_critique_node)
flow.add_node("write_report", write_report_node)
flow.set_entry_point("gather_financials")
flow.add_conditional_edges(
    "compare_performance",
    should_continue,
    {END: END, "collect_feedback": "collect_feedback"},
)

flow.add_edge("gather_financials", "analyze_data")
flow.add_edge("analyze_data", "research_competitors")
flow.add_edge("research_competitors", "compare_performance")
flow.add_edge("collect_feedback", "research_critique")
flow.add_edge("research_critique", "compare_performance")
flow.add_edge("compare_performance", "write_report")

graph = flow.compile(checkpointer=memory)


def read_csv_file(csv_file_path: str) -> str:
    with open(csv_file_path, "r") as file:
        print("Reading CSV file...")
        return file.read()


if __name__ == "__main__":
    task = "Analyze the financial data for our company (Awesome Software Inc.) comparing to our competitors."
    competitors = ["invidia"]
    csv_file_path = "./data/financial_data.csv"

    if not os.path.exists(csv_file_path):
        print(f"File {csv_file_path} does not exist")
    else:
        print("starting...")
        csv_data = read_csv_file(csv_file_path)

        initial_state = AgentState(
            task=task,
            competitors=competitors,
            csv_file=csv_data,
            revision_number=1,
            max_revision_number=2,
            financial_data="",
            analysis="",
            competitor_data="",
            comparison="",
            feedback="",
            report="",
            content=[],
        )

    thread = {"configurable": {"thread_id": "1"}}

    for s in graph.stream(initial_state, thread):
        print(s)
