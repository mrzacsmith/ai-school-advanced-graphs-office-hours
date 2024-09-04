import os
import json
import operator
import pandas as ps
from io import StringIO
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated, List

memory = SqliteSaver.from_conn_string(":memory:")

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
    competition: List[str]
    csv_file: str
    financial_data: str
    analysis: str
    competitor_data: str
    comparison: str
    feedback: str
    report: str
    content: List[str]
    revision_number: int
    max_revision_number: int


class Queries(BaseModel):
    queries: List[str]
