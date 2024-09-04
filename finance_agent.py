import os
import json
import operator
import pandas as ps
from io import StringIO
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated

memory = SqliteSaver.from_conn_string(":memory:")

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY2"))
