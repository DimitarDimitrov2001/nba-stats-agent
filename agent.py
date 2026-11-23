"""Give the model read-only SQL tools and collect the results of each turn."""

import os
import re
import sqlite3
from contextlib import closing
from functools import cache
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import AIMessage, ToolMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

from build_db import DB_PATH

MODEL = "gpt-4o-mini"
SELECT_QUERY = re.compile(r"\s*(SELECT|WITH)\b", re.IGNORECASE)
TOOL_STEPS = {
    "sql_db_list_tables": ("Listing tables", "Available tables"),
    "sql_db_schema": ("Inspecting the table schema", "Table schema"),
    "sql_db_query_checker": ("Checking SQL for common mistakes", "SQL check result"),
    "sql_db_query": ("Running read-only SQL", "Database result"),
}


def _read_only_connection():
    if not DB_PATH.is_file():
        raise FileNotFoundError("Run python build_db.py to create the database first.")
    db = sqlite3.connect(f"{DB_PATH.as_uri()}?mode=ro", uri=True)
    db.execute("PRAGMA query_only=ON")
    return db


@wrap_tool_call
def _select_only(request, handler):
    call = request.tool_call
    query = call["args"].get("query")
    if call["name"] == "sql_db_query" and isinstance(query, str) and not SELECT_QUERY.match(query):
        return ToolMessage(
            content="Only SELECT queries are allowed.", tool_call_id=call["id"], status="error"
        )
    return handler(request)


@cache
def _get_agent():
    # Load once on the first question so the app can show setup errors in chat.
    with closing(_read_only_connection()) as db:
        ranges = db.execute("""
            SELECT season_type, MIN(season), MAX(season), MIN(game_date), MAX(game_date)
            FROM player_games GROUP BY season_type
        """).fetchall()
    coverage = "\n".join(
        f"{kind} spans seasons {first} through {last}, with games from {first_date} through {last_date}."
        for kind, first, last, first_date, last_date in ranges
    )

    load_dotenv(Path(__file__).with_name(".env"))
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY in .env and try again.")
    model = ChatOpenAI(model=MODEL, temperature=0, timeout=30)
    engine = create_engine("sqlite://", creator=_read_only_connection, poolclass=NullPool)
    db = SQLDatabase(engine, include_tables=["player_games"], sample_rows_in_table_info=0)
    tools = SQLDatabaseToolkit(db=db, llm=model).get_tools()

    prompt = f"""
    Answer NBA statistics questions in plain English using the SQLite database.
    Query player_games, the only table. Use its actual schema when writing SQL.
    {db.get_table_info()}

    Use SELECT (WITH is allowed). Check SQL with sql_db_query_checker before running it.
    Limit multi-row results to 10 by default.

    Default to Regular Season unless the user asks for Playoffs. Each row is one player
    in one game. Count distinct game_id for game counts. Group players by player_id and
    use player_name for display. Use team_id across seasons for a franchise because
    abbreviations may change, such as NJN/BKN and NOH/NOP. Use total makes divided by
    total attempts for aggregate shooting percentages.

    The database covers these dates and seasons.
    {coverage}
    The most recent season may be in progress. Identify partial totals or averages and
    include the latest available date when it matters. Check that the requested slice
    has data. Say when the requested season is outside the available range or the data
    lacks a statistic. Base numbers on query results and never guess."""
    return create_agent(model, tools, system_prompt=prompt, middleware=[_select_only])


def stream_ask(question, history):
    """Yield the current answer, SQL queries, and user-visible tool activity."""
    messages = [{"role": message["role"], "content": message["content"]} for message in history]
    messages.append({"role": "user", "content": question})

    answer = ""
    sql_calls = {}
    queries = []
    activity = ["Reading your question."]
    yield answer, [], activity[0]

    # Each snapshot includes earlier messages. Process only the new ones.
    seen = len(messages)
    for state in _get_agent().stream(
        {"messages": messages}, config={"recursion_limit": 24}, stream_mode="values"
    ):
        new_messages = state["messages"][seen:]
        if not new_messages:
            continue
        seen += len(new_messages)
        for message in new_messages:
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    name = call["name"]
                    step, _ = TOOL_STEPS.get(name, (name, name))
                    query = call["args"].get("query")
                    details = query or call["args"].get("table_names")
                    activity.append(f"{step}\n{details}" if details else step)
                    if name == "sql_db_query" and query is not None:
                        sql_calls[call["id"]] = query
                if not message.tool_calls:
                    answer = message.text
                    activity.append("Answer ready.")
            elif isinstance(message, ToolMessage):
                if message.status == "success" and message.tool_call_id in sql_calls:
                    queries.append(sql_calls[message.tool_call_id])
                _, label = TOOL_STEPS.get(message.name, ("", "Tool result"))
                output = message.text
                if len(output) > 1200:
                    output = output[:1200] + "\n…"
                activity.append(f"{label}\n{output}")
        yield answer, queries.copy(), "\n\n".join(activity)


def ask(question, history):
    """Return the final answer and SQL for scripts and evaluation."""
    for answer, queries, _ in stream_ask(question, history):
        pass
    return answer, queries
