import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
import anthropic
from embeddings import build_knowledge_base, retrieve_context

load_dotenv()
build_knowledge_base()

# ── Database ─────────────────────────────────────────────────────
DB_PATH = "railway_mock.db"

def get_schema() -> str:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    schema_parts = []
    for (table_name,) in tables:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        col_names = [f"{col[1]} ({col[2]})" for col in columns]
        schema_parts.append(
            f"Table: {table_name}\nColumns: {', '.join(col_names)}"
        )
    conn.close()
    return "\n\n".join(schema_parts)

def execute_sql(sql: str) -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return {"success": True, "data": df.to_dict(orient="records"),
                "rows": len(df)}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ── SQL Validator ────────────────────────────────────────────────
BLOCKED = ["DROP", "DELETE", "UPDATE", "INSERT", "TRUNCATE", "ALTER"]

def validate_sql(sql: str) -> dict:
    sql_upper = sql.upper().strip()
    for keyword in BLOCKED:
        if keyword in sql_upper:
            return {"valid": False, "reason": f"Unsafe operation: {keyword}"}
    if not sql_upper.startswith("SELECT"):
        return {"valid": False, "reason": "Only SELECT queries allowed"}
    return {"valid": True, "reason": "OK"}

# ── Claude API ───────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Agent State ──────────────────────────────────────────────────
class AgentState(TypedDict):
    question: str
    schema: str
    rag_context: str
    sql: str
    validation: dict
    result: dict
    final_answer: str
    retry_count: int
    conversation_history: List[dict]

# ── Nodes ────────────────────────────────────────────────────────
def node_fetch_schema(state: AgentState) -> AgentState:
    state["schema"] = get_schema()
    print("📋 Schema fetched")
    return state

def node_retrieve_rag(state: AgentState) -> AgentState:
    state["rag_context"] = retrieve_context(state["question"], n_results=3)
    print("🔍 RAG context retrieved")
    return state

def node_generate_sql(state: AgentState) -> AgentState:
    system_prompt = f"""You are an expert SQL analyst for a railway infrastructure
database. Help engineers query track maintenance and ballast data.

DATABASE SCHEMA:
{state['schema']}

DOMAIN KNOWLEDGE:
{state['rag_context']}

RULES:
- Write a single valid SQLite SELECT query only
- Never use DROP, DELETE, UPDATE, INSERT, TRUNCATE or ALTER
- Use conversation history to resolve references like "those segments"
  or "that subdivision"
- Return ONLY the raw SQL, no explanation, no markdown, no backticks"""

    messages = []
    for turn in state["conversation_history"]:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant",
                         "content": f"SQL: {turn['sql']}\nAnswer: {turn['answer']}"})

    messages.append({
        "role": "user",
        "content": f"Question: {state['question']}\n\nReturn only the SQL query:"
    })

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=messages
    )

    sql = response.content[0].text.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    state["sql"] = sql
    print(f"🔧 SQL: {sql}")
    return state

def node_validate_sql(state: AgentState) -> AgentState:
    state["validation"] = validate_sql(state["sql"])
    print(f"✅ Validation: {state['validation']}")
    return state

def node_execute_sql(state: AgentState) -> AgentState:
    if not state["validation"]["valid"]:
        state["result"] = {"success": False,
                           "error": state["validation"]["reason"]}
        return state
    state["result"] = execute_sql(state["sql"])
    print(f"📊 Rows: {state['result'].get('rows', 0)}")
    return state

def node_self_reflect(state: AgentState) -> AgentState:
    print(f"🔄 Retry attempt {state['retry_count'] + 1}")
    fix_prompt = f"""Query failed: {state['result']['error']}
Question: {state['question']}
Failed SQL: {state['sql']}
Schema: {state['schema']}
Write a corrected SQLite SELECT query only, no markdown:"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": fix_prompt}]
    )
    sql = response.content[0].text.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    state["sql"] = sql
    state["retry_count"] += 1
    return state

def node_format_answer(state: AgentState) -> AgentState:
    if not state["result"].get("success"):
        state["final_answer"] = (
            f"I wasn't able to answer that. "
            f"Error: {state['result'].get('error', 'Unknown')}. "
            f"Please try rephrasing."
        )
        return state

    messages = []
    for turn in state["conversation_history"]:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})

    messages.append({
        "role": "user",
        "content": f"""User asked: {state['question']}
Query returned {state['result']['rows']} rows: {state['result']['data'][:10]}
Domain context: {state['rag_context']}

Write a clear, friendly 2-3 sentence answer. Reference conversation
history naturally if relevant."""
    })

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=messages
    )
    state["final_answer"] = response.content[0].text.strip()
    return state

# ── Routing ──────────────────────────────────────────────────────
def should_retry(state: AgentState) -> str:
    if not state["result"].get("success") and state["retry_count"] < 2:
        return "self_reflect"
    return "format_answer"

# ── Build Graph ──────────────────────────────────────────────────
def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("fetch_schema",  node_fetch_schema)
    graph.add_node("retrieve_rag",  node_retrieve_rag)
    graph.add_node("generate_sql",  node_generate_sql)
    graph.add_node("validate_sql",  node_validate_sql)
    graph.add_node("execute_sql",   node_execute_sql)
    graph.add_node("self_reflect",  node_self_reflect)
    graph.add_node("format_answer", node_format_answer)
    graph.set_entry_point("fetch_schema")
    graph.add_edge("fetch_schema",  "retrieve_rag")
    graph.add_edge("retrieve_rag",  "generate_sql")
    graph.add_edge("generate_sql",  "validate_sql")
    graph.add_edge("validate_sql",  "execute_sql")
    graph.add_conditional_edges("execute_sql", should_retry)
    graph.add_edge("self_reflect",  "generate_sql")
    graph.add_edge("format_answer", END)
    return graph.compile()

agent = build_agent()

# ── Quick test ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚂 Railway SQL Agent — Memory + RAG\n")
    history = []
    questions = [
        "Which segments have CRITICAL status?",
        "How many of those are in the Clovis subdivision?",
        "What maintenance work has been done on those segments?"
    ]
    for q in questions:
        print(f"\n❓ {q}")
        result = agent.invoke({
            "question": q,
            "schema": "",
            "rag_context": "",
            "sql": "",
            "validation": {},
            "result": {},
            "final_answer": "",
            "retry_count": 0,
            "conversation_history": history
        })
        print(f"💬 {result['final_answer']}")
        history.append({
            "question": q,
            "sql": result["sql"],
            "answer": result["final_answer"]
        })