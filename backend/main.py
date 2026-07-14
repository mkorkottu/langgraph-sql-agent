from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from agent import agent

app = FastAPI(title="Railway SQL Agent API")

# Allow React frontend to talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://20.118.126.174", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    conversation_history: List[dict] = []

class QueryResponse(BaseModel):
    answer: str
    sql: str
    rag_context: str
    rows: int
    data: list
    success: bool

@app.get("/")
def root():
    return {"status": "Railway SQL Agent API is running"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    result = agent.invoke({
        "question": request.question,
        "schema": "",
        "rag_context": "",
        "sql": "",
        "validation": {},
        "result": {},
        "final_answer": "",
        "retry_count": 0,
        "conversation_history": request.conversation_history
    })

    return QueryResponse(
        answer=result["final_answer"],
        sql=result["sql"],
        rag_context=result["rag_context"],
        rows=result["result"].get("rows", 0),
        data=result["result"].get("data", []),
        success=result["result"].get("success", False)
    )

@app.get("/health")
def health():
    return {"status": "ok"}