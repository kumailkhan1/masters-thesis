
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llm import query_llm

app = FastAPI()

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust this list if you have other origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class RetrievedNode(BaseModel):
    metadata: dict
    score: float

class QueryResponse(BaseModel):
    response: str
    retrieved_nodes: List[RetrievedNode]


@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    query = request.query
    try:
        data = await query_llm(query)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
