import os
import sys
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the PYTHONPATH from the .env file
sys.path.append(os.getenv("PYTHONPATH"))

from llm import query_llm, LLMResponse

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

@app.post("/query", response_model=LLMResponse)
async def handle_query(request: QueryRequest):
    query = request.query
    try:
        data = await query_llm(query, generate_queries_flag=True)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
