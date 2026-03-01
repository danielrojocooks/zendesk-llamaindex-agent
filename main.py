import os
from fastapi import FastAPI
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import ReActAgent

load_dotenv()

app = FastAPI()

index = None
agent = None


@app.on_event("startup")
def startup_event():
    global index, agent

    # Load documents from docs folder
    documents = SimpleDirectoryReader("docs").load_data()

    # Build vector index
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=3)

    # Create RAG tool
    rag_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="kb_search",
        description="Search internal documentation."
    )

    # LLM
    llm = OpenAI(model="gpt-4.1-mini", temperature=0)

    # ReAct agent
    agent = ReActAgent(
    tools=[rag_tool],
    llm=llm,
    verbose=True
)
    


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/test")
async def test():
    handler = agent.run("What are the key themes in the documentation?")
    result = await handler
    return {"response": str(result)}

from pydantic import BaseModel

class ZendeskTicket(BaseModel):
    subject: str
    description: str
    requester_email: str | None = None


@app.post("/zendesk")
async def zendesk(ticket: ZendeskTicket):

    query = f"""
    ...
    """

response = agent.run(query)

    import json

    if isinstance(response, dict):
        return response

    if isinstance(response, str):
        return json.loads(response)

    if hasattr(response, "response"):
        return json.loads(response.response)

    return {"error": "Unexpected response type"}