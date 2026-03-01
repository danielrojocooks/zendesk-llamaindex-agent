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
    You are a Zendesk AI support agent.

    Determine:
    1. Whether this ticket is answerable from documentation.
    2. If yes, generate a professional customer reply.
    3. If not, recommend escalation and summarize why.

    Ticket:
    Subject: {ticket.subject}
    Description: {ticket.description}

    Respond in this exact JSON format:

    {{
        "answerable": true or false,
        "customer_reply": "...",
        "internal_summary": "...",
        "confidence": 0-1,
        "recommended_tags": ["..."]
    }}
    """

    handler = agent.run(query)
    result = await handler

    return {"agent_output": str(result)}