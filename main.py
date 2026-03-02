import os
import json
import requests
from typing import Optional, List

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from openai import OpenAI

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.schema import NodeWithScore

load_dotenv()

app = FastAPI()

APP_BUILD = "fc-tools-v4-stable"

# -------------------------
# ENV
# -------------------------
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

TOP_K = int(os.environ.get("TOP_K", "3"))
SIMILARITY_CUTOFF = float(os.environ.get("SIMILARITY_CUTOFF", "0.78"))

ZENDESK_SUBDOMAIN = os.environ.get("ZENDESK_SUBDOMAIN")
ZENDESK_EMAIL = os.environ.get("ZENDESK_EMAIL")
ZENDESK_API_TOKEN = os.environ.get("ZENDESK_API_TOKEN")

client = OpenAI(api_key=OPENAI_API_KEY)

index: Optional[VectorStoreIndex] = None
retriever = None

# -------------------------
# Tool Definitions
# -------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "reply_to_customer",
            "description": "Send a complete customer-facing email reply.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_body": {"type": "string"}
                },
                "required": ["email_body"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_ticket",
            "description": "Escalate the ticket with a reason.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"}
                },
                "required": ["reason"],
                "additionalProperties": False
            }
        }
    }
]

SYSTEM_PROMPT = (
    "You are a Zendesk triage system.\n"
    "Respond ONLY by calling exactly ONE tool.\n"
    "- If KB answers the question → reply_to_customer.\n"
    "- If KB insufficient → escalate_ticket.\n"
)

# -------------------------
# Startup
# -------------------------
@app.on_event("startup")
def startup_event():
    global index, retriever
    documents = SimpleDirectoryReader("docs").load_data()
    index = VectorStoreIndex.from_documents(documents)
    retriever = index.as_retriever(similarity_top_k=TOP_K)

# -------------------------
# Health
# -------------------------
@app.get("/health")
def health():
    return {"ok": True, "build": APP_BUILD}

# -------------------------
# Zendesk Helpers
# -------------------------
def zendesk_ready() -> bool:
    return bool(ZENDESK_SUBDOMAIN and ZENDESK_EMAIL and ZENDESK_API_TOKEN)

def zendesk_auth():
    return (f"{ZENDESK_EMAIL}/token", ZENDESK_API_TOKEN)

def zendesk_api_url(path: str) -> str:
    return f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/api/v2{path}"

def zendesk_add_public_reply(ticket_id: int, body: str):
    if not zendesk_ready():
        raise HTTPException(status_code=500, detail="Zendesk not configured")

    url = zendesk_api_url(f"/tickets/{ticket_id}.json")
    payload = {
        "ticket": {
            "comment": {
                "public": True,
                "body": body
            }
        }
    }

    r = requests.put(url, auth=zendesk_auth(), json=payload, timeout=20)
    if r.status_code >= 300:
        raise HTTPException(status_code=502, detail=r.text)

def zendesk_add_internal_note(ticket_id: int, note: str):
    if not zendesk_ready():
        raise HTTPException(status_code=500, detail="Zendesk not configured")

    url = zendesk_api_url(f"/tickets/{ticket_id}.json")
    payload = {
        "ticket": {
            "comment": {"public": False, "body": note}
        }
    }

    r = requests.put(url, auth=zendesk_auth(), json=payload, timeout=20)
    if r.status_code >= 300:
        raise HTTPException(status_code=502, detail=r.text)

# -------------------------
# Retrieval
# -------------------------
def retrieve_kb(query: str) -> List[NodeWithScore]:
    if retriever is None:
        return []
    return retriever.retrieve(query)

def is_relevant_hit(nodes: List[NodeWithScore]) -> bool:
    if not nodes:
        return False
    top = nodes[0]
    if top.score is None:
        return True
    return float(top.score) >= SIMILARITY_CUTOFF

def format_kb_context(nodes: List[NodeWithScore]) -> str:
    parts = []
    for n in nodes:
        text = (n.node.get_content() or "").strip()[:1500]
        parts.append(text)
    return "\n\n".join(parts)

# -------------------------
# Ticket Model
# -------------------------
class ZendeskTicket(BaseModel):
    ticket_id: Optional[int] = None
    subject: str
    description: str
    requester_email: Optional[str] = None

# -------------------------
# Webhook Endpoint
# -------------------------
@app.post("/zendesk")
async def zendesk(req: Request):

    raw_body = await req.body()
    body_text = raw_body.decode("utf-8", errors="ignore").strip()

    print("RAW BODY:", body_text)

    # Fix malformed body like "{}{ ... }"
    if body_text.startswith("{}{"):
        body_text = body_text[2:]

    try:
        payload = json.loads(body_text)
    except Exception as e:
        print("JSON ERROR:", str(e))
        return {"status": "invalid_json"}

    # Support both flat and nested payloads
    if "ticket" in payload:
        data = payload["ticket"]
    else:
        data = payload

    ticket = ZendeskTicket(
        ticket_id=int(data.get("ticket_id") or data.get("id") or 0),
        subject=data.get("subject") or "",
        description=data.get("description") or "",
        requester_email=data.get("requester_email"),
    )

    query_text = f"{ticket.subject}\n{ticket.description}".strip()

    nodes = retrieve_kb(query_text)

    if not is_relevant_hit(nodes):
        if ticket.ticket_id:
            zendesk_add_internal_note(
                ticket.ticket_id,
                "Auto-escalated: No relevant KB hit."
            )
        return {"status": "escalated"}

    kb_context = format_kb_context(nodes)

    user_message = (
        f"Ticket:\n{query_text}\n\n"
        f"KB:\n{kb_context}\n\n"
        "Choose exactly one tool."
    )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        tools=TOOLS,
        tool_choice="required",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    msg = resp.choices[0].message
    tool_calls = msg.tool_calls or []

    if len(tool_calls) != 1:
        raise HTTPException(status_code=502, detail="Model must return exactly one tool call.")

    call = tool_calls[0]
    fn = call.function.name
    args = json.loads(call.function.arguments or "{}")

    if fn == "reply_to_customer":
        body = (args.get("email_body") or "").strip()
        if ticket.ticket_id:
            zendesk_add_public_reply(ticket.ticket_id, body)
        return {"status": "replied"}

    if fn == "escalate_ticket":
        reason = (args.get("reason") or "").strip()
        if ticket.ticket_id:
            zendesk_add_internal_note(
                ticket.ticket_id,
                f"Auto-escalated: {reason}"
            )
        return {"status": "escalated"}

    raise HTTPException(status_code=502, detail="Unknown tool")