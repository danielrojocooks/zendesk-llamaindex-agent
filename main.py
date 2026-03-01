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

APP_BUILD = "fc-tools-v2"

# -------------------------
# ENV
# -------------------------
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

# Minimal governance
TOP_K = int(os.environ.get("TOP_K", "3"))
SIMILARITY_CUTOFF = float(os.environ.get("SIMILARITY_CUTOFF", "0.78"))

# Zendesk (needed for escalate side-effect)
ZENDESK_SUBDOMAIN = os.environ.get("ZENDESK_SUBDOMAIN")          # e.g. "acme"
ZENDESK_EMAIL = os.environ.get("ZENDESK_EMAIL")                  # e.g. "bot@acme.com"
ZENDESK_API_TOKEN = os.environ.get("ZENDESK_API_TOKEN")          # token
ZENDESK_SECRET = os.environ.get("ZENDESK_WEBHOOK_SHARED_SECRET", "")  # optional header verification

client = OpenAI(api_key=OPENAI_API_KEY)

index: Optional[VectorStoreIndex] = None
retriever = None

# -------------------------
# OpenAI Tool definitions
# -------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "reply_to_customer",
            "description": "Send a customer-facing email reply body. Must be complete and ready to send.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_body": {"type": "string", "description": "Full email reply body, plain text."}
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
            "description": "Escalate when KB is missing or insufficient. Provide a short reason.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why escalation is required."}
                },
                "required": ["reason"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_tags",
            "description": "Apply tags for tracking. Use lowercase snake_case.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tags": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["tags"],
                "additionalProperties": False
            }
        }
    }
]

SYSTEM_PROMPT = (
    "You are a Zendesk triage system.\n"
    "Hard rule: Respond ONLY by calling exactly ONE tool. No prose.\n"
    "Policy:\n"
    "- If KB context answers the question, call reply_to_customer(email_body).\n"
    "- If KB context is missing or insufficient, call escalate_ticket(reason).\n"
    "- Call apply_tags(tags) only when tags are materially useful.\n"
    "The email must be professional, concise, and based ONLY on provided KB context.\n"
)

# -------------------------
# Startup: load docs -> index -> retriever
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
    return {"ok": True, "build": APP_BUILD, "top_k": TOP_K, "cutoff": SIMILARITY_CUTOFF}

# -------------------------
# Zendesk helpers
# -------------------------
def zendesk_add_public_reply(ticket_id: int, body: str) -> None:
    if not zendesk_ready():
        raise HTTPException(status_code=500, detail="Zendesk env vars missing (subdomain/email/api_token).")
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
        raise HTTPException(status_code=502, detail=f"Zendesk reply failed: {r.status_code} {r.text}")

def zendesk_ready() -> bool:
    return bool(ZENDESK_SUBDOMAIN and ZENDESK_EMAIL and ZENDESK_API_TOKEN)

def zendesk_auth():
    # Zendesk token auth: email/token as username, token as password
    return (f"{ZENDESK_EMAIL}/token", ZENDESK_API_TOKEN)

def zendesk_api_url(path: str) -> str:
    return f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/api/v2{path}"

def zendesk_add_internal_note_and_tags(ticket_id: int, note: str, tags: List[str]) -> None:
    if not zendesk_ready():
        raise HTTPException(status_code=500, detail="Zendesk env vars missing (subdomain/email/api_token).")
    url = zendesk_api_url(f"/tickets/{ticket_id}.json")
    payload = {
        "ticket": {
            "comment": {"public": False, "body": note},
            "additional_tags": tags
        }
    }
    r = requests.put(url, auth=zendesk_auth(), json=payload, timeout=20)
    if r.status_code >= 300:
        raise HTTPException(status_code=502, detail=f"Zendesk update failed: {r.status_code} {r.text}")

def zendesk_add_tags(ticket_id: int, tags: List[str]) -> None:
    if not zendesk_ready():
        raise HTTPException(status_code=500, detail="Zendesk env vars missing (subdomain/email/api_token).")
    url = zendesk_api_url(f"/tickets/{ticket_id}.json")
    payload = {"ticket": {"additional_tags": tags}}
    r = requests.put(url, auth=zendesk_auth(), json=payload, timeout=20)
    if r.status_code >= 300:
        raise HTTPException(status_code=502, detail=f"Zendesk tag update failed: {r.status_code} {r.text}")

# -------------------------
# Retrieval + gating
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
    blocks = []
    for i, n in enumerate(nodes, start=1):
        meta = n.node.metadata or {}
        title = meta.get("title") or meta.get("filename") or meta.get("source") or "KB"
        url = meta.get("url") or meta.get("link") or ""
        text = (n.node.get_content() or "").strip()
        text = text[:1800]
        score = n.score if n.score is not None else 0.0
        blocks.append(
            f"[KB {i}] score={score:.3f}\n"
            f"title={title}\n"
            f"url={url}\n"
            f"excerpt:\n{text}\n"
        )
    return "\n---\n".join(blocks).strip()

# -------------------------
# Payload model
# -------------------------
class ZendeskTicket(BaseModel):
    # Add ticket_id if your webhook can send it (strongly recommended for escalation side-effect).
    ticket_id: Optional[int] = None

    subject: str
    description: str
    requester_email: Optional[str] = None

# -------------------------
# Main endpoint
# -------------------------
@app.post("/fc_test")
async def fc_test():
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        tools=TOOLS,
        tool_choice="required",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Call a tool now."},
        ],
    )

    msg = resp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None) or []

    return {
        "build": APP_BUILD,
        "tool_calls_len": len(tool_calls),
        "content": msg.content,
        "tool_name": tool_calls[0].function.name if tool_calls else None,
        "tool_args": json.loads(tool_calls[0].function.arguments) if tool_calls else None,
    }

@app.post("/zendesk")
async def zendesk(ticket: ZendeskTicket, req: Request):
    # Optional secret verification if you configured it in Zendesk webhook headers
    if ZENDESK_SECRET:
        got = req.headers.get("x-zendesk-secret", "")
        if got != ZENDESK_SECRET:
            raise HTTPException(status_code=401, detail="Invalid secret")

    query_text = f"Subject: {ticket.subject}\nDescription: {ticket.description}".strip()

    # 1) Deterministic gate: no relevant KB hit => escalate without LLM
    nodes = retrieve_kb(query_text)
    if not is_relevant_hit(nodes):
        reason = "No relevant KB hit above similarity cutoff."
        if ticket.ticket_id is not None:
            zendesk_add_internal_note_and_tags(
                ticket_id=ticket.ticket_id,
                note=f"Auto-escalated: {reason}",
                tags=["auto_escalated", "kb_miss"]
            )
        # Return tool-shaped payload for your downstream router if desired
        return {
            "action": "escalate_ticket",
            "reason": reason
        }
    
    

    kb_context = format_kb_context(nodes)

    # 2) Function call step: force exactly one tool call, no prose
    user_message = (
        "Ticket:\n"
        f"{query_text}\n\n"
        "KB context (use ONLY this):\n"
        f"{kb_context}\n\n"
        "Choose exactly one tool call per policy."
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
    tool_calls = getattr(msg, "tool_calls", None) or []
    if len(tool_calls) != 1:
        raise HTTPException(status_code=502, detail=f"Model returned {len(tool_calls)} tool calls; expected 1.")

    call = tool_calls[0]
    fn = call.function.name
    args = json.loads(call.function.arguments or "{}")

    # 3) Execute tool effects (minimal)

   if fn == "reply_to_customer":
    email_body = (args.get("email_body") or "").strip()
    if not email_body:
        raise HTTPException(status_code=502, detail="Empty email_body from model")

    if ticket.ticket_id:
        zendesk_add_public_reply(ticket.ticket_id, email_body)
        zendesk_add_tags(ticket.ticket_id, ["ai_replied"])

    return {"status": "replied"}

elif fn == "escalate_ticket":
    reason = (args.get("reason") or "").strip() or "Escalated: KB insufficient."

    if ticket.ticket_id:
        zendesk_add_internal_note_and_tags(
            ticket.ticket_id,
            f"Auto-escalated: {reason}",
            ["auto_escalated", "ai_replied"]
        )

    return {"status": "escalated"}

    if ticket.ticket_id is not None:
        zendesk_add_public_reply(ticket.ticket_id, email_body)s
        zendesk_add_tags(ticket.ticket_id, ["ai_replied"])

    return {"status": "replied"}s

    if ticket.ticket_id is not None:
        zendesk_add_public_reply(ticket.ticket_id, email_body)
        zendesk_add_tags(ticket.ticket_id, ["ai_replied"])

    return {"status": "replied"}

    if fn == "escalate_ticket":
        reason = (args.get("reason") or "").strip() or "Escalated: KB insufficient."
        if ticket.ticket_id is not None:
            zendesk_add_internal_note_and_tags(
                ticket_id=ticket.ticket_id,
                note=f"Auto-escalated: {reason}",
                tags=["auto_escalated", "kb_insufficient, ai_replied"]
            )
        return {"action": "escalate_ticket", "reason": reason}

    if fn == "apply_tags":
        tags = args.get("tags") or []
        # If we have a ticket_id, apply tags in Zendesk; otherwise just return them.
        if ticket.ticket_id is not None and tags:
            zendesk_add_tags(ticket.ticket_id, tags)
        return {"action": "apply_tags", "tags": tags}

    raise HTTPException(status_code=502, detail=f"Unknown tool: {fn}")