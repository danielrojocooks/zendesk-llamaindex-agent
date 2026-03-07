zendesk-llamaindex-agent
An AI-assisted support automation agent that handles Zendesk tickets end to end. Not a chatbot. Not a demo. An automation service with an AI decision layer.
What it does
A user emails support. Zendesk creates a ticket and fires a webhook. This system receives the webhook, retrieves semantically relevant documentation, and uses a tool-constrained LLM to either post a grounded reply directly to the ticket or escalate with an internal note. The user receives an email. The whole loop closes in under 60 seconds.
How it works
User email
     ↓
Zendesk ticket
     ↓
Zendesk webhook
     ↓
FastAPI endpoint (Railway)
     ↓
Ticket parsed
     ↓
Query → vector retrieval (LlamaIndex)
     ↓
Relevance gate (similarity cutoff)
     ↓
LLM chooses tool
     ↓
Tool executes Zendesk API call
     ↓
Reply posted → Zendesk emails customer
What makes it production-grade
Deterministic gating: A similarity cutoff prevents the model from answering questions the knowledge base can't support. Out-of-scope tickets escalate automatically rather than hallucinate.
Tool-constrained LLM: The model doesn't freely generate text. It must choose exactly one action: reply to the customer or escalate the ticket. Structured outputs only.
Real system side effects: This isn't a notebook. Replies post to live Zendesk tickets and land in the user's inbox.
Production deployment: Public endpoint running on Railway with full observability via deploy and HTTP logs.
Stack

LlamaIndex for RAG pipeline and document retrieval
FastAPI for the webhook endpoint
OpenAI for embeddings and chat completions
Railway for deployment
Zendesk for ticket management, webhook trigger, and reply delivery

Setup

Clone the repo
Add your docs to the /docs folder
Set environment variables in Railway or a local .env file:

OPENAI_API_KEY
ZENDESK_SUBDOMAIN
ZENDESK_EMAIL
ZENDESK_API_TOKEN
OPENAI_MODEL (default: gpt-4.1-mini)
TOP_K (default: 3)
SIMILARITY_CUTOFF (default: 0.78)


Deploy to Railway or run locally with uvicorn main:app

Built as
A proof of concept for what a production RAG backend looks like behind a real support operation. Built in approximately 6 hours.
