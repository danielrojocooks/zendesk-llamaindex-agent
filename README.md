zendesk-llamaindex-agent

A LlamaIndex RAG pipeline with a FastAPI integration layer that handles Zendesk support tickets end to end.
When a ticket comes in, the system retrieves relevant documentation chunks, generates a grounded response using OpenAI, and posts the reply directly back to the Zendesk ticket. Out-of-scope questions are automatically escalated with an internal note.
How it works

Zendesk trigger fires on new ticket and sends a webhook to the Railway-hosted endpoint
LlamaIndex retrieves the most relevant documentation chunks based on semantic similarity
OpenAI generates a response grounded in the retrieved content using function calling
If KB is sufficient: public reply posted to the ticket and delivered to the user via email
If KB is insufficient: internal escalation note added and ticket routed for human review

Stack

LlamaIndex for RAG pipeline and document retrieval
FastAPI for the webhook endpoint
OpenAI for embeddings and response generation
Railway for deployment
Zendesk for ticket management and webhook trigger

Built as
A proof of concept extending a native Zendesk AI agent with a custom RAG backend. Demonstrates end-to-end ticket resolution from webhook intake to AI-generated reply delivered via email.
﻿# zendesk-llamaindex-agent


