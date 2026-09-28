# LLM workforce impact: RAG chatbot

**Live demo:** https://huggingface.co/spaces/udayahfs/databricks-rag-chatbot

A retrieval-augmented generation (RAG) chatbot that answers questions about how generative AI and large language models affect the workforce, wages, and the economy, grounded in a set of research PDFs.

## Architecture
1. The question is embedded with OpenAI `text-embedding-3-small`.
2. The embedding is used to search a Databricks Vector Search index.
3. The retrieved passages go into a prompt for Llama 3.3 70B (Meta's open-weight model, hosted on Databricks).
4. The model answers from those passages.

Deployed as a Gradio app on Hugging Face Spaces. Credentials are Space secrets and are not in this repo.

## What I ran into
- Vector search queries were blocked from Databricks Free Edition notebooks, but worked once the app ran on Hugging Face and called the REST API.
