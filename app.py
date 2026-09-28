import os
import gradio as gr
from mlflow.deployments import get_deploy_client
from databricks.vector_search.client import VectorSearchClient
from langchain_openai import OpenAIEmbeddings

# --- Authentication ---
# dbutils/spark only exist inside a Databricks notebook session. On HF,
# these three values come from plain environment variables — set as
# Space secrets in Settings > Repository secrets, injected automatically
# at startup. No code needed to "fetch" them beyond reading os.environ.
DATABRICKS_HOST = os.environ["DATABRICKS_HOST"]
DATABRICKS_TOKEN = os.environ["DATABRICKS_TOKEN"]
# get_deploy_client("databricks") below also reads DATABRICKS_HOST/TOKEN
# from the environment automatically — no extra config needed for it.

# --- Retriever (same logic as the Databricks notebook) ---
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")  # reads OPENAI_API_KEY from env automatically
VSC_ENDPOINT_NAME = "one-env-shared-endpoint-1"
INDEX_NAME = "ml_in_action.rag_chatbot.docs_vsc_idx_cont"

vsc = VectorSearchClient(workspace_url=DATABRICKS_HOST, personal_access_token=DATABRICKS_TOKEN, disable_notice=True)
vs_index = vsc.get_index(endpoint_name=VSC_ENDPOINT_NAME, index_name=INDEX_NAME)

def get_relevant_documents(query, k=4):
    query_vector = embedding_model.embed_query(query)
    raw = vs_index.similarity_search(query_vector=query_vector, columns=["content"], num_results=k)
    # The actual matches live nested here, as rows of column values, not
    # ready-made dicts. Since we only requested the "content" column,
    # each row is a one-element list.
    data_array = raw.get("result", {}).get("data_array", [])
    return [{"content": row[0]} for row in data_array]
    
# def get_relevant_documents(query, k=4):
#     query_vector = embedding_model.embed_query(query)
#     return vs_index.similarity_search(query_vector=query_vector, columns=["content"], num_results=k)

# --- Chat model ---
deploy_client = get_deploy_client("databricks")
CHAT_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

TEMPLATE = """
You are an assistant for the AI Swat Team. You are answering questions related to the GenerativeAI and LLM's and how they impact humans life, labour, economic and financial impact. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Use the following pieces of context to answer the question at the end:
{context}
Question: {question}
Answer:
"""
def ask_chatbot(question: str, k: int = 4) -> str:
    try:
        docs = get_relevant_documents(question, k=k)
        context = "\n\n".join(d["content"] for d in docs)
        prompt = TEMPLATE.format(context=context, question=question)
        response = deploy_client.predict(
            endpoint=CHAT_ENDPOINT,
            inputs={"messages": [{"role": "user", "content": prompt}], "max_tokens": 200}
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Something went wrong: {e}"
# def ask_chatbot(question: str, k: int = 4) -> str:
#     # Minimal safety net for a PUBLIC demo: a raw Python traceback shown
#     # to a random visitor looks broken. This is separate from the earlier
#     # "don't fake an answer without context" concern — it still refuses
#     # to call the LLM on a retrieval failure, just with a clean message
#     # instead of crashing the whole Gradio callback.
#     try:
#         docs = get_relevant_documents(question, k=k)
#     except Exception as e:
#         return f"Sorry, I couldn't search the knowledge base right now ({e}). Please try again shortly."

#     context = "\n\n".join(d["content"] for d in docs)
#     prompt = TEMPLATE.format(context=context, question=question)

#     try:
#         response = deploy_client.predict(
#             endpoint=CHAT_ENDPOINT,
#             inputs={"messages": [{"role": "user", "content": prompt}], "max_tokens": 200}
#         )
#         return response["choices"][0]["message"]["content"]
#     except Exception as e:
#         return f"Sorry, something went wrong generating an answer ({e})."

# --- Gradio UI ---
def respond(message, history):
    return ask_chatbot(message)

with gr.Blocks() as demo:
    gr.Markdown("""
    # RAG chatbot — GenAI & LLM economic impact
    Ask questions about how GenAI and LLMs affect the workforce, wages,
    and the economy. Answers are grounded in a curated set of source
    documents via retrieval-augmented generation.

    **Try asking:** "Will AI impact work forces in the US?" or
    "Can LLMs impact wages, and how?"
    """)
    gr.ChatInterface(fn=respond)

if __name__ == "__main__":
    demo.launch()