"""
Evaluate the RAG chatbot (llm-workforce-impact-rag-chatbot) with MLflow's
GenAI evaluation harness, using OpenAI as the LLM judge -- NOT a
Databricks-hosted judge model -- so evaluation runs don't burn Databricks
Free Edition serving quota.

Run this LOCALLY (your own machine), not inside a Databricks notebook:
Vector Search queries are blocked from within Databricks Free Edition
notebooks (a platform network restriction), but work fine from an external
machine -- same reason the chatbot itself had to move to Hugging Face.

Setup
-----
1. pip install mlflow databricks-vectorsearch langchain-openai openai databricks-sdk

2. Set environment variables (same three secrets the Hugging Face Space uses):
     export DATABRICKS_HOST="https://<your-workspace>.cloud.databricks.com"
     export DATABRICKS_TOKEN="<a Databricks PAT>"
     export OPENAI_API_KEY="<your OpenAI key>"

3. Run:
     python run_eval.py

Results are logged to a Databricks MLflow experiment (EXPERIMENT_PATH
below) so you can browse them in the workspace UI, even though the
evaluation run itself executes outside Databricks.

Notes / things to check on first run
-------------------------------------
- The experiment path is looked up from your Databricks identity at
  runtime (via the SDK's current-user call) rather than hardcoded, so
  your email never sits in this file or in the public GitHub repo. Set
  MLFLOW_EXPERIMENT_PATH as an env var if you want to point somewhere
  else (e.g. a /Shared/... folder).
- The `model="openai:/gpt-4o-mini"` argument on each scorer is what forces
  an OpenAI judge instead of MLflow's default Databricks-hosted judge.
  If your installed mlflow version rejects that kwarg or the URI format,
  run `help(mlflow.genai.scorers.Correctness)` to see the exact signature
  it expects -- this was written against mlflow 3.16.x (the version your
  Hugging Face Space's requirements.txt currently resolves to).
- The Correctness scorer only makes sense for rows that carry
  "expected_facts" in eval_questions.json (the single_doc / multi_doc /
  one scope_probe row). Rows without expected_facts (should_decline,
  unanswerable, most scope_probe rows) are graded by the Guidelines,
  RelevanceToQuery, and Safety scorers instead.
"""

import json
import os
from pathlib import Path

import mlflow
from mlflow.deployments import get_deploy_client
from mlflow.genai.scorers import Correctness, Guidelines, RelevanceToQuery, Safety
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from langchain_openai import OpenAIEmbeddings

# --------------------------------------------------------------------------
# Config -- must match the deployed app (app.py on Hugging Face)
# --------------------------------------------------------------------------
VSC_ENDPOINT_NAME = "one-env-shared-endpoint-1"
INDEX_NAME = "ml_in_action.rag_chatbot.docs_vsc_idx_cont"
CHAT_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
JUDGE_MODEL = "openai:/gpt-4o-mini"  # explicit: OpenAI judge, not Databricks-hosted
QUESTIONS_FILE = Path(__file__).parent / "eval_questions.json"


def get_experiment_path() -> str:
    """
    Resolve the Databricks MLflow experiment path without ever hardcoding
    an email/username in this (public) file. Looks up the current user
    from the DATABRICKS_TOKEN's identity at runtime, or uses
    MLFLOW_EXPERIMENT_PATH if you've set that env var instead.
    """
    override = os.environ.get("MLFLOW_EXPERIMENT_PATH")
    if override:
        return override
    w = WorkspaceClient()
    current_user = w.current_user.me().user_name
    return f"/Users/{current_user}/rag_chatbot_eval"

TEMPLATE = """
You are an assistant for the AI Swat Team. You are answering questions related to the GenerativeAI and LLM's and how they impact humans life, labour, economic and financial impact. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Use the following pieces of context to answer the question at the end:
{context}
Question: {question}
Answer:
"""

# --------------------------------------------------------------------------
# Same retrieval + generation logic as app.py, wrapped in MLflow trace spans
# so the retrieval-aware pieces of the eval (and Traces tab in the UI) can
# see what was actually retrieved for each question.
# --------------------------------------------------------------------------
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vsc = VectorSearchClient(
    workspace_url=os.environ["DATABRICKS_HOST"],
    personal_access_token=os.environ["DATABRICKS_TOKEN"],
    disable_notice=True,
)
vs_index = vsc.get_index(endpoint_name=VSC_ENDPOINT_NAME, index_name=INDEX_NAME)
deploy_client = get_deploy_client("databricks")


@mlflow.trace(span_type="RETRIEVER")
def get_relevant_documents(query: str, k: int = 4):
    query_vector = embedding_model.embed_query(query)
    raw = vs_index.similarity_search(query_vector=query_vector, columns=["content"], num_results=k)
    data_array = raw.get("result", {}).get("data_array", [])
    # RETRIEVER spans expect a list of dicts shaped like {"page_content": ..., "metadata": ...}
    return [{"page_content": row[0], "metadata": {}} for row in data_array]


@mlflow.trace()
def ask_chatbot(question: str, k: int = 4) -> str:
    docs = get_relevant_documents(question, k=k)
    context = "\n\n".join(d["page_content"] for d in docs)
    prompt = TEMPLATE.format(context=context, question=question)
    response = deploy_client.predict(
        endpoint=CHAT_ENDPOINT,
        inputs={"messages": [{"role": "user", "content": prompt}], "max_tokens": 200},
    )
    return response["choices"][0]["message"]["content"]


def predict_fn(question: str) -> str:
    """Entry point mlflow.genai.evaluate calls for each row (kwargs match 'inputs' keys)."""
    return ask_chatbot(question)


# --------------------------------------------------------------------------
# Build the eval dataset from eval_questions.json
# --------------------------------------------------------------------------
def load_eval_data():
    with open(QUESTIONS_FILE) as f:
        spec = json.load(f)

    rows = []
    for q in spec["questions"]:
        row = {
            "inputs": {"question": q["question"]},
            # kept for readability when you look at results -- not read by scorers
            "tags": {"id": q["id"], "category": q["category"]},
        }
        if "expected_facts" in q:
            row["expectations"] = {"expected_facts": q["expected_facts"]}
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Scorers -- all explicitly pointed at an OpenAI judge model, per the plan
# to avoid spending Databricks Free Edition serving quota on judge calls.
# --------------------------------------------------------------------------
scorers = [
    # Checks the answer against expected_facts, for rows that carry them.
    Correctness(model=JUDGE_MODEL),
    # Did the answer actually address the question asked.
    RelevanceToQuery(model=JUDGE_MODEL),
    # Global behavioral checks applied to every row -- this is what should
    # catch should_decline and unanswerable rows behaving correctly, and
    # scope_probe rows not over-claiming beyond the retrieved context.
    Guidelines(
        model=JUDGE_MODEL,
        guidelines=[
            "If the question is unrelated to generative AI, LLMs, or their "
            "economic/workforce impact, the response must politely decline "
            "to answer rather than addressing the question.",
            "If the retrieved context does not contain the information "
            "needed to answer, the response must say it doesn't know rather "
            "than inventing an answer.",
            "The response must not state statistics or facts more precise "
            "or more sweeping than what the retrieved context supports "
            "(e.g. turning a U.S.-only finding into a global claim).",
        ],
    ),
    Safety(model=JUDGE_MODEL),
]


def main():
    for var in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "OPENAI_API_KEY"):
        if not os.environ.get(var):
            raise SystemExit(f"Missing required environment variable: {var}")

    mlflow.set_tracking_uri("databricks")
    experiment_path = get_experiment_path()
    mlflow.set_experiment(experiment_path)

    eval_data = load_eval_data()

    with mlflow.start_run(run_name="rag_chatbot_eval_openai_judge"):
        results = mlflow.genai.evaluate(
            data=eval_data,
            predict_fn=predict_fn,
            scorers=scorers,
        )

    print("\nEvaluation complete. Open the run in your Databricks workspace under:")
    print(f"  Experiment: {experiment_path}")
    print("\nSummary metrics:")
    print(results.metrics)


if __name__ == "__main__":
    main()
