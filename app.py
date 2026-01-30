import os
import json
import requests
import numpy as np
import streamlit as st
from openai import OpenAI
import re

st.set_page_config(page_title="Azure QCA Classifier", layout="centered")
st.title("QCA Classifier (Azure Hosted)")
# ----------------------------
# Configuration & Secrets
# ----------------------------
# These should be set in Azure App Service "Environment Variables"
#try:
SCORING_URI = os.environ.get("AZURE_ML_SCORING_URI")
AZURE_ML_KEY = os.environ.get("AZURE_ML_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
#except:
#SCORING_URI = st.secrets.get("AZURE_ML_SCORING_URI")
#AZURE_ML_KEY = st.secrets.get("AZURE_ML_API_KEY")
#HF_TOKEN = st.secrets.get("HF_TOKEN")

LABELS = ["contradiction", "factual", "irrelevant"]
DEFAULT_JUDGE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# ----------------------------
# Utilities
# ----------------------------
def build_text(answer: str, context: str, question: str) -> str:
    return (
        f"Answer: {(answer or '').strip()}\n"
        f"Question: {(question or '').strip()}\n"
        f"Context: {(context or '').strip()}"
    )

# ----------------------------
# Prediction via Azure ML REST API
# ----------------------------
def predict_via_azure(text: str):
    """
    Sends a request to the Azure ML Online Endpoint.
    This replaces the heavy local 'transformers' loading.
    """
    if not SCORING_URI or not AZURE_ML_KEY:
        raise ValueError("Azure ML Scoring URI or API Key is missing in Environment Variables.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AZURE_ML_KEY}"
    }
    
    # Payload format depends on your model's expected input (usually 'input_data' or 'inputs')

    payload = {
        "inputs": [text_input],
        "max_len": 512
    }


    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AZURE_ML_KEY}",
    }

    resp = requests.post(SCORING_URI, headers=headers, json=payload, timeout=60)

    
    if resp.status_code != 200:
        raise Exception(f"Request failed with status {resp.status_code}: {resp.text}")
    
    return resp.text  

# ----------------------------
# LLM Judge (Hugging Face API)
# ----------------------------
def parse_llm_json(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match: return "unknown", text.strip()
    try:
        data = json.loads(match.group())
        label = data.get("label", "").strip().lower()
        reason = data.get("reason", "").strip()
        return (label if label in ["factual", "contradictory", "irrelevant"] else "unknown"), reason
    except:
        return "unknown", text.strip()

def hf_llm_judge(question: str, context: str, answer: str, judge_model: str):
    token = st.secrets.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN missing in Streamlit secrets")

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token
    )

    prompt = f"""
You are a classification API.

IMPORTANT RULES:
- Do NOT include reasoning, thoughts, analysis, or tags like <think>.
- Do NOT include any text before or after the JSON.
- Output MUST be valid JSON ONLY.

Return JSON with exactly:
- label: one of ["factual","contradictory","irrelevant"]
- reason: 2–4 concise sentences

Definitions:
- factual: answer is supported by the context
- contradictory: answer conflicts with the context
- irrelevant: context is insufficient or unrelated

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}
""".strip()

    completion = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    text = completion.choices[0].message.content.strip()
    return parse_llm_json(text)


# ----------------------------
# Streamlit UI
# ----------------------------
# Hugging Face LLM Model
judge_model = st.selectbox(
    "Judge model",
    [
        "HuggingFaceTB/SmolLM3-3B:hf-inference",
    ],
    index=0
)

question = st.text_area("Question", height=100)
context = st.text_area("Context", height=100)
answer = st.text_area("Answer", height=100)

if st.button("Predict (Classifier)", type="primary"):
    try:
        with st.spinner("Calling Azure ML Endpoint..."):
            text_input = build_text(question, context, answer)
            result = json.loads(predict_via_azure(text_input))
            label = result["pred_label"]
            probs = result["probs"]
            
            st.success(f"Result: {label}")
            if probs:
                # Display probabilities in columns
                col1, col2, col3 = st.columns(3)
                col1.metric("Contradiction", f"{probs[0]:.4f}")
                col2.metric("Factual", f"{probs[1]:.4f}")
                col3.metric("Irrelevant", f"{probs[2]:.4f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")



if st.button("Judge with LLM"):
    try:
        label_llm, reason_llm = hf_llm_judge(question, context, answer, judge_model)

        st.info(f"**LLM label:** {label_llm}")
        st.write(f"**Why:** {reason_llm}")

    except Exception as e:

        st.error(f"LLM judge failed:\n\n{e}")

