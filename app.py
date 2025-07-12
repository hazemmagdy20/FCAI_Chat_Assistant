from flask import Flask, render_template, request, jsonify
import os
import json
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

app = Flask(__name__)

# ──────────────────────────────
# Reuse your functions
# ──────────────────────────────

# Extract Info from files.
def extract_text_from_json(data, prefix=""):
    text_chunks = []
    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix} -> {key}" if prefix else key
            text_chunks.extend(extract_text_from_json(value, new_prefix))
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            new_prefix = f"{prefix} [{idx}]"
            text_chunks.extend(extract_text_from_json(item, new_prefix))
    else:
        text_chunks.append(f"{prefix}: {data}")
    return text_chunks

# create vector store which will store emdedding of ou data.
def create_vector_store(folder_path, vectorstore_path="bylaws_vector_index"):
    index_file = os.path.join(vectorstore_path, "index.faiss")

    if os.path.exists(index_file):
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
        return FAISS.load_local(
            vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    all_texts = []
    for fn in os.listdir(folder_path):
        if fn.lower().endswith(".json"):
            with open(os.path.join(folder_path, fn), encoding='utf-8') as f:
                data = json.load(f)
            all_texts.extend(extract_text_from_json(data))

    text = "\n".join(all_texts)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=150)
    docs = splitter.create_documents([text])

    for d in docs:
        d.metadata = {"source": "bylaw"}

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
    store = FAISS.from_documents(docs, embeddings)
    store.save_local(vectorstore_path)
    return store

# Calling LLM from OpenRouter.
def call_llm(prompt):
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer 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",
                "Content-Type": "application/json",
                "X-Title": "BylawAssistant",
            },
            data=json.dumps({
                "model": "deepseek/deepseek-r1-0528-qwen3-8b:free",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant trained on university academic bylaws."},
                    {"role": "user", "content": prompt}
                ]
            })
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"⚠️ LLM Error: {str(e)}"

# Get query and response for it.
def process_query(user_query):
    folder_path = "FCAI_Bylaw_2018"  
    vector_store = create_vector_store(folder_path)
    retriever = vector_store.as_retriever(search_kwargs={"k": 20})
    docs = retriever.get_relevant_documents(user_query)

    if not docs:
        return "⚠️ No relevant documents found."

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = (
        "You are a helpful assistant answering questions based on academic bylaws.\n\n"
        "Context:\n"
        "---------------------\n"
        f"{context}\n"
        "---------------------\n\n"
        f"User question:\n{user_query}\n\n"
        "Answer in Arabic if the question is in Arabic, or in English otherwise.\n\n"
        "Helpful answer:"
    )
    return call_llm(prompt)


# ──────────────────────────────
# Routes
# ──────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    user_query = request.json.get("message", "")
    answer = process_query(user_query)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)
