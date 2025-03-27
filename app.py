import os
import boto3
import requests
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms.base import LLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

##########################################
# Global System Prompt
##########################################
SYSTEM_PROMPT = (
    "You are an intelligent and helpful answer agent. "
    "When you are asked a question, you must use all available resources, including RAG from the document retrieval system and internet search, "
    "to provide the best possible answer. "
    "Do not guess; if you do not know the answer, simply answer 'I do not know'."
)

##########################################
# Initialize Flask App and Configurations
##########################################
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# S3 bucket configuration (EC2 instance with proper IAM role)
s3_bucket_name = "your-s3-bucket-name"
s3_prefix = "uploaded_documents"  # S3 folder for uploaded docs
aws_region = "us-east-1"
s3 = boto3.client("s3", region_name=aws_region)

# Set your OpenAI API key (ensure it's set in your environment)
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

##########################################
# Initialize FAISS Vector Store using OpenAI Embeddings
##########################################
embedding = OpenAIEmbeddings()  # Uses your OPENAI_API_KEY
faiss_index_path = "faiss_index"

if os.path.exists(faiss_index_path):
    vector_store = FAISS.load_local(faiss_index_path, embedding)
else:
    # Create an empty FAISS index with no initial texts.
    vector_store = FAISS.from_texts([], embedding, metadatas=[])

##########################################
# Custom LLM for VLLM Inference
##########################################
class VLLMLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "vllm_custom"
    
    def _call(self, prompt: str, stop=None) -> str:
        payload = {"prompt": prompt}
        if stop:
            payload["stop"] = stop
        response = requests.post("http://192.168.10.10:8000", json=payload)
        response.raise_for_status()
        return response.json().get("text", "")

vllm_llm = VLLMLLM()

##########################################
# Set Up Conversational Retrieval Chain with Memory and Custom Prompt
##########################################
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Create a custom prompt that includes the system prompt.
custom_prompt = (
    f"{SYSTEM_PROMPT}\n\n"
    "Context:\n{{context}}\n\n"
    "Question: {{question}}\n\n"
    "Answer:"
)

conversational_chain = ConversationalRetrievalChain.from_llm(
    llm=vllm_llm,
    retriever=retriever,
    memory=memory,
    chain_type="stuff",  # You can choose a different chain type if needed.
    combine_docs_chain_kwargs={"prompt": custom_prompt}
)

##########################################
# RAG Re-Ranking Function
##########################################
def rerank_documents(query: str, documents):
    """
    Re-rank retrieved documents by using the LLM to score each document's relevance.
    Returns the list of documents sorted in descending order of relevance.
    """
    scored_docs = []
    for doc in documents:
        # Prompt the LLM to score the document on a scale from 1 to 10.
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Rate the relevance of the following document to the query on a scale from 1 to 10.\n\n"
            f"Query: {query}\n\nDocument: {doc.page_content}\n\n"
            "Answer with a number only."
        )
        score_str = vllm_llm(prompt)
        try:
            score = float(score_str.strip())
        except Exception:
            score = 0.0
        scored_docs.append((doc, score))
    # Sort documents by score (highest first)
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    ranked_docs = [doc for doc, score in scored_docs]
    return ranked_docs

##########################################
# Internet Search Functionality using Bing Web Search API
##########################################
BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"
BING_API_KEY = os.environ.get("BING_SEARCH_API_KEY")  # Set this in your environment

def internet_search(query: str):
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {"q": query, "textDecorations": True, "textFormat": "HTML"}
    response = requests.get(BING_SEARCH_URL, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    results = []
    for web_page in data.get("webPages", {}).get("value", [])[:3]:  # Return top 3 results
        results.append({
            "name": web_page.get("name"),
            "snippet": web_page.get("snippet"),
            "url": web_page.get("url")
        })
    return results

##########################################
# Routes
##########################################

# Homepage: Serves the frontend
@app.route('/')
def index():
    return render_template('index.html')

# Query endpoint for chat using the Conversational Retrieval Chain
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    result = conversational_chain({"question": question})
    return jsonify(result)

# New endpoint: Query with RAG Re-Ranking
@app.route('/ask_rerank', methods=['POST'])
def ask_rerank():
    data = request.get_json()
    question = data.get("question", "")
    # Retrieve candidate documents using the retriever.
    docs = retriever.get_relevant_documents(question)
    # Re-rank the retrieved documents.
    ranked_docs = rerank_documents(question, docs)
    # Combine the content of the top-ranked documents into context.
    context = "\n\n".join([doc.page_content for doc in ranked_docs])
    # Create a prompt that uses the system prompt along with the re-ranked context.
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Based on the following context:\n\n{context}\n\n"
        f"Answer the question: {question}"
    )
    answer = vllm_llm(prompt)
    return jsonify({"answer": answer})

# Upload endpoint for documents
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Upload original file to S3.
    s3_key = os.path.join(s3_prefix, filename)
    s3.upload_file(file_path, s3_bucket_name, s3_key)
    
    # Process file: extract text from PDFs or read text files.
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    
    # Chunk the text into manageable pieces.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    metadatas = [{"source": filename, "chunk": i} for i in range(len(chunks))]
    
    # Update FAISS index with the new text chunks.
    vector_store.add_texts(chunks, metadatas=metadatas)
    vector_store.save_local(faiss_index_path)
    
    # Clean up temporary file.
    os.remove(file_path)
    
    return jsonify({"message": "File uploaded and processed successfully", "chunks_added": len(chunks)})

# Internet search endpoint
@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get("query", "")
    try:
        results = internet_search(query)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app (accessible externally on EC2)
    app.run(host="0.0.0.0", port=5000, debug=True)
