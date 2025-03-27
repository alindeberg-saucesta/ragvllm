# ragvllm
install dependencies
pip install flask boto3 requests langchain openai langsmith faiss-cpu PyPDF2 langchain-community tiktoken python-multipart uvicorn

List of Dependencies:
Flask: For the backend web server.
Boto3: For interacting with AWS S3.
Requests: For HTTP requests (to VLLM and external APIs).
LangChain, OpenAI, LangSmith: For LLM chain, embeddings, and experiment tracking.
FAISS (faiss-cpu): For vector storage and retrieval.
PyPDF2: For PDF processing and text extraction.

Need OpenAI API Key and Brave Search API Key
Set Enviroment Variables:
OPENAI_API_KEY: Your OpenAI API key.
BRAVE_SEARCH_API_KEY: API key for the Brave Web Search API.
