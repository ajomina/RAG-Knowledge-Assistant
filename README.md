📚 RAG Knowledge Assistant

A Retrieval-Augmented Generation (RAG) based document intelligence system that enables users to upload PDF documents, perform semantic search, retrieve relevant information, generate context-aware answers using Large Language Models, and evaluate response quality.

🚀 Features
📄 Ingestion Layer
PDF Upload
PDF Text Extraction
Recursive Text Chunking
Embedding Generation using Sentence Transformers
🔍 Retrieval Layer
Semantic Search using FAISS
Top-K Retrieval
Cross-Encoder Re-ranking
Context Construction
🤖 Generation Layer
Ollama Llama 3.2 Integration
Context-Aware Question Answering
Research Paper Summarization
Follow-up Question Handling
📚 Additional Features
Citation Generation
Source References
Conversation Memory
Response Time Tracking
📊 Evaluation Metrics
Retrieval Quality
Response Quality
Hallucination Rate
🎨 User Interface
Streamlit Dashboard
Chatbot Style Interface
Analytics Dashboard
Document Information Panel
System Health Monitoring
🏗️ Architecture

PDF Upload
↓
Text Extraction
↓
Chunk Creation
↓
Embedding Generation
↓
FAISS Vector Store
↓
Semantic Retrieval
↓
Cross Encoder Reranking
↓
Context Construction
↓
Llama 3.2 Generation
↓
Answer + Citations + Metrics

🛠️ Technology Stack
Frontend
Streamlit
Backend
Python 3.10
Embedding Model
all-MiniLM-L6-v2
Vector Database
FAISS
Re-Ranker
cross-encoder/ms-marco-MiniLM-L-6-v2
LLM
Ollama
Llama 3.2
Libraries
LangChain
Sentence Transformers
Pandas
NumPy
Plotly
PyMuPDF
📂 Project Structure
rag-knowledge-assistant/
│
├── app/
│   ├── ingestion/
│   ├── retrieval/
│   ├── generation/
│   ├── memory/
│   ├── evaluation/
│
├── data/
│   ├── vector_db/
│
├── uploads/
│
├── run.py
├── requirements.txt
├── README.md
⚙️ Installation
Clone Repository
git clone https://github.com/your-username/RAG-Knowledge-Assistant.git

cd RAG-Knowledge-Assistant
Create Virtual Environment
python -m venv venv
Activate Virtual Environment
Windows
venv\Scripts\activate
Linux / Mac
source venv/bin/activate
Install Dependencies
pip install -r requirements.txt
🤖 Install Ollama

Download and install:

https://ollama.com

Pull Llama 3.2:

ollama pull llama3.2

Verify:

ollama list
▶️ Run Application
streamlit run run.py

Application URL:

http://localhost:8501
📈 Evaluation Metrics
Retrieval Quality

Measures how effectively the retriever fetches relevant chunks.

Response Quality

Measures keyword coverage and answer relevance.

Hallucination Rate

Measures unsupported information generated outside retrieved context.

📋 Sample Questions
Research Paper Questions
What is self-attention?
What datasets were used?
What BLEU scores were achieved?
How does the Transformer differ from RNNs?
Summarize the paper.
Explain the paper in detail.
What are the main contributions?
What conclusions were reached?
🎯 Future Enhancements
Multi-PDF Support
Hybrid Search (BM25 + Vector Search)
Query Logging
User Authentication
Chat Export (PDF/CSV)
Confidence Scoring
Advanced Analytics Dashboard
