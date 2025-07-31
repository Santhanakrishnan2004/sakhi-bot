
from flask import Flask, request, jsonify, render_template
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# Load and split PDF
pdf_path = "data/sakhi_pdf.pdf"  # put your PDF here
loader = PyPDFLoader(pdf_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
split_docs = splitter.split_documents(documents)

# Embedding + Vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(split_docs, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Prompt Template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are Sakhi-bot 🤖, a friendly assistant. Say “Hi! I'm Sakhi-bot 🤖” when greeted at first time only .

<b>Question</b>: {question}

<b>Context</b>:
{context}

Instructions:
- Respond clearly in full sentences.
- Avoid saying "Based on the context" or that you're an AI.
- Say “Not found in available data.” only if truly no info.
- dont say u have provided , you .
- talk in third man manner

<b>Answer</b>:
"""
)

# Home page
@app.route("/")
def index():
    return render_template("index.html")

# Ask endpoint
@app.route("/ask", methods=["POST"])
def ask():
    try:
        question = request.json.get("message", "").lower()
        greetings = ["hi", "hello", "hey", "hai", "hii"]
        if any(greet in question for greet in greetings):
            return jsonify({"reply": "Hi! I'm Sakhi-bot 🤖. Ask me anything!"})

        llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.2)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt}
        )

        result = qa_chain.invoke(question)
        answer = result["result"]
        return jsonify({"reply": answer})
    except Exception as e:
        return jsonify({"reply": f"❌ Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
