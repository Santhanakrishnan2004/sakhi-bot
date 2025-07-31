# from flask import Flask, request, jsonify, render_template
# import os
# import requests
# from urllib.parse import urljoin
# from bs4 import BeautifulSoup
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_google_vertexai import ChatVertexAI
# from langchain.chains import RetrievalQA
# from langchain_core.prompts import PromptTemplate
# from datetime import datetime
# import csv
# import smtplib
# from email.mime.text import MIMEText
# from langchain_google_genai import ChatGoogleGenerativeAI
# import pyodbc
# # from dotenv import load_dotenv
# import os

# app = Flask(__name__)
# # ✅ Test write access to 'logs' folder
# try:
#     os.makedirs("logs", exist_ok=True)  # Ensure the folder exists
#     with open(os.path.join("logs", "test_write.txt"), "w", encoding="utf-8") as f:
#         f.write("✅ Flask has write permission.\n")
#     print("✅ Log folder write test passed.")
# except Exception as e:
#     print("❌ File write test failed:", e)

# # Ensure data directory exists
# os.makedirs("data", exist_ok=True)

# # load_dotenv()

# GOOGLE_API_KEY="AIzaSyDu8-wXGQNgVfQ2CSROie7Epf45Dp37DnY"


# # ✅ Now you can access the key
# # GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# # ✅ Set Google API Key
# # ✅ Target URLs
# TARGET_URLS = [
#     "https://icsr.iitm.ac.in/Projects.php",
#     "https://icsr.iitm.ac.in/Recuritment.php",
#     "https://icsr.iitm.ac.in/Purchase.php",
#     "https://icsr.iitm.ac.in/Accounts.php",
#     "https://icsr.iitm.ac.in/patent.php",
#     "https://icsr.iitm.ac.in/conffacility.php",
#     "https://icsr.iitm.ac.in/facility.php",
#     "https://icsr.iitm.ac.in/contact.html",
#     "https://icsr.iitm.ac.in/Form.php",
#     "https://icsr.iitm.ac.in/Hotel.php",
#     "https://icsr.iitm.ac.in/airline.php",
#     "https://icsr.iitm.ac.in/vechile.php",
#     "https://icsr.iitm.ac.in/videos.php"
# ]

# def scrape_icsr_page(url):
#     try:
#         response = requests.get(url)
#         soup = BeautifulSoup(response.content, "html.parser")
#         for tag in soup(["script", "style", "noscript"]):
#             tag.decompose()
#         for a in soup.find_all("a", href=True):
#             full_link = urljoin(url, a['href'])
#             if a.string:
#                 a.insert_after(f" (Link: {full_link})")
#         text = soup.get_text(separator="\n", strip=True)
#         print(f"✅ Scraped {url}: {len(text)} chars")
#         return Document(page_content=text, metadata={"url": url})
#     except Exception as e:
#         print(f"❌ Error scraping {url}: {e}")
#         return None

# def load_csv_as_documents(csv_path):
#     documents = []
#     try:
#         with open(csv_path, newline='', encoding='cp1252') as csvfile:
#             reader = csv.DictReader(csvfile)
#             for row in reader:
#                 content = "\n".join([f"{key}: {value}" for key, value in row.items()])
#                 documents.append(Document(page_content=content, metadata={"source": "csv"}))
#         print(f"✅ Loaded {len(documents)} documents from CSV")
#     except Exception as e:
#         print(f"❌ Error loading CSV: {e}")
#     return documents

# web_documents = [doc for url in TARGET_URLS if (doc := scrape_icsr_page(url))]
# csv_docs = load_csv_as_documents("data/qa_datasets.csv")
# all_documents = web_documents + csv_docs

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1500,
#     chunk_overlap=200,
#     separators=["\n\n", "\n", ".", " ", ""]
# )
# split_docs = splitter.split_documents(all_documents)

# embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vectorstore = FAISS.from_documents(split_docs, embedding_model)
# retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
# You are an assistant trained to help with information from the ICSR IIT Madras website.

# if greeting such as hi , hello , just into about ur self such as asssitant of icsr
# and dont start with Based on the provided text, 

# dont use "Based on the provided text" line when answering

# <b>Question</b>: {question}

# <b>Context</b>:
# {context}

# Instructions:
# - Respond clearly and professionally in full sentences.
# - Whenever a form (e.g., Form S7) is mentioned, describe it briefly and provide a clickable link in this format:
#   <a href='https://icsr.iitm.ac.in/Form.php' target='_blank'>Form S7</a>
# - Format answers using <b>, <i>, <ul>, <li>, <a>, and <br> tags where appropriate.
# - Do not mention you are an AI or that you cannot provide information.
# - Only say "Not found in available data." if the answer is truly unavailable.

# <b>Answer</b>:


# (Note: Dean (IC & SR): Prof. Manu Santhanam – deanicsr@zmail.iitm.ac.in – Ext: 8060, Secretary: 8061)
# """
# )


# app = Flask(__name__)



# # ✅ Database config
# DB_SERVER = "10.18.0.48"
# DB_NAME = "Talk2Tula"
# DB_USER = "pydev_chatbot"
# DB_PASSWORD = "PYd#vCh@tb0t"

# # ✅ DB connection helper
# def get_db_connection():
#     conn_str = (
#         f"DRIVER={{ODBC Driver 17 for SQL Server}};"
#         f"SERVER={DB_SERVER};DATABASE={DB_NAME};"
#         f"UID={DB_USER};PWD={DB_PASSWORD}"
#     )
#     return pyodbc.connect(conn_str)


# @app.route("/ask", methods=["POST"])
# def ask():
#     try:
#         question = request.json.get("message", "")
#         piname = request.args.get("piname", "").strip()
#         mailid = request.args.get("mailid", "").strip()
#         piid = request.args.get("piid", "").strip()
#         sessionid = request.args.get("sessionid", "").strip()
#         print(f"📥 sessionid: {sessionid}")

#         if piid and sessionid:
#             try:
#                 conn = get_db_connection()
#                 cursor = conn.cursor()

#                 # Check if same session already logged for this user
#                 cursor.execute("""
#                     SELECT COUNT(*) FROM TblChatLog 
#                     WHERE UserID = ? AND sessionid = ?
#                 """, (piid, sessionid))
#                 existing_session = cursor.fetchone()[0]

#                 if existing_session == 0:
#                     print(f"🆕 New session — inserting row for user {piid}")
#                     cursor.execute("""
#                         INSERT INTO TblChatLog (UserID, FirstChatTimestamp, sessionid)
#                         VALUES (?, ?, ?)
#                     """, (piid, datetime.now(), sessionid))
#                 else:
#                     print(f"✅ Session {sessionid} already logged for user {piid}")

#                 conn.commit()
#                 cursor.close()
#                 conn.close()
#             except Exception as e:
#                 print(f"❌ Error logging session: {e}")

#         # ✅ Generate answer
#         llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.1 ,max_output_tokens=500)
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             retriever=retriever,
#             chain_type="stuff",
#             chain_type_kwargs={"prompt": prompt}
#         )
#         response = qa_chain.invoke(question)
#         answer = response["result"]

#         return jsonify({"reply": answer})

#     except Exception as e:
#         print(f"❌ Error in /ask: {e}")
#         return jsonify({"reply": f"❌ Error: {str(e)}"})


# @app.route("/feedback", methods=["POST"])
# def save_feedback():
#     data = request.get_json()
#     feedback_text = data.get("feedback", "").strip()

#     piname = request.args.get("piname", "").strip()
#     mailid = request.args.get("mailid", "").strip()
#     piid = request.args.get("piid", "").strip()

#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()

#         # ✅ Insert feedback into the DB
#         cursor.execute("""
#             INSERT INTO TblChatFeedback (UserID, UserName, Feedback, dttm)
#             VALUES (?, ?, ?, ?)
#         """, (piid, piname or "Anonymous", feedback_text, datetime.now()))
        
#         conn.commit()
#         cursor.close()
#         conn.close()
#         print(f"✅ Feedback saved to DB for {piid}")
#     except Exception as e:
#         print(f"❌ DB insert failed: {e}")
#         return jsonify({"status": "error", "message": "❌ Failed to save feedback to DB", "error": str(e)}), 500

#     # ✅ Optionally also send email
#     try:
#         smtp_address = "smtp.office365.com"
#         sender_email = "noreply@icsrpis.iitm.ac.in"
#         app_password = "N0replY@123$"
#         subject = "📝 New Feedback Received"

#         body = f"Feedback from {piname or 'Anonymous'} (ID: {piid}):\n\n{feedback_text}"

#         msg = MIMEText(body)
#         msg["Subject"] = subject
#         msg["From"] = sender_email
#         msg["To"] =mailid

#         with smtplib.SMTP(smtp_address, 587) as server:
#             server.starttls()
#             server.login(sender_email, app_password)
#             server.sendmail(sender_email, [mailid], msg.as_string())

#         return jsonify({"status": "ok", "message": f"✅ Thanks for giving feedback, {piname}!"})
#     except Exception as e:
#         return jsonify({"status": "error", "message": "❌ Failed to send feedback email.", "error": str(e)}), 500



# @app.route("/feedback_response", methods=["POST"])
# def log_response_feedback():
#     data = request.get_json()
#     prompt = data.get("prompt", "")
#     response = data.get("response", "")
#     feedback = data.get("feedback", "")

#     piname = request.args.get("piname", "").strip()
#     mailid = request.args.get("mailid", "").strip()
#     piid = request.args.get("piid", "").strip()

#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()
#         cursor.execute("""
#             INSERT INTO TblResponseFeedback (UserID, Prompt, BotResponse, Feedback, dttm)
#             VALUES (?, ?, ?, ?, ?)
#         """, (piid, prompt, response, feedback, datetime.now()))
#         conn.commit()
#         cursor.close()
#         conn.close()
#     except Exception as e:
#         return jsonify({"status": "error", "message": "❌ Failed to log thumbs feedback.", "error": str(e)})

#     return jsonify({"status": "ok"})



# @app.route("/")
# def index():
#     piname = request.args.get("piname", "").strip()
#     return render_template("index.html", piname=piname)
# @app.route("/api/get_sessionid")
# def get_sessionid():
#     piid = request.args.get("piid", "").strip()
#     if not piid:
#         return jsonify({"error": "Missing piid"}), 400

#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()
#         cursor.execute("SELECT sessionid FROM TblChatLog WHERE UserID = ?", (piid,))
#         row = cursor.fetchone()
#         cursor.close()
#         conn.close()

#         return jsonify({"sessionid": row[0] if row else ""})
#     except Exception as e:
#         print(f"❌ Error fetching sessionid for {piid}: {e}")
#         return jsonify({"sessionid": ""})


# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=8080)
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
