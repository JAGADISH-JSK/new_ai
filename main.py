import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
import os
from dotenv import load_dotenv

# 🔐 Load API Key
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 🎨 PAGE CONFIG
st.set_page_config(page_title="Notebar PDF Chatbot", layout="wide")

# 🎨 CUSTOM CSS
st.markdown("""
<style>
.user-msg {
    background-color: #1f2937;
    padding: 14px;
    border-radius: 12px;
    margin: 10px 0;
    display: flex;
    gap: 10px;
}
.bot-msg {
    background-color: #111827;
    padding: 14px;
    border-radius: 12px;
    margin: 10px 0;
    display: flex;
    gap: 10px;
}
</style>
""", unsafe_allow_html=True)

# 🧠 SESSION STATE
if "chat" not in st.session_state:
    st.session_state.chat = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# 📁 SIDEBAR
with st.sidebar:
    st.title("📁 My Notes")

    file = st.file_uploader("Upload notes PDF", type="pdf")

    if file:
        pdf_reader = PdfReader(file)

        st.success("✅ PDF uploaded")
        st.markdown("### 📊 PDF Info")
        st.write(f"📄 Name: {file.name}")
        st.write(f"📦 Size: {round(file.size / 1024, 2)} KB")
        st.write(f"📑 Pages: {len(pdf_reader.pages)}")

        if st.button("🧹 Clear Chat"):
            st.session_state.chat = []

# 📄 PROCESS PDF
if file and st.session_state.vector_store is None:
    text = ""
    pdf = PdfReader(file)

    for page in pdf.pages:
        content = page.extract_text()
        if content:
            text += content

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings()
    st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)

# 🏷️ TITLE
st.markdown("# 📘 Notebar - PDF Chatbot")

# ✅ SUCCESS
if file and st.session_state.vector_store:
    st.success("✅ PDF processed successfully!")

# 💬 CHAT DISPLAY
for msg in st.session_state.chat:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="user-msg">😎 {msg['content']}</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="bot-msg">🤖 {msg['content']}</div>
        """, unsafe_allow_html=True)

# 📝 INPUT
query = st.chat_input("Ask something about your PDF...")

# 🚨 HANDLE QUERY
if query:
    if st.session_state.vector_store is None:
        st.warning("⚠️ Please upload a PDF first!")
        st.stop()

    # Save user message
    st.session_state.chat.append({"role": "user", "content": query})

    # 🔍 SEARCH
    docs = st.session_state.vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    thinking = st.empty()
    thinking.markdown("⏳ Thinking...")

    try:
        # 🤖 GROQ CALL
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Answer ONLY using the PDF context. If not found, say 'Not found in document'."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            model="llama-3.3-70b-versatile"
        )

        answer = response.choices[0].message.content

    except Exception as e:
        answer = f"❌ Error: {str(e)}"

    thinking.empty()

    # Save response
    st.session_state.chat.append({"role": "assistant", "content": answer})

    st.rerun()