import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

# 🔐 GROQ (Streamlit Cloud safe)
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# 🎨 PAGE CONFIG
st.set_page_config(page_title="Notebar PDF Chatbot", layout="wide")

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
        chunk_size=800,          # 🔥 improved
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"   # 🔥 better embeddings
    )

    st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)

# 🏷️ TITLE
st.markdown("# 📘 Notebar - PDF Chatbot")

# ✅ SUCCESS
if file and st.session_state.vector_store:
    st.success("✅ PDF processed successfully!")

# 💬 CHAT DISPLAY
for msg in st.session_state.chat:
    if msg["role"] == "user":
        st.markdown(f"😎 {msg['content']}")
    else:
        st.markdown(f"🤖 {msg['content']}")

# 📝 INPUT
query = st.chat_input("Ask something about your PDF...")

# 🚨 HANDLE QUERY
if query:
    if st.session_state.vector_store is None:
        st.warning("⚠️ Please upload a PDF first!")
        st.stop()

    # Save user message
    st.session_state.chat.append({"role": "user", "content": query})

    # 🔍 SEARCH (🔥 increased k)
    docs = st.session_state.vector_store.similarity_search(query, k=5)

    context = "\n".join([doc.page_content for doc in docs])

    thinking = st.empty()
    thinking.markdown("⏳ Thinking...")

    try:
        # 🤖 GROQ CALL (🔥 better model + prompt)
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """
You are a helpful assistant.

Answer using ONLY the provided context.
Format your answer clearly:
- Use bullet points or numbered lists
- Each point must be on a new line
- Keep answers clean and structured

If answer is not found, say: Not found in document
"""
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                }
            ],
            model="llama-3.3-70b-versatile"   # 🔥 upgraded model
        )

        answer = response.choices[0].message.content

        # 🔧 FIX formatting (force vertical output)
        answer = answer.replace(". ", ".\n")

    except Exception as e:
        answer = f"❌ Error: {str(e)}"

    thinking.empty()

    # Save response
    st.session_state.chat.append({"role": "assistant", "content": answer})

    st.rerun()
