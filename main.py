import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

st.set_page_config(page_title="Notebar PDF Chatbot", layout="wide")

if "chat" not in st.session_state:
    st.session_state.chat = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

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

if file and st.session_state.vector_store is None:
    text = ""
    pdf = PdfReader(file)

    for page in pdf.pages:
        content = page.extract_text()
        if content:
            text += content

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,         
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"   
    )

    st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)

st.markdown("# 📘 Notebar - PDF Chatbot")

if file and st.session_state.vector_store:
    st.success("✅ PDF processed successfully!")

for msg in st.session_state.chat:
    if msg["role"] == "user":
        st.markdown(f"📘 {msg['content']}")
    else:
        st.markdown(f"✅ {msg['content']}")

query = st.chat_input("Ask something about your PDF...")

if query:
    if st.session_state.vector_store is None:
        st.warning("⚠️ Please upload a PDF first!")
        st.stop()

    st.session_state.chat.append({"role": "user", "content": query})

    docs = st.session_state.vector_store.similarity_search(query, k=5)

    context = "\n".join([doc.page_content for doc in docs])

    thinking = st.empty()
    thinking.markdown("⏳ Thinking...")

    try:
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
            model="llama-3.3-70b-versatile"  
        )

        answer = response.choices[0].message.content

        answer = answer.replace(". ", ".\n")

    except Exception as e:
        answer = f"❌ Error: {str(e)}"

    thinking.empty()

    # Save response
    st.session_state.chat.append({"role": "assistant", "content": answer})

    st.rerun()
