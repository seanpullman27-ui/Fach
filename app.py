import streamlit as st
import requests
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

st.set_page_config(page_title="🦴 Facharzt Orthopädie Trainer")
st.title("🦴 Facharzt Orthopädie & Unfallchirurgie Trainer")

# أدخل مفتاح OpenAI
openai_api_key = st.text_input("OpenAI API Key", type="password")

# رابط PDF من Google Drive
pdf_links = [
    "https://drive.google.com/uc?id=1b5TuW29kKL16idA7YcrEfuuQD4_nuIC3"
]

@st.cache_resource
def load_documents():
    docs = []
    for link in pdf_links:
        r = requests.get(link)
        f = BytesIO(r.content)
        reader = PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        docs.append({"page_content": text, "metadata": {}})
    
    # تقسيم النص
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma.from_documents(split_docs, embeddings)
    
    return vectorstore

if openai_api_key:
    vectorstore = load_documents()
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=openai_api_key
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )
    
    mode = st.selectbox(
        "Modus",
        ["Normale Frage", "Zusammenfassung", "Prüfungssimulation"]
    )
    
    question = st.text_input("Frage eingeben")
    
    if st.button("Senden"):
        if mode == "Zusammenfassung":
            question = f"Erstelle eine prüfungsrelevante Zusammenfassung: {question}"
        
        if mode == "Prüfungssimulation":
            question = f"Simuliere eine mündliche Facharztprüfung: {question}"
        
        answer = qa_chain.run(question)
        st.write(answer)
