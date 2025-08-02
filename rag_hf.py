import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Load and Split Documents from PDF ---
loader = PyPDFLoader("rok-hjp-survind-kumar.pdf")  # Change your filename here
docs = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# --- Embedding Model ---
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# --- Vector Store ---
vectorstore = FAISS.from_documents(splits, embedding_model)
retriever = vectorstore.as_retriever()
vectorstore.save_local("sql_faiss_index")

# --- LLM ---
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.5)

# --- Prompt Template ---
prompt = ChatPromptTemplate.from_template(
    "Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}"
)

# --- RAG Chain ---
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)

# --- Run RAG ---
query = "Give me passenger details and journey information"
response = rag_chain.invoke({"query": query})
print(response["result"])
