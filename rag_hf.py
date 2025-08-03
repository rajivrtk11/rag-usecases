import os
import base64
from io import BytesIO
import fitz  # PyMuPDF
from PIL import Image
from dotenv import load_dotenv
import pytesseract
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Optional chart OCR library
import plotdigitizer

# Load environment variables
load_dotenv()

# Gemini Vision model for text + image analysis
vision_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2,
    convert_system_message_to_human=True,
)

# Gemini embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def encode_image_to_base64_data_uri(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"

def ocr_from_image(image: Image.Image) -> str:
    text = pytesseract.image_to_string(image)
    return text.strip()

def extract_chart_data(image: Image.Image) -> str:
    """
    Attempt to extract numeric data from charts using plotdigitizer.
    Returns a string representation of the extracted data.
    """
    try:
        import numpy as np
        img_array = np.array(image)

        digitizer = Digitizer(image=img_array)
        # NOTE: plotdigitizer is usually interactive.
        # Here we provide a placeholder string to demonstrate integration.
        return "Chart data extraction placeholder - integrate your workflow here."
    except Exception as e:
        return f"Chart OCR failed: {e}"

def analyze_page_for_text_and_images(page, page_number):
    text = page.get_text()
    images = []
    ocr_texts = []
    chart_ocr_texts = []

    for img in page.get_images(full=True):
        xref = img[0]
        base_image = page.parent.extract_image(xref)
        image = Image.open(BytesIO(base_image["image"])).convert("RGB")
        image_data_uri = encode_image_to_base64_data_uri(image)
        images.append(image_data_uri)

        ocr_text = ocr_from_image(image)
        if ocr_text:
            ocr_texts.append(ocr_text)

        chart_text = extract_chart_data(image)
        if chart_text:
            chart_ocr_texts.append(chart_text)

    # Combine all extracted text before Gemini Vision
    pre_vision_text = (
        f"**Extracted Text:**\n{text}\n\n"
        f"**OCR Text from Images:**\n{'\n\n'.join(ocr_texts)}\n\n"
        f"**Chart OCR Data:**\n{'\n\n'.join(chart_ocr_texts)}"
    )

    # Call Gemini Vision
    response = vision_llm.invoke([
    HumanMessage(content=[
        {
                "type": "text",
                "text": (
                    "You are a professional data analyst. "
                    "Carefully analyze any charts, graphs, or tables in this image. "
                    "Identify each line or data series separately by their color and legend label (for example: 'European Union (blue line)' and 'Malta (orange line)'). "
                    "Extract all available numeric data points for each series, including the exact or approximate value for each year shown on the x-axis. "
                    "When reading line charts, also consider line color, legend labels, axis labels, and units. "
                    "Provide the extracted data as structured tables, with a clear indication of which series the data belongs to. "
                    "If any axis scale or reference year (such as '2010=100') is indicated, take this into account in the numeric values. "
                    "Then, summarize any additional descriptive text, footnotes, or source information found in the image.\n\n"
                    f"{pre_vision_text}"
                )
            },
            *[
                {"type": "image_url", "image_url": img_uri}
                for img_uri in images
            ]
        ])
    ])


    print(f"✅ Processed Page {page_number}")
    print(f"🔍 Gemini Vision Response: {response.content}")

    # Combine all information
    combined_content = (
        f"{pre_vision_text}\n\n"
        f"**Gemini Vision Summary and Data:**\n{response.content}"
    )
    return combined_content

def extract_and_analyze_pages(pdf_path: str) -> list[str]:
    doc = fitz.open(pdf_path)
    analyzed_pages = []

    for i, page in enumerate(doc):
        combined_text = analyze_page_for_text_and_images(page, i + 1)
        analyzed_pages.append(combined_text)

    return analyzed_pages

def store_in_vector_db(page_contents: list[str]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for i, content in enumerate(page_contents):
        chunks = splitter.split_text(content)
        docs.extend([
            Document(page_content=chunk, metadata={"page": i + 1})
            for chunk in chunks
        ])

    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local("faiss_index")
    print("✅ Vector DB saved to disk.")
    return vectorstore

def load_vector_db():
    return FAISS.load_local(
        "faiss_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )

def ask_question(vectorstore, question: str):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(
            model="models/gemini-1.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
        ),
        retriever=retriever,
        return_source_documents=True,
    )

    result = qa_chain.invoke({"query": question})
    print("\n💬 Question:", question)
    print("\n🧠 Answer:\n", result["result"])
    print("\n📚 Source Pages:", [doc.metadata["page"] for doc in result["source_documents"]])

if __name__ == "__main__":
    pdf_path = "document/IPCC_AR6_SYR_SPM.pdf"
    read_from_db = True

    if read_from_db:
        # Step 1: Extract and analyze each page with Gemini Vision + OCR + chart OCR
        content_pages = extract_and_analyze_pages(pdf_path)

        # Step 2: Store into FAISS vector database
        vectorstore = store_in_vector_db(content_pages)
    else:
        # Load existing vector database if not reading from PDF
        print("Loading existing vector database...")
        vectorstore = load_vector_db()

    # Step 3: Ask questions (RAG)
    ask_question(
        vectorstore,
        # "What was the Malta in 2017 for Chart II: Retail Trade?"
        # "What was the eu trade in 2011 for Chart II: Retail Trade?"
        # "what is malta house price index in 2017"
        "What this file is about? What are the main findings and conclusions?"
    )

