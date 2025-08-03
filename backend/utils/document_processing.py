import os
from typing import List
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain_core.messages import HumanMessage
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gemini Vision model for text + image analysis
def get_vision_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.2,
        convert_system_message_to_human=True,
    )

# Gemini embedding model
def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(
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
    try:
        import numpy as np
        # Placeholder for chart OCR logic
        return "Chart data extraction placeholder."
    except Exception as e:
        return f"Chart OCR failed: {e}"

def analyze_page_for_text_and_images(page, page_number, vision_llm):
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

    pre_vision_text = (
        f"**Extracted Text:**\n{text}\n\n"
        f"**OCR Text from Images:**\n{'\n\n'.join(ocr_texts)}\n\n"
        f"**Chart OCR Data:**\n{'\n\n'.join(chart_ocr_texts)}"
    )

    response = vision_llm.invoke([
        HumanMessage(content=[
            {
                "type": "text",
                "text": (
                    "You are a professional data analyst. "
                    "Carefully analyze any charts, graphs, or tables in this image. "
                    "Identify each line or data series separately by their color and legend label. "
                    "Extract all available numeric data points for each series, including the exact or approximate value for each year shown on the x-axis. "
                    "When reading line charts, also consider line color, legend labels, axis labels, and units. "
                    "Provide the extracted data as structured tables, with a clear indication of which series the data belongs to. "
                    "If any axis scale or reference year is indicated, take this into account in the numeric values. "
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

    combined_content = (
        f"{pre_vision_text}\n\n"
        f"**Gemini Vision Summary and Data:**\n{response.content}"
    )
    return combined_content

def extract_and_analyze_pages(pdf_path: str) -> List[str]:
    doc = fitz.open(pdf_path)
    analyzed_pages = []
    vision_llm = get_vision_llm()
    for i, page in enumerate(doc):
        combined_text = analyze_page_for_text_and_images(page, i + 1, vision_llm)
        analyzed_pages.append(combined_text)
    return analyzed_pages

def store_in_vector_db(page_contents: List[str], index_path: str = "faiss_index"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for i, content in enumerate(page_contents):
        chunks = splitter.split_text(content)
        docs.extend([
            Document(page_content=chunk, metadata={"page": i + 1})
            for chunk in chunks
        ])
    embedding_model = get_embedding_model()
    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local(index_path)
    return vectorstore

def load_vector_db(index_path: str = "faiss_index"):
    embedding_model = get_embedding_model()
    return FAISS.load_local(
        index_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )
