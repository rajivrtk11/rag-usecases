import fitz  # PyMuPDF
import base64
import os
from PIL import Image
from io import BytesIO

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load Gemini Pro Vision
vision_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2,
    convert_system_message_to_human=True
)

def encode_image_to_base64_data_uri(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"

def extract_text_and_images_from_pdf(pdf_path: str):
    doc = fitz.open(pdf_path)
    results = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        images = []

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            image_data_uri = encode_image_to_base64_data_uri(image)
            images.append(image_data_uri)

        # Send both text and images to Gemini Vision
        response = vision_llm.invoke([
            HumanMessage(content=[
                {"type": "text", "text": f"Extract information from this PDF page:\n{text}"},
                *[
                    {"type": "image_url", "image_url": image_data_uri}
                    for image_data_uri in images
                ]
            ])
        ])

        print(f"\n--- Page {page_num + 1} ---")
        print(response.content)
        results.append(response.content)

    return results

if __name__ == "__main__":
    pdf_path = "IPCC_AR6_SYR_SPM.pdf"  # Replace with your actual PDF path
    extract_text_and_images_from_pdf(pdf_path)
