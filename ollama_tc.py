from langchain_ollama import OllamaLLM
from pydantic import BaseModel
from typing import List
import gradio as gr
import fitz  # PyMuPDF
import json

# Define the Pydantic models
class Block(BaseModel):
    Test_Case_Id: str
    objective: str
    Test_Data: str
    Steps: str

class TC(BaseModel):
    test_cases: List[Block]

# Initialize the language model
llm = OllamaLLM(model="llama3.1")

# Function to parse the output into the correct format
def parse_output(output):
    try:
        return Block(**json.loads(output))
    except (json.JSONDecodeError, TypeError):
        return None

# Function to generate test cases
def generate_test_cases(pdf_file_path):
    # Read the PDF content using PyMuPDF
    with open(pdf_file_path, "rb") as f:
        pdf_bytes = f.read()
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    input_data = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        input_data += page.get_text()
        print("Extracted Text:", input_data)  # Debugging line

    test_cases = []
    for chunk in llm.stream(f"Generate the test cases for the input data: {input_data}"):
        print("Model Output Chunk:", chunk)  # Debugging line
        parsed_chunk = parse_output(chunk)
        if parsed_chunk:
            test_cases.append(parsed_chunk)
    
    return TC(test_cases=test_cases)

# Gradio interface
def process_pdf(pdf_file):
    result = generate_test_cases(pdf_file.name)
    return result.json()

# Create Gradio interface
iface = gr.Interface(
    fn=process_pdf,
    inputs=gr.File(type="filepath", label="Upload PDF"),
    outputs="json",
    title="Test Case Generator",
    description="Upload a PDF file to generate test cases using a language model."
)

# Launch the interface
iface.launch()
