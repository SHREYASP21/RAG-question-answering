import os
import pdfplumber
import pytesseract
import pandas as pd
from pdf2image import convert_from_path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load Hugging Face API Key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Load LLaMA-3 Model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# ✅ Extract both text & tables together (better accuracy)
def extract_text_and_tables(pdf_path):
    text = []
    tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract text
            page_text = page.extract_text() or ""
            if page_text.strip():
                text.append(page_text)

            # Extract tables
            table = page.extract_table()
            if table:
                df = pd.DataFrame(table).dropna(how="all")  # Remove empty rows
                tables.append(df.to_dict(orient="records"))

    # If no text is found, use OCR (for scanned PDFs)
    if not text:
        text = ["\n".join(pytesseract.image_to_string(img) for img in convert_from_path(pdf_path))]

    return {"text": "\n".join(text), "tables": tables}

# ✅ Store extracted text in FAISS for retrieval
def store_in_faiss(text):
    text_chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(text)
    vectorstore = FAISS.from_texts(text_chunks, embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    return vectorstore

# ✅ Answer queries using FAISS (Text) + Table Search
def answer_query(text_store, tables, query):
    # Retrieve answer from FAISS (text-based)
    vectorstore = store_in_faiss(text_store)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    answer = qa_chain.invoke({"query": query})["result"]

    # Search for relevant tables
    relevant_tables = []
    for table in tables:
        for row in table:
            if any(query.lower() in str(value).lower() for value in row.values()):
                relevant_tables.append(table)
                break  # Avoid adding duplicate tables

    return {"answer": answer, "tables": relevant_tables if relevant_tables else "No relevant table found"}

# ✅ Main execution
def main():
    pdf_path = input("Enter PDF file path: ").strip()
    if not os.path.exists(pdf_path):
        print("❌ Error: File not found!")
        return

    print("\n⏳ Processing PDF...\n")
    processed_data = extract_text_and_tables(pdf_path)

    print("\n✅ Processing Complete! You can now ask questions.\n")
    while True:
        query = input("❓ Ask a question (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break

        result = answer_query(processed_data["text"], processed_data["tables"], query)
        print("\n🤖 LLaMA-3's Answer:", result["answer"])

        # ✅ Display tables in a clean format
        if isinstance(result["tables"], list) and result["tables"]:
            print("\n🔹 Relevant Table(s):\n")
            for table in result["tables"]:
                df = pd.DataFrame(table)
                print(df.to_string(index=False))  # Prints tables in a readable format
        else:
            print("\nNo relevant table found.")

        print("-" * 50)

if __name__ == "__main__":
    main()
