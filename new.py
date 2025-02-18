import os
import pdfplumber
import pytesseract
import pandas as pd
from pdf2image import convert_from_path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


def extract_tables_from_pdf(pdf_path):
    """Enhanced table extraction with no limits on table size or count"""
    tables = []

    with pdfplumber.open(pdf_path) as pdf:
        # Set maximum pages to None to process all pages
        total_pages = len(pdf.pages)
        print(f"Processing {total_pages} pages for tables...")

        for page_num, page in enumerate(pdf.pages, 1):
            print(f"Processing page {page_num}/{total_pages}")

            # Enhanced table detection settings with maximum coverage
            table_settings = {
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "explicit_vertical_lines": page.curves + page.edges,
                "explicit_horizontal_lines": page.curves + page.edges,
                "intersection_y_tolerance": 12,  # Increased tolerance
                "intersection_x_tolerance": 12,
                "snap_tolerance": 6,  # Increased for better cell detection
                "join_tolerance": 6,
                "edge_min_length": 1,  # Reduced to catch smaller cells
                "min_words_vertical": 1,  # Reduced to catch all possible tables
                "min_words_horizontal": 1,
                "keep_blank_chars": True,
                "text_tolerance": 5,
                "text_x_tolerance": 5,
                "text_y_tolerance": 5,
                "merge_duplicate_headers": True
            }

            try:
                # First attempt with text-based strategy
                page_tables = page.extract_tables(table_settings)

                if not page_tables:
                    # Try multiple strategies if no tables found
                    strategies = [
                        {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
                        {"vertical_strategy": "explicit", "horizontal_strategy": "lines"},
                        {"vertical_strategy": "text", "horizontal_strategy": "lines"},
                        {"vertical_strategy": "lines", "horizontal_strategy": "text"}
                    ]

                    for strategy in strategies:
                        table_settings.update(strategy)
                        page_tables = page.extract_tables(table_settings)
                        if page_tables:
                            break

                for table in page_tables:
                    if table:  # Remove minimum row requirement
                        # Process and clean the table
                        cleaned_table = []
                        max_cols = max(len(row) if row else 0 for row in table)

                        for row in table:
                            cleaned_row = []
                            # Extend short rows to match the maximum column count
                            row = row if row else []
                            row.extend([None] * (max_cols - len(row)))

                            for cell in row:
                                if cell is None:
                                    cleaned_row.append("")
                                else:
                                    # Enhanced cell cleaning
                                    cleaned_cell = str(cell).strip()
                                    cleaned_cell = ' '.join(cleaned_cell.split())
                                    # Remove common PDF artifacts
                                    cleaned_cell = cleaned_cell.replace('\x00', '')
                                    cleaned_row.append(cleaned_cell)
                            cleaned_table.append(cleaned_row)

                        # Handle headers
                        headers = cleaned_table[0] if cleaned_table else []
                        data = cleaned_table[1:] if len(cleaned_table) > 1 else cleaned_table

                        # Generate headers if missing
                        if not any(headers) or all(h == '' for h in headers):
                            headers = [f"Column_{i + 1}" for i in range(max_cols)]

                        # Ensure header uniqueness while preserving meaningful names
                        unique_headers = []
                        header_counts = {}
                        for header in headers:
                            header = header if header.strip() else f"Column_{len(unique_headers) + 1}"
                            if header in header_counts:
                                header_counts[header] += 1
                                unique_headers.append(f"{header}_{header_counts[header]}")
                            else:
                                header_counts[header] = 1
                                unique_headers.append(header)

                        # Create DataFrame with no size limits
                        df = pd.DataFrame(data, columns=unique_headers)

                        # Minimal post-processing to preserve data
                        df = df.replace(r'^\s*$', pd.NA, regex=True)  # Only convert completely empty cells to NA
                        df = df.dropna(how='all')  # Only drop rows that are completely empty

                        if not df.empty:
                            tables.append({
                                'page': page_num,
                                'data': df,
                                'rows': len(df),
                                'columns': len(df.columns)
                            })
                            print(f"Found table on page {page_num} with {len(df)} rows and {len(df.columns)} columns")

            except Exception as e:
                print(f"Warning: Table extraction issue on page {page_num}: {str(e)}")
                continue

    print(f"Total tables extracted: {len(tables)}")
    return tables


def process_pdf(pdf_path):
    """Process PDF with unlimited table extraction"""
    text = extract_text_from_pdf(pdf_path)
    tables = extract_tables_from_pdf(pdf_path)

    # Convert all tables to text without limits
    table_texts = []
    for table_info in tables:
        try:
            table_header = f"\n[Table on Page {table_info['page']}]\n"
            table_metadata = f"Columns: {table_info['columns']} | Rows: {table_info['rows']}\n"
            md_table = table_info['data'].to_markdown(index=False, tablefmt="grid")
            table_texts.append(f"{table_header}{table_metadata}{md_table}\n[End Table]\n")
        except Exception as e:
            print(f"Warning: Failed to convert table from page {table_info['page']} to text: {str(e)}")

    return {
        "text": text + "\n".join(table_texts),
        "tables": tables
    }


def answer_query(vectorstore, query):
    """Generate answer using retrieval-augmented generation with unlimited table references"""
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),  # Increased from 3 to 5
        return_source_documents=True
    )
    result = qa_chain.invoke({"query": query})

    # Return all relevant tables without limit
    table_refs = [doc.page_content for doc in result["source_documents"]
                  if "[Table" in doc.page_content]

    return {
        "answer": result["result"],
        "references": table_refs  # No limit on number of references
    }


def main():
    pdf_path = input("Enter PDF file path: ").strip()
    if not os.path.exists(pdf_path):
        print("Error: File not found!")
        return

    print("\nProcessing PDF...")
    processed_data = process_pdf(pdf_path)
    vectorstore = store_in_faiss(processed_data["text"])

    print("\nProcessing complete! Ask questions about the document:")
    while True:
        query = input("\nYour question (type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break

        result = answer_query(vectorstore, query)
        print("\nAnswer:", result["answer"])
        if result["references"]:
            print("\nRelevant tables found in document:")
            for table in result["references"]:
                print(table)
        print("-" * 80)


if __name__ == "__main__":
    main()