import os
import json
import fitz  # PyMuPDF
import datetime
from sentence_transformers import SentenceTransformer, util

# Folder paths inside the Docker container
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models", "all-MiniLM-L6-v2")
INPUT_DIR = "/app/input/Collection 2"
PDF_DIR = os.path.join(INPUT_DIR, "PDFs")
INPUT_JSON = os.path.join(INPUT_DIR, "challenge1b_input.json")
OUTPUT_DIR = "/app/output"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "challenge1b_output.json")

# Load MiniLM model from local disk
model = SentenceTransformer(MODEL_DIR)


def extract_sections_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    sections = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")['blocks']
        for block in blocks:
            if block['type'] == 0:  # text block
                for line in block['lines']:
                    line_text = " ".join([span['text'] for span in line['spans']])
                    font_size = line['spans'][0]['size']
                    if font_size >= 12 and len(line_text.strip()) > 5:
                        sections.append({
                            "text": line_text.strip(),
                            "page": page_num + 1
                        })
    return sections


def rank_sections(sections, persona, job):
    query = f"{persona} needs to: {job}"
    query_embedding = model.encode(query, convert_to_tensor=True)

    ranked_sections = []
    for section in sections:
        section_embedding = model.encode(section['text'], convert_to_tensor=True)
        score = util.pytorch_cos_sim(query_embedding, section_embedding).item()
        ranked_sections.append((score, section))

    ranked_sections.sort(key=lambda x: x[0], reverse=True)
    return ranked_sections


def generate_output_json(input_files, persona, job, ranked_sections, top_k=5):
    timestamp = datetime.datetime.now().isoformat()

    output = {
        "metadata": {
            "documents": input_files,
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": timestamp
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }

    for idx, (score, section) in enumerate(ranked_sections[:top_k]):
        output["extracted_sections"].append({
            "document": section.get("doc", "unknown.pdf"),
            "page_number": section['page'],
            "section_title": section['text'],
            "importance_rank": idx + 1
        })
        output["subsection_analysis"].append({
            "document": section.get("doc", "unknown.pdf"),
            "page_number": section['page'],
            "refined_text": section['text']
        })

    return output


def main():
    # Load persona + job from input JSON
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        job_data = json.load(f)

    persona = job_data.get("persona", "Generic User")
    job = job_data.get("job_to_be_done", "Understand document")

    input_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    all_sections = []

    for file in input_files:
        path = os.path.join(PDF_DIR, file)
        sections = extract_sections_from_pdf(path)
        for sec in sections:
            sec['doc'] = file
        all_sections.extend(sections)

    ranked = rank_sections(all_sections, persona, job)
    output_json = generate_output_json(input_files, persona, job, ranked)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
