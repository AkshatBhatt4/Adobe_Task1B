import os
import json
import fitz  # PyMuPDF
import datetime
from sentence_transformers import SentenceTransformer, util

# Local paths
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models", "all-MiniLM-L6-v2")
ROOT_INPUT_DIR = BASE_DIR

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
            "input_documents": input_files,
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
            "section_title": section['text'],
            "importance_rank": idx + 1,
            "page_number": section['page']
        })
        output["subsection_analysis"].append({
            "document": section.get("doc", "unknown.pdf"),
            "refined_text": section['text'],
            "page_number": section['page']
        })

    return output


def process_collection(collection_path):
    input_json = os.path.join(collection_path, "challenge1b_input.json")
    pdf_dir = os.path.join(collection_path, "PDFs")
    output_json = os.path.join(collection_path, "challenge1b_output.json")

    if not os.path.exists(input_json) or not os.path.exists(pdf_dir):
        print(f"[!] Skipping {collection_path}: Missing input file or PDFs folder")
        return

    with open(input_json, 'r', encoding='utf-8') as f:
        job_data = json.load(f)

    # Extract job + persona
    persona = job_data.get("persona")
    if isinstance(persona, dict):
        persona = persona.get("role", "Generic User")

    job = job_data.get("job_to_be_done")
    if isinstance(job, dict):
        job = job.get("task", "Understand document")

    input_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    all_sections = []

    for file in input_files:
        path = os.path.join(pdf_dir, file)
        sections = extract_sections_from_pdf(path)
        for sec in sections:
            sec['doc'] = file
        all_sections.extend(sections)

    ranked = rank_sections(all_sections, persona, job)
    output_data = generate_output_json(input_files, persona, job, ranked)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"[✓] Finished processing {os.path.basename(collection_path)}")


def main():
    for folder in sorted(os.listdir(ROOT_INPUT_DIR)):
        if folder.lower().startswith("collection"):
            full_path = os.path.join(ROOT_INPUT_DIR, folder)
            if os.path.isdir(full_path):
                process_collection(full_path)


if __name__ == '__main__':
    main()
