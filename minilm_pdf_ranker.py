import os
import json
import fitz  # PyMuPDF
import datetime
from sentence_transformers import SentenceTransformer, util

INPUT_DIR = "./input"
OUTPUT_DIR = "./output"

# Load model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


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
            "refined_text": section['text']  # Could add summarization here
        })

    return output


def main():
    persona = "PhD Researcher in Computational Biology"
    job = "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"

    input_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]
    all_sections = []

    for file in input_files:
        path = os.path.join(INPUT_DIR, file)
        sections = extract_sections_from_pdf(path)
        for sec in sections:
            sec['doc'] = file
        all_sections.extend(sections)

    ranked = rank_sections(all_sections, persona, job)

    output_json = generate_output_json(input_files, persona, job, ranked)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "output.json"), "w", encoding='utf-8') as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
