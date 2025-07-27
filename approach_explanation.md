**Approach Explanation: Adobe Hackathon 1B**

**Challenge Objective:**
The goal of Round 1B is to create a system that acts as a personalized document analyst. Given a persona, their job-to-be-done, and a collection of PDF documents, the system should extract the most relevant sections and sub-sections from those documents and return a structured output.

---

**Methodology:**

1. **PDF Parsing and Section Extraction**  
   We use **PyMuPDF** to parse PDFs and extract section-level text. It allows us to retrieve both the content and metadata (e.g., page numbers, font sizes). We apply simple heuristics based on font size and line length to identify candidate section titles.

2. **Semantic Embedding Model**  
   To understand the intent of the persona and match it semantically with section texts, we use the **`all-MiniLM-L6-v2`** model from the `sentence-transformers` library. This lightweight transformer model (only ~80MB) is well-suited for semantic similarity tasks and performs efficiently on CPU.

3. **Persona and Job-to-be-Done Encoding**  
   The persona and task description are combined into a query and embedded into a semantic vector. Each candidate section text is also encoded using the same model. We calculate the **cosine similarity** between the query and section embeddings to assess relevance.

4. **Ranking and Selection**  
   Sections are sorted in descending order of relevance score. We select the top-k sections (default = 5) as the most important ones for the given persona and task. These are used in the final JSON output with their document name, page number, and text.

5. **Sub-Section Analysis**  
   The most relevant section texts are reused as sub-sections for refined analysis. These can optionally be enhanced by sentence-level extraction or summarization techniques.

6. **Structured Output Generation**  
   The final output JSON includes metadata (persona, job, timestamp), a ranked list of extracted sections, and a corresponding sub-section analysis.

---

**Why MiniLM?**
- Fast and CPU-efficient
- High performance in semantic similarity tasks
- ~80MB size complies with Docker and offline execution constraints

---

**Constraints Addressed:**
- ✅ Runs completely offline
- ✅ Compatible with CPU-only environments
- ✅ Model size well below 1GB
- ✅ Execution time < 60 seconds for 3-5 PDFs

---

**Potential Enhancements:**
- Sentence-level analysis for finer granularity
- Named Entity Recognition to highlight key concepts
- Optional multilingual handling using XLM-RoBERTa or LaBSE (size permitting)

---

**Conclusion:**
This modular and scalable approach ensures semantic relevance, generalization across domains, and efficient performance, making it ideal for the Adobe Hackathon Round 1B requirements.

