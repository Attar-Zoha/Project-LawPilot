import os
import fitz  # PyMuPDF
import json

# ---------------- CONFIG ----------------
PDF_FOLDER = "data/indian_judgements/sc_data/english"
OUTPUT_FILE = "case_domains.json"

# Define keywords for mapping to domains
DOMAIN_KEYWORDS = {
    "Consumer Law / E-Commerce": ["consumer", "e-commerce", "trade", "goods", "defective", "sale", "retail"],
    "Corporate / Company Law": ["company", "corporate", "sebi", "shareholder", "directors", "company act", "merger"],
    "Tax / GST / Income": ["tax", "income tax", "gst", "indirect tax", "taxation"],
    "Labour / Employment Law": ["labour", "employee", "employment", "wages", "industrial"],
    "Environmental Law": ["environment", "pollution", "forest", "wildlife", "ecology"],
    "IT / Cyber Law / Privacy": ["privacy", "data protection", "it act", "cyber", "hacking", "information technology"],
    "Constitutional / Fundamental Rights": ["constitution", "fundamental rights", "article", "rights", "amendment"],
    "Criminal Law / IPC": ["criminal", "ipc", "section", "offence", "penal", "trial", "sentence"],
    "Other": []
}

# ---------------- HELPER FUNCTIONS ----------------
def detect_domain(text):
    text_lower = text.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return domain
    return "Other"

# ---------------- MAIN PROCESS ----------------
case_list = []

for file in os.listdir(PDF_FOLDER):
    if file.endswith(".pdf"):
        path = os.path.join(PDF_FOLDER, file)
        try:
            doc = fitz.open(path)
            first_page_text = doc[0].get_text()
            doc.close()

            # crude extraction: first non-empty line as case name
            lines = [line.strip() for line in first_page_text.split("\n") if line.strip()]
            case_name = lines[0] if lines else file
            domain = detect_domain(first_page_text)

            case_list.append({
                "file": file,
                "case_name": case_name,
                "domain": domain
            })
        except Exception as e:
            print(f"Error reading {file}: {e}")

# ---------------- SAVE RESULTS ----------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(case_list, f, indent=2, ensure_ascii=False)

print(f"✅ Domain extraction complete. Results saved to {OUTPUT_FILE}")
