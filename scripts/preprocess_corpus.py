# scripts/preprocess_corpus.py
import zipfile
import json
from pathlib import Path
import fitz    # PyMuPDF
import argparse
import re

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def extract_metadata_from_zip(zip_path, out_dir):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    with zipfile.ZipFile(zip_path, 'r') as z:
        for name in z.namelist():
            if not name.endswith('.json'):
                continue
            data = json.loads(z.read(name).decode('utf-8', errors='ignore'))
            # Each metadata contains at least 'path'
            path = data.get('path')
            if not path:
                continue
            # Write the metadata to file: out_dir/{path}.metadata.json
            with open(out_dir / f"{path}.metadata.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

def extract_pdfs_and_text(english_zip_path, regional_zip_path, metadata_dir, out_docs_dir):
    ensure_dir(out_docs_dir)
    # Map from PDF filename inside zip -> bytes
    def process_zip(zip_path):
        mapping = {}
        if not zip_path.exists():
            return mapping
        with zipfile.ZipFile(zip_path, 'r') as z:
            for name in z.namelist():
                if name.endswith('.pdf'):
                    mapping[Path(name).name] = z.read(name)
        return mapping

    english_map = process_zip(english_zip_path)
    regional_map = process_zip(regional_zip_path)

    # For each metadata file, find pdf content and extract text
    for meta_file in sorted(Path(metadata_dir).glob("*.metadata.json")):
        meta = json.load(open(meta_file, 'r', encoding='utf-8'))
        path = meta.get("path")
        if not path:
            continue

        # Try english filename: path_EN.pdf
        pdf_name_en = f"{path}_EN.pdf"
        pdf_name_reg = f"{path}.pdf"   # sometimes path.pdf or path_<lang>.pdf patterns
        pdf_bytes = None
        if pdf_name_en in english_map:
            pdf_bytes = english_map[pdf_name_en]
        elif pdf_name_reg in english_map:
            pdf_bytes = english_map[pdf_name_reg]
        else:
            # try any regional map variants
            if pdf_name_en in regional_map:
                pdf_bytes = regional_map[pdf_name_en]
            else:
                # fallback: look for keys that start with path
                for k in english_map:
                    if k.startswith(path):
                        pdf_bytes = english_map[k]; break
                if not pdf_bytes:
                    for k in regional_map:
                        if k.startswith(path):
                            pdf_bytes = regional_map[k]; break

        text = ""
        if pdf_bytes:
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                pages = []
                for p in doc:
                    pages.append(p.get_text().strip())
                text = "\n\n".join(pages).strip()
            except Exception as e:
                print(f"Error extracting {path}: {e}")

        # Minimal cleanup: remove multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text or "")

        out_json = {
            "path": path,
            "metadata": meta,
            "text": text
        }

        with open(Path(out_docs_dir) / f"{path}.json", "w", encoding="utf-8") as f:
            json.dump(out_json, f, ensure_ascii=False, indent=2)

def main(packages_dir, out_dir, years=None):
    packages_dir = Path(packages_dir)
    out_dir = Path(out_dir)
    metadata_dir = out_dir / "metadata_files"
    docs_out = out_dir / "docs"
    ensure_dir(metadata_dir)
    ensure_dir(docs_out)

    # For each year zip set (we assume files named like sc-judgments-YYYY-metadata.zip, -english.zip, -regional.zip)
    years_to_process = years or []
    if not years_to_process:
        # attempt to discover years from filenames
        for z in packages_dir.glob("*.zip"):
            m = re.search(r"sc-judgments-(\d{4})-metadata.zip", z.name)
            if m:
                years_to_process.append(int(m.group(1)))
        years_to_process = sorted(set(years_to_process))

    print("Years to process:", years_to_process)

    for year in years_to_process:
        meta_zip = packages_dir / f"sc-judgments-{year}-metadata.zip"
        english_zip = packages_dir / f"sc-judgments-{year}-english.zip"
        regional_zip = packages_dir / f"sc-judgments-{year}-regional.zip"
        print("Processing:", year)
        if meta_zip.exists():
            extract_metadata_from_zip(meta_zip, metadata_dir)
        else:
            print("Missing metadata zip for", year)
        # Now extract pdf bytes & text
        extract_pdfs_and_text(english_zip, regional_zip, metadata_dir, docs_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--packages-dir", default="data/indian_judgements/packages")
    parser.add_argument("--out-dir", default="data/legal_corpus")
    parser.add_argument("--years", nargs="*", type=int, help="years to process e.g. 2019 2020")
    args = parser.parse_args()
    main(args.packages_dir, args.out_dir, args.years)
