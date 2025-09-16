import pdfplumber
import re
import json
import spacy
from dateutil import parser
from transformers import pipeline
from tqdm import tqdm

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Load summarizer
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Extract text from PDF
def extract_text(pdf_path):
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in tqdm(pdf.pages, desc="Extracting PDF"):
            all_text += page.extract_text() + "\n"
    return all_text

# Extract complex Tender Reference Number
def extract_reference_number(text):
    match = re.search(
        r"(tender\s*ref(?:erence)?\s*number|tender\s*no)[\s:\-]*([^\n\rdt]+)",
        text, re.IGNORECASE
    )
    if match:
        ref_number = match.group(2).strip()
        ref_number = re.split(r"\bdt\b|\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", ref_number, flags=re.IGNORECASE)[0].strip()
        return ref_number
    return ""

# Parse dates with pattern matching
def parse_date(patterns, text):
    for pattern in patterns.split("|"):
        regex = rf"{pattern}\s+(\d{{1,2}}[-\s]?[A-Za-z]{{3,9}}[-\s]?\d{{2,4}})"
        match = re.search(regex, text, re.IGNORECASE)
        if match:
            try:
                return parser.parse(match.group(1), fuzzy=True).strftime("%Y-%m-%d")
            except Exception as e:
                print(f"[ERROR] Failed to parse date: {match.group(1)} — {e}")
    return None

# Extract key fields
def extract_fields(text):
    result = {}

    # Title
    title_match = re.search(r"(tender\s*title|title)\s*[:\-]?\s*(.*)", text, re.IGNORECASE)
    result["Title"] = title_match.group(2).strip() if title_match else ""

    # Reference Number
    result["Reference Number"] = extract_reference_number(text)

    # Organization Name
    org_match = re.search(r"(inviting authority|organization|company)\s*[:\-]?\s*(.*)", text, re.IGNORECASE)
    result["Organization Name"] = org_match.group(2).split('\n')[0].strip() if org_match else ""

    # Dates
    result["Bid Submission Start Date"] = parse_date("bid submission start date|start date", text)
    result["Bid Submission End Date"] = parse_date("submission end date|last date of submission|closing date", text)
    result["Bid Opening Date"] = parse_date("bid opening date|opening date|technical bid opening", text)

    # Description
    result["Description"] = " ".join(text.split()[:1000])

    return result

# Summarize text
def get_short_summary(text):
    try:
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        summary = ""
        for chunk in chunks[:2]:  # limit to 2 chunks for speed
            summary += summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text'] + " "
        return summary.strip()
    except Exception as e:
        return ""

# Main pipeline
def process_tender(pdf_path):
    raw_text = extract_text(pdf_path)
    fields = extract_fields(raw_text)
    fields["Short Summary"] = get_short_summary(raw_text)
    return fields

# Run the program
if __name__ == "__main__":
    pdf_file = "ten.pdf"
    data = process_tender(pdf_file)
    with open("tender_summary.json", "w") as f:
        json.dump(data, f, indent=4)
    print("✅ JSON summary saved to 'tender_summary.json'")
    
    
   