import json
import os
import re

import fitz  # PyMuPDF
from src.config import RAW_DIR, CHUNKS_PATH as OUT_PATH
from src.indexing.chunk import clean_text_for_embedding, chunk_text_semantically


def extract_clean_markdown_from_pdf(path: str) -> str:
    """
    Extracts PDF text block-by-block, repairing missing spaces between words using
    bounding-box coordinates (x0/x1 gap detection) and converting headings to Markdown.
    """
    doc = fitz.open(path)
    font_sizes = []

    # Pass 1: Gather font size distribution to identify body text size
    for page in doc:
        blocks = page.get_text("dict").get("blocks", [])
        for block in blocks:
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text", "").strip():
                            font_sizes.append(round(span["size"], 1))

    body_size = max(set(font_sizes), key=font_sizes.count) if font_sizes else 10.0

    # Pass 2: Reconstruct text with physical bounding-box gap detection
    markdown_lines = []
    for page in doc:
        blocks = page.get_text("dict").get("blocks", [])
        for block in blocks:
            if block.get("type") == 0:
                for line in block.get("lines", []):
                    line_parts = []
                    line_max_size = 0.0
                    is_bold = False
                    spans = line.get("spans", [])

                    for i, span in enumerate(spans):
                        text = span.get("text", "")
                        if not text:
                            continue

                        size = span.get("size", body_size)
                        if size > line_max_size:
                            line_max_size = size

                        font_name = span.get("font", "").lower()
                        if span.get("flags", 0) & 2 or "bold" in font_name:
                            is_bold = True

                        # Fix missing spaces: Calculate horizontal gap between current and previous span
                        if i > 0:
                            prev_span = spans[i - 1]
                            gap = span["bbox"][0] - prev_span["bbox"][2]  # start_x - end_x
                            if gap > 1.0 and not line_parts[-1].endswith(" ") and not text.startswith(" "):
                                line_parts.append(" ")

                        line_parts.append(text)

                    full_line = "".join(line_parts).strip()
                    if not full_line:
                        continue

                    # Assign Markdown headers based on relative font size
                    if line_max_size >= body_size * 1.4:
                        markdown_lines.append(f"# {full_line}")
                    elif line_max_size >= body_size * 1.2:
                        markdown_lines.append(f"## {full_line}")
                    elif line_max_size >= body_size * 1.1 and is_bold:
                        markdown_lines.append(f"### {full_line}")
                    elif full_line.startswith("#"):
                        markdown_lines.append(full_line)
                    else:
                        markdown_lines.append(full_line)

                markdown_lines.append("")  # Block separator

    doc.close()
    return "\n".join(markdown_lines)


def parse_pdf_to_sections(path: str) -> list[dict]:
    """Parse PDF into section blocks using Markdown header hierarchy."""
    md_text = extract_clean_markdown_from_pdf(path)

    sections = []
    header_stack = {}  # Tracks level (int) -> heading title (str)
    buffer = []

    current_heading = "Overview"
    current_path = "Overview"

    for line in md_text.split("\n"):
        line_str = line.strip()

        # Parse Markdown headers (# H1, ## H2, ### H3, etc.)
        if line_str.startswith("#"):
            match = re.match(r"^(#+)\s*(.*)", line_str)
            if match:
                hashes, title = match.groups()
                level = len(hashes)
                title = clean_text_for_embedding(title.strip())

                if buffer:
                    sections.append({
                        "heading": current_heading,
                        "section_path": current_path,
                        "body": "\n".join(buffer)
                    })
                    buffer = []

                # Update header tree level
                header_stack = {lvl: h for lvl, h in header_stack.items() if lvl < level}
                header_stack[level] = title

                current_heading = title
                current_path = " > ".join([header_stack[lvl] for lvl in sorted(header_stack.keys())])
        else:
            if line_str:
                buffer.append(line_str)

    if buffer:
        sections.append({
            "heading": current_heading,
            "section_path": current_path,
            "body": "\n".join(buffer)
        })

    return sections


def parse_pdf(path: str) -> list[dict]:
    """Extract, clean, and semantically chunk any PDF file."""
    filename = os.path.basename(path)
    doc_stem = os.path.splitext(filename)[0]

    sections = parse_pdf_to_sections(path)
    title = (
        sections[0]["heading"]
        if sections and sections[0]["heading"] != "Overview"
        else doc_stem.replace("_", " ").replace("-", " ").title()
    )

    records = []
    for section in sections:
        clean_body = clean_text_for_embedding(section["body"])
        if not clean_body:
            continue

        for piece in chunk_text_semantically(clean_body):
            if len(piece.split()) < 20:  # Skip trivial fragments
                continue

            records.append({
                "topic": title,
                "section": section["heading"],
                "section_path": section["section_path"],
                "heading": section["heading"],
                "text": piece,
                "url": filename,
                "source": doc_stem,
                "corpus": "pdf_documents",
            })

    return records


def load_jsonl(path: str) -> list[dict]:
    """Load pre-formatted JSONL datasets."""
    records = []
    corpus_name = os.path.splitext(os.path.basename(path))[0]

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            row.pop("id", None)
            row["corpus"] = corpus_name

            if "text" in row:
                row["text"] = clean_text_for_embedding(row["text"])

            if "section_path" not in row:
                row["section_path"] = row.get("section") or row.get("heading") or "Overview"

            records.append(row)

    return records


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    records = []

    if not os.path.exists(RAW_DIR):
        print(f"Directory '{RAW_DIR}' not found.")
        return

    for name in sorted(os.listdir(RAW_DIR)):
        file_path = os.path.join(RAW_DIR, name)

        if name.lower().endswith(".jsonl"):
            rows = load_jsonl(file_path)
            records.extend(rows)
            print(f"{name}: {len(rows)} chunks (JSONL)")

        elif name.lower().endswith(".pdf"):
            rows = parse_pdf(file_path)
            records.extend(rows)
            print(f"{name}: {len(rows)} chunks (PDF)")

    # Global unique ID assignment
    for n, record in enumerate(records):
        record["id"] = f"{record['corpus']}-{n:04d}"

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    total_words = sum(len(r["text"].split()) for r in records)
    print(f"\nTotal: {len(records)} chunks, ~{total_words:,} words -> {OUT_PATH}")


if __name__ == "__main__":
    main()