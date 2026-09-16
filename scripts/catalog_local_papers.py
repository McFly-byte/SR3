#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers-root", required=True)
    parser.add_argument("--pymupdf-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.pymupdf_dir).resolve()))
    import pymupdf

    root = Path(args.papers_root).resolve()
    rows = []
    for path in sorted(root.glob("*.pdf")):
        record = {"path": str(path), "filename": path.name, "file_size": path.stat().st_size}
        try:
            data = path.read_bytes()
            record["file_sha256"] = sha256_bytes(data)
            with pymupdf.open(path) as doc:
                text_parts = [page.get_text("text", sort=True) for page in doc]
                full_text = "\n\n".join(text_parts)
                first_pages = "\n\n".join(text_parts[:2])
                record.update({
                    "page_count": doc.page_count,
                    "metadata": doc.metadata,
                    "text_chars": len(full_text),
                    "text_sha256": sha256_bytes(full_text.encode("utf-8")),
                    "first_two_pages": first_pages[:20000],
                    "text_status": "PASS" if len(full_text.strip()) >= 500 else "TOO_LITTLE_TEXT",
                })
        except Exception as exc:
            record.update({"parse_status": "ERROR", "error": repr(exc)})
        else:
            record["parse_status"] = "PASS"
        rows.append(record)
    output = Path(args.output).resolve()
    output.write_text(json.dumps({"paper_count": len(rows), "papers": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    duplicate_groups = {}
    for row in rows:
        duplicate_groups.setdefault(row.get("text_sha256", row.get("file_sha256")), []).append(row["filename"])
    duplicates = [names for names in duplicate_groups.values() if len(names) > 1]
    print(json.dumps({"paper_count": len(rows), "parse_pass": sum(r.get("parse_status") == "PASS" for r in rows), "duplicate_groups": duplicates, "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
