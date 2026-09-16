import json
import re
from pathlib import Path

catalog = json.loads(Path(r"D:\LMC\papers\pdf_first_pages_catalog.json").read_text(encoding="utf-8"))
pattern = re.compile(
    r"self-supervised|zero-shot|low-rank|repeated acquisition|unsupervised|internal learning|"
    r"physics-guided|super-resolution magnetic resonance spectroscopic|deuterium",
    re.IGNORECASE,
)
for item in catalog["papers"]:
    haystack = item["filename"] + " " + item.get("first_two_pages", "")[:4000]
    if pattern.search(haystack):
        print(f"{item['filename']}\tpages={item.get('page_count')}\tmeta={item.get('metadata', {}).get('title', '')}")
