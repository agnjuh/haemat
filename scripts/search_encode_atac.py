import urllib.parse
import urllib.request
import json
import pandas as pd
from pathlib import Path

OUTDIR = Path("data/regulatory/encode_search")
OUTDIR.mkdir(parents=True, exist_ok=True)

BASE = "https://www.encodeproject.org/search/"

params = {
    "type": "File",
    "assay_title": "ATAC-seq",
    "file_format": "bed",
    "assembly": "GRCh38",
    "status": "released",
    "limit": "all",
    "format": "json",
}

url = BASE + "?" + urllib.parse.urlencode(params)

print("Query:")
print(url)

req = urllib.request.Request(
    url,
    headers={
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0"
    }
)

with urllib.request.urlopen(req) as response:
    data = json.loads(response.read().decode("utf-8"))

rows = []

for f in data.get("@graph", []):
    href = f.get("href", "")
    if not href:
        continue

    output_type = str(f.get("output_type", ""))
    if "peak" not in output_type.lower():
        continue

    biosample = f.get("biosample_ontology", {}).get("term_name", "")
    dataset = f.get("dataset", "")
    accession = f.get("accession", "")
    assembly = f.get("assembly", "")
    file_format = f.get("file_format", "")

    rows.append({
        "accession": accession,
        "biosample": biosample,
        "output_type": output_type,
        "file_format": file_format,
        "assembly": assembly,
        "dataset": dataset,
        "download_url": "https://www.encodeproject.org" + href,
    })

df = pd.DataFrame(rows)

df.to_csv(OUTDIR / "all_encode_atac_peak_files.csv", index=False)

keywords = [
    "PBMC",
    "monocyte",
    "T cell",
    "CD4",
    "CD8",
    "natural killer",
    "NK",
    "megakaryocyte",
    "platelet",
]

if not df.empty:
    mask = df["biosample"].fillna("").str.contains("|".join(keywords), case=False, regex=True)
    filtered = df[mask].copy()
else:
    filtered = pd.DataFrame()

filtered.to_csv(OUTDIR / "filtered_immune_atac_peak_files.csv", index=False)

print("\nAll ATAC peak files:", len(df))
print("Filtered immune-like files:", len(filtered))

if not filtered.empty:
    print(filtered[["accession", "biosample", "output_type", "assembly", "download_url"]].head(30))
else:
    print("No immune-like files found with current keyword filter.")
