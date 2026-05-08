import pandas as pd
import os

GTF = "data/reference/gencode.gtf"   # ide majd letöltjük
GENES = "config/axis_gene_sets.tsv"

OUT = "results/regulatory/axis_promoters.bed"
os.makedirs("results/regulatory", exist_ok=True)


# load gene list
genes = pd.read_csv(GENES, sep="\t")
gene_set = set(genes["gene"])


# parse GTF
records = []

with open(GTF) as f:
    for line in f:
        if line.startswith("#"):
            continue
        
        fields = line.strip().split("\t")
        if fields[2] != "gene":
            continue
        
        chrom = fields[0]
        start = int(fields[3])
        end = int(fields[4])
        strand = fields[6]
        info = fields[8]

        # gene_name extraction
        if 'gene_name "' not in info:
            continue
        
        gene_name = info.split('gene_name "')[1].split('"')[0]

        if gene_name not in gene_set:
            continue

        if strand == "+":
            tss = start
        else:
            tss = end

        prom_start = max(0, tss - 2000)
        prom_end = tss + 2000

        records.append([chrom, prom_start, prom_end, gene_name])


# save BED
df = pd.DataFrame(records, columns=["chr","start","end","gene"])
df.to_csv(OUT, sep="\t", header=False, index=False)

print("Saved:", OUT)
print("Genes found:", len(df))
