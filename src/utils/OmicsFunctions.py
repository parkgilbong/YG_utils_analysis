import pandas as pd
from collections import defaultdict
import gzip
from typing import Set


def load_genes(path: str, inline_genes=None):
    """
        Load gene names from a file or a provided list, removing duplicates and empty entries.

        Args:
            path (str): Path to the gene list file.
            inline_genes (list, optional): List of gene names provided inline.

        Returns:
            list: Unique gene names.

        Example:
            # Load from file
            genes = load_genes("genes.txt")

            # Load from inline list
            genes = load_genes("", inline_genes=["TP53", "BRCA1", "TP53"])
    """
    if inline_genes:
        genes = [g.strip() for g in inline_genes if str(g).strip()]
    else:
        with open(path, "r") as f:
            genes = [line.strip() for line in f if line.strip()]
    seen = set()
    uniq = []
    for g in genes:
        if g not in seen:
            uniq.append(g); seen.add(g)
    return uniq

def filter_genes_by_thresholds(df, gene_col, padj_col, log2fc_col, adj_p_cutoff, log2fc_cutoff, direction="both"):
    """
    Filter genes based on adjusted p-value and log2 fold change thresholds.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        gene_col (str): Column name for gene identifiers.
        padj_col (str): Column name for adjusted p-values.
        log2fc_col (str): Column name for log2 fold change.
        adj_p_cutoff (float): Adjusted p-value cutoff.
        log2fc_cutoff (float): Log2 fold change cutoff.
        direction (str, optional): "up", "down", or "both" for filtering direction.

    Returns:
        tuple: (filtered DataFrame, upregulated DataFrame, downregulated DataFrame, list of unique gene names)

    Example:
        # Example DataFrame
        df = pd.DataFrame({
            "gene": ["TP53", "BRCA1", "EGFR", "MYC"],
            "padj": [0.01, 0.2, 0.03, 0.04],
            "log2fc": [2.5, -1.2, 0.8, -2.1]
        })

        # Filter for significant genes
        filt, up, down, genes = filter_genes_by_thresholds(
            df,
            gene_col="gene",
            padj_col="padj",
            log2fc_col="log2fc",
            adj_p_cutoff=0.05,
            log2fc_cutoff=1.0,
            direction="both"
        )
    """
    work = df.dropna(subset=[gene_col, padj_col, log2fc_col]).copy()
    work[padj_col] = pd.to_numeric(work[padj_col], errors="coerce")
    work[log2fc_col] = pd.to_numeric(work[log2fc_col], errors="coerce")
    work = work.dropna(subset=[padj_col, log2fc_col])

    sig = work[work[padj_col] <= adj_p_cutoff]
    up = sig[sig[log2fc_col] >= log2fc_cutoff]
    down = sig[sig[log2fc_col] <= -abs(log2fc_cutoff)]
    if direction == "up":
        filt = up
    elif direction == "down":
        filt = down
    else:
        filt = pd.concat([up, down], axis=0).drop_duplicates()

    genes = []
    seen = set()
    for g in filt[gene_col].astype(str).tolist():
        if g not in seen:
            genes.append(g); seen.add(g)
    return filt, up, down, genes

def convert_gene_ids(genes, species: str, id_type: str):
    """
    Convert gene identifiers to gene symbols using mygene, if available.

    Args:
        genes (list): List of gene identifiers.
        species (str): Species name (e.g., "human", "mouse").
        id_type (str): Type of gene ID ("symbol", "entrez", "ensembl", "auto").

    Returns:
        list: Unique gene symbols.

    Example:
        # Convert Entrez IDs to gene symbols for human
        symbols = convert_gene_ids([7157, 672, 1956], species="human", id_type="entrez")
    """
    try:
        import mygene
    except Exception as e:
        print("[warn] mygene not available:", e); return genes
    mg = mygene.MyGeneInfo()
    scopes = {"symbol":"symbol","entrez":"entrezgene","ensembl":"ensembl.gene","auto":"symbol,entrezgene,ensembl.gene"}.get(id_type,"symbol")
    if not genes: return []
    res = mg.querymany(genes, scopes=scopes, fields="symbol", species=species, as_dataframe=False, returnall=False)
    out = []
    for x in res:
        if isinstance(x, dict) and "symbol" in x: out.append(x["symbol"])
        elif isinstance(x, dict) and "query" in x: out.append(str(x["query"]))
    out = [g for g in out if g]
    uniq = []; seen=set()
    for g in out:
        if g not in seen: uniq.append(g); seen.add(g)
    return uniq

def load_symbol_assoc_from_gaf(gaf_path: str, taxon="9606", aspect="all"):
    """
    GAF(2.2) 파일에서 'DB Object Symbol'(유전자 심볼) -> {GO_ID} 매핑을 생성합니다.
    - taxon: "9606"(인간) 또는 ["9606","10090"] 식의 리스트도 허용
    - aspect: "all" | "BP" | "MF" | "CC"  (GAF col 8: P/F/C)
    반환: dict[str, set[str]]
    """
    if isinstance(taxon, (int,)):
        taxon = str(taxon)
    taxon_list = {str(taxon)} if isinstance(taxon, str) else {str(x) for x in taxon}
    aspect = (aspect or "all").upper()

    id2gos = defaultdict(set)
    Opener = gzip.open if gaf_path.endswith(".gz") else open
    with Opener(gaf_path, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line or line.startswith("!"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 15:
                continue
            # GAF columns (0-based):
            # 0:DB, 1:DB Object ID, 2:DB Object Symbol, 4:GO ID, 8:Aspect (F/P/C), 12:Taxon
            symbol = cols[2].strip()
            go_id  = cols[4].strip()
            asp    = cols[8].strip().upper() if len(cols) > 8 else ""
            taxfld = cols[12] if len(cols) > 12 else ""  # e.g. "taxon:9606" or "taxon:9606|taxon:2697049"

            # taxon 필터
            if not any(f"taxon:{t}" in taxfld for t in taxon_list):
                continue
            # aspect 필터
            if aspect != "ALL":
                if asp not in {"P","F","C"}:
                    continue
                if (aspect == "BP" and asp != "P") or (aspect == "MF" and asp != "F") or (aspect == "CC" and asp != "C"):
                    continue

            if symbol and go_id:
                id2gos[symbol].add(go_id)
    return dict(id2gos)

def strip_ensembl_version(x: str) -> str:
    """
    Removes the version suffix from an Ensembl identifier.

    Parameters:
        x (str): An Ensembl identifier, potentially with a version suffix (e.g., 'ENSG00000139618.15').

    Returns:
        str: The Ensembl identifier without the version suffix (e.g., 'ENSG00000139618').
    """
    return str(x).split(".", 1)[0]

def convert_ensembl_to_symbol(genes_ens, mapping_df, ens_col="ensembl_gene_id", sym_col="symbol"):
    """
    Converts a list of Ensembl gene IDs to their corresponding gene symbols using a mapping DataFrame.

    Parameters:
        genes_ens (list): List of Ensembl gene IDs (strings), possibly with version numbers.
        mapping_df (pd.DataFrame): DataFrame containing Ensembl IDs and gene symbols.
        ens_col (str, optional): Column name in mapping_df for Ensembl gene IDs. Default is "ensembl_gene_id".
        sym_col (str, optional): Column name in mapping_df for gene symbols. Default is "symbol".

    Returns:
        list: List of unique gene symbols corresponding to the input Ensembl IDs, preserving input order.
    """
    mdict = dict(zip(mapping_df[ens_col].astype(str), mapping_df[sym_col].astype(str)))
    out, seen = [], set()
    for g in genes_ens:
        key = strip_ensembl_version(g)
        sym = mdict.get(key)
        if sym and sym not in seen:
            out.append(sym); seen.add(sym)
    return out

def read_gene_list(path):
    """
    Read a gene list file (one gene per line) and return as a list of strings.
    """
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]