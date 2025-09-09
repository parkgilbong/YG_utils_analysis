import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_gene_ontology_bar(go_data_path: Path, output_path: Path, top_n: int = 15, ax=None):
    """
    Generates a bar plot of the top Gene Ontology (GO) terms based on adjusted p-values.

    Parameters
    ----------
    go_data_path : Path
        Path to the CSV file containing GO enrichment results. The file must include at least
        'p.adjust' (adjusted p-value) and 'Description' (GO term description) columns.
    output_path : Path
        Path to save the generated bar plot image (e.g., PNG file).
    top_n : int, optional
        Number of top GO terms to display in the plot (default is 15).
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes object to plot on. If None, a new figure and axes are created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot.
    """
    df = pd.read_csv(go_data_path)
    df = df.sort_values('p.adjust', ascending=True).head(top_n)
    if df.empty:
        print(f"[warn] plot_gene_ontology_bar skipped: No significant GO terms found in {go_data_path}")
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No significant GO terms to plot', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title('Top GO Terms')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        return ax
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x='p.adjust', y='Description', data=df, palette='viridis', ax=ax)
    ax.set_xlabel('Adjusted P-value')
    ax.set_ylabel('GO Term')
    ax.set_title('Top GO Terms')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return ax

def plot_heatmap(pivot_df, title, output_path, pdf=None, ax=None):
    """
    Plots a heatmap from a given DataFrame and saves it to the specified output path.

    Parameters
    ----------
    pivot_df : pandas.DataFrame
        The data to be visualized as a heatmap. Typically, this should be a pivoted DataFrame.
    title : str
        The title for the heatmap plot. Used to determine the colormap and center value.
    output_path : str
        The file path where the heatmap image will be saved.
    pdf : matplotlib.backends.backend_pdf.PdfPages, optional
        If provided, the plot will also be saved to the given PDF object.
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes object to plot on. If None, a new figure and axes are created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        pivot_df,
        annot=True,
        cmap='viridis' if 'p_val' in title else 'coolwarm',
        center=0 if 'avg' in title else 0.05,
        ax=ax
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    if pdf:
        pdf.savefig()
    plt.close()
    return ax

def barplot(enr_res2d, out_png, top_terms=20, title="GO Enrichment (Top)", ax=None):
    """
    Create a horizontal bar plot for GO enrichment results.

    Args:
        enr_res2d (pandas.DataFrame): Enrichment results DataFrame.
        out_png (str): Path to save the plot image.
        top_terms (int, optional): Number of top terms to plot.
        title (str, optional): Plot title.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates new axes.

    Returns:
        matplotlib.axes.Axes: The axes object with the plot.
    """
    df = enr_res2d.sort_values("Adjusted P-value").head(top_terms).copy()
    if df.empty:
        print("[warn] barplot skipped: empty results"); return ax
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(df["Term"][::-1], df["Combined Score"][::-1])
    ax.set_xlabel("Combined Score")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] wrote {out_png}")
    return ax

def dotplot(enr_res2d, out_png, top_terms=20, title="GO Enrichment (Top)", ax=None):
    """
    Create a dot plot for GO enrichment results, showing overlap ratios.

    Args:
        enr_res2d (pandas.DataFrame): Enrichment results DataFrame.
        out_png (str): Path to save the plot image.
        top_terms (int, optional): Number of top terms to plot.
        title (str, optional): Plot title.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates new axes.

    Returns:
        matplotlib.axes.Axes: The axes object with the plot.
    """
    df = enr_res2d.sort_values("Adjusted P-value").head(top_terms).copy()
    if df.empty:
        print("[warn] dotplot skipped: empty results"); return ax
    ratios = []
    for s in df["Overlap"]:
        try:
            num, den = s.split("/"); ratios.append(float(num)/float(den))
        except Exception:
            ratios.append(0.0)
    df["overlap_ratio"] = ratios
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(df["Adjusted P-value"], range(len(df)), s=(df["overlap_ratio"]*400)+10)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["Term"])
    ax.invert_yaxis()
    ax.set_xlabel("Adjusted P-value")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] wrote {out_png}")
    return ax

def volcano_plot(df, log2fc_col, padj_col, out_png, log2fc_cutoff, adj_p_cutoff, ax=None):
    """
    Create a volcano plot for differential expression results.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        log2fc_col (str): Column name for log2 fold change.
        padj_col (str): Column name for adjusted p-value.
        out_png (str): Path to save the plot image.
        log2fc_cutoff (float): Log2 fold change cutoff for significance.
        adj_p_cutoff (float): Adjusted p-value cutoff for significance.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates new axes.

    Returns:
        matplotlib.axes.Axes: The axes object with the plot.
    """
    import numpy as np
    work = df.dropna(subset=[log2fc_col, padj_col]).copy()
    work[log2fc_col] = pd.to_numeric(work[log2fc_col], errors="coerce")
    work[padj_col] = pd.to_numeric(work[padj_col], errors="coerce")
    work = work.dropna(subset=[log2fc_col, padj_col])
    work["neglog10_padj"] = -np.log10(work[padj_col].clip(lower=1e-300))
    if ax is None:
        fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(work[log2fc_col], work["neglog10_padj"])
    y_thr = -np.log10(adj_p_cutoff if adj_p_cutoff>0 else 1e-300)
    ax.axhline(y=y_thr)
    ax.axvline(x=log2fc_cutoff)
    ax.axvline(x=-abs(log2fc_cutoff))
    ax.set_xlabel("log2FC")
    ax.set_ylabel("-log10(adj p-value)")
    ax.set_title("Volcano plot")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] wrote {out_png}")
    return ax