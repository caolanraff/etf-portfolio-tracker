"""
Utility functions.

Author: Caolan Rafferty
Date: 2023-07-02
"""

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdfrw
from matplotlib.backends.backend_pdf import PdfPages

saved_pdf_files = []


def df_to_pdf(
    title: str,
    df: pd.DataFrame,
    file: str,
    highlight_columns: Optional[List[str]] = None,
    thresholds: Optional[List[float]] = None,
    operators: Optional[List[str]] = None,
    highlight_colour: Optional[str] = None,
) -> None:
    """
    Save a DataFrame as a PDF file with optional highlighting of cells based on specified conditions.

    Args:
        title: Title of the PDF document.
        df: The DataFrame to be saved as a PDF.
        file: The path and filename of the PDF file to be created.
        highlight_columns: List of column names to be highlighted. Defaults to None.
        thresholds: List of threshold values for highlighting. Defaults to None.
        operators: List of comparison operators ('>' or '<') for highlighting. Defaults to None.
        highlight_colour: The colour for highlighting the cells. Defaults to None.
    """
    max_rows = 14
    if len(df) > max_rows:
        dfs = np.array_split(df, np.ceil(len(df) / max_rows))
        for i, sub_df in enumerate(dfs):
            new_file = f"{file[:-4]}_{i}.pdf"
            df_to_pdf(
                title,
                sub_df,
                new_file,
                highlight_columns,
                thresholds,
                operators,
                highlight_colour,
            )
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold", ha="left")
        else:
            cell.set_text_props(ha="left")
            if highlight_columns and thresholds and operators and highlight_colour:
                for i, col_name in enumerate(highlight_columns):
                    try:
                        col_index = df.columns.get_loc(col_name)
                    except KeyError:
                        raise ValueError(f"Column '{col_name}' not found in dataframe")
                    if col == col_index:
                        cell_value = float(cell.get_text().get_text())
                        if operators[i] == ">" and cell_value > thresholds[i]:
                            cell.set_facecolor(highlight_colour)
                        elif operators[i] == "<" and cell_value < thresholds[i]:
                            cell.set_facecolor(highlight_colour)

    ax.set_title(title, fontsize=12, fontweight="bold", y=0.9)
    pp = PdfPages(file)
    pp.savefig(fig, bbox_inches="tight")
    pp.close()
    saved_pdf_files.append(file)


def merge_pdfs(input_files: List[str], output_file: str) -> None:
    """
    Merge multiple PDF files into a single PDF file.

    Args:
        input_files: A list of input file paths (strings) representing the PDF files to be merged.
        output_file: The output file path (string) where the merged PDF file will be saved.
    """
    pdf_output = pdfrw.PdfWriter()
    for file_name in input_files:
        pdf_input = pdfrw.PdfReader(file_name)
        for page in pdf_input.pages:
            pdf_output.addpage(page)
        os.remove(file_name)
    pdf_output.write(output_file)
