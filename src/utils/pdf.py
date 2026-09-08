"""
Utility functions.

Author: Caolan Rafferty
Date: 2023-07-02
"""

import io
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pdfrw
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, SimpleDocTemplate

from src.utils.types import Frame
from src.utils.util import convert_to_snake_case


def df_to_pdf_inner(
    title: str,
    df: Frame,
    page: int,
    output_dir: str,
    highlight_columns: Optional[List[str]] = None,
    thresholds: Optional[List[float]] = None,
    operators: Optional[List[str]] = None,
    highlight_colour: Optional[str] = None,
) -> str:
    """
    Save a DataFrame as a PDF file with optional highlighting of cells based on specified conditions.

    Parameters:
    title (str): Title of the PDF document.
    df (Frame): The DataFrame to be saved as a PDF.
    page (int): Page number of the file.
    output_dir (str): The path to the output directory.
    highlight_columns (List[str]): List of column names to be highlighted. Defaults to None.
    thresholds (List[float]): List of threshold values for highlighting. Defaults to None.
    operators (List[str]): List of comparison operators ('>' or '<') for highlighting. Defaults to None.
    highlight_colour (str): The colour for highlighting the cells. Defaults to None.
    """
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
                        cell_value = cell.get_text().get_text()
                        if cell_value == "-":
                            continue
                        if operators[i] == ">" and float(cell_value) > thresholds[i]:
                            cell.set_facecolor(highlight_colour)
                        elif operators[i] == "<" and float(cell_value) < thresholds[i]:
                            cell.set_facecolor(highlight_colour)

    ax.set_title(title, fontsize=12, fontweight="bold", y=0.9)

    file = f"{output_dir}/{convert_to_snake_case(title)}_{page}.pdf"
    pp = PdfPages(file)
    pp.savefig(fig, bbox_inches="tight")
    pp.close()
    return file


def df_to_pdf(
    title: str,
    df: Frame,
    output_dir: str,
    highlight_columns: Optional[List[str]] = None,
    thresholds: Optional[List[float]] = None,
    operators: Optional[List[str]] = None,
    highlight_colour: Optional[str] = None,
    max_rows: int = 14,
) -> list[str]:
    """
    Save a DataFrame as a PDF file with optional highlighting of cells based on specified conditions.

    Parameters:
    title (str): Title of the PDF document.
    df (Frame): The DataFrame to be saved as a PDF.
    output_dir (str): The path to the output directory.
    highlight_columns (List[str]): List of column names to be highlighted. Defaults to None.
    thresholds (List[float]): List of threshold values for highlighting. Defaults to None.
    operators (List[str]): List of comparison operators ('>' or '<') for highlighting. Defaults to None.
    highlight_colour (str): The colour for highlighting the cells. Defaults to None.
    """
    if len(df) <= max_rows:
        file = df_to_pdf_inner(
            title,
            df,
            1,
            output_dir,
            highlight_columns,
            thresholds,
            operators,
            highlight_colour,
        )
        return [file]

    dfs = np.array_split(df, np.ceil(len(df) / max_rows))
    file_list = []
    for i, sub_df in enumerate(dfs):
        file = df_to_pdf_inner(
            title,
            sub_df,
            i + 1,
            output_dir,
            highlight_columns,
            thresholds,
            operators,
            highlight_colour,
        )
        file_list.append(file)

    return file_list


def add_page_numbers(pdf_output: pdfrw.PdfWriter) -> None:
    """
    Stamp "Page X of Y" onto the bottom-right corner of every page already added to a PdfWriter.

    The bottom-right corner is used (rather than bottom-centre) because some pages - e.g. the
    matplotlib chart pages - have rotated x-axis tick labels that run close to the bottom edge
    and would otherwise overlap a centred page number.

    Each page keeps its own MediaBox, so the overlay is sized and positioned per-page rather
    than assuming a single fixed page size across the whole report.

    Parameters:
    pdf_output (pdfrw.PdfWriter): The writer holding the pages to number, in place.
    """
    pages = pdf_output.pagearray
    total_pages = len(pages)

    for i, page in enumerate(pages, start=1):
        media_box = [float(x) for x in page.MediaBox]
        width, height = media_box[2] - media_box[0], media_box[3] - media_box[1]

        packet = io.BytesIO()
        pdf_canvas = canvas.Canvas(packet, pagesize=(width, height))
        pdf_canvas.setFont("Helvetica", 8)
        pdf_canvas.drawRightString(width - 20, 12, f"Page {i} of {total_pages}")
        pdf_canvas.save()
        packet.seek(0)

        overlay = pdfrw.PdfReader(packet).pages[0]
        pdfrw.PageMerge(page).add(overlay).render()


def merge_pdfs(input_files: List[str], output_file: str) -> None:
    """
    Merge multiple PDF files into a single PDF file, with page numbers stamped on every page.

    Parameters:
    input_files (List[str]): A list of input file paths (strings) representing the PDF files to be merged.
    output_file (str): The output file path (string) where the merged PDF file will be saved.
    """
    pdf_output = pdfrw.PdfWriter()
    for file_name in input_files:
        pdf_input = pdfrw.PdfReader(file_name)
        for page in pdf_input.pages:
            pdf_output.addpage(page)
        os.remove(file_name)
    add_page_numbers(pdf_output)
    pdf_output.write(output_file)


def save_paragraphs_to_pdf(
    title: str, headings: List[str], paragraphs: List[str], output_dir: str
) -> str:
    """
    Save paragraphs to a PDF file with specified title, headings, and output file.

    Parameters:
    title (str): The main title of the document.
    headings (List[str]): A list of heading strings.
    paragraphs (List[str]): A list of paragraph strings.
    output_dir (str): The path to the output directory.

    Returns:
    str: The file path of the created PDF.
    """
    file = f"{output_dir}/{convert_to_snake_case(title)}.pdf"
    doc = SimpleDocTemplate(file, pagesize=letter)
    styles = getSampleStyleSheet()
    main_title_style = ParagraphStyle(
        name="MainTitle", parent=styles["Heading1"], alignment=1, spaceAfter=24
    )

    title_style = styles["Heading3"]
    title_style.alignment = 0
    paragraph_style = styles["Normal"]
    paragraph_style.fontSize = 10
    paragraph_style.spaceAfter = 12
    elements = []
    main_title_element = Paragraph(title, main_title_style)
    elements.append(main_title_element)

    for title, paragraph in zip(headings, paragraphs):
        title_element = Paragraph(title, title_style)
        elements.append(title_element)
        paragraph_element = Paragraph(paragraph, paragraph_style)
        elements.append(paragraph_element)

    doc.build(elements)
    return file
