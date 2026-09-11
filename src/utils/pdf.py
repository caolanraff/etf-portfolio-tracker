"""
Utility functions.

Author: Caolan Rafferty
Date: 2023-07-02
"""

import io
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pdfrw
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer

from src.cli.const import ACCENT_COLOR, ACCENT_COLOR_TINT
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
    table.scale(1, 1.4)

    # Cell borders are drawn manually below (as horizontal separator lines) instead of via
    # matplotlib's per-cell edges: a Cell's fill only renders when all 4 edges are visible, so
    # `visible_edges` can't give a horizontal-only look without also breaking the facecolor.
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0)

        if row == 0:
            cell.set_text_props(fontweight="bold", ha="left", color="white")
            cell.set_facecolor(ACCENT_COLOR)
        else:
            cell.set_text_props(ha="left")
            cell.set_facecolor(ACCENT_COLOR_TINT if row % 2 == 0 else "white")
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

    # A figure-level suptitle (rather than an axes-level title) stays clear of the table even
    # when it grows taller than the axes for large row counts.
    fig.suptitle(title, fontsize=12, fontweight="bold", color=ACCENT_COLOR)

    # Cell positions aren't computed until the figure is drawn, so this must run before the
    # separator lines below can be positioned.
    fig.canvas.draw()

    n_cols = len(df.columns)
    n_rows = len(df) + 1
    x_left = table[(0, 0)].get_x()
    x_right = table[(0, n_cols - 1)].get_x() + table[(0, n_cols - 1)].get_width()

    for row in range(n_rows + 1):
        if row == 0:
            y = table[(0, 0)].get_y() + table[(0, 0)].get_height()
        else:
            y = table[(row - 1, 0)].get_y()
        is_header_separator = row == 1
        ax.plot(
            [x_left, x_right],
            [y, y],
            transform=ax.transAxes,
            color=ACCENT_COLOR if is_header_separator else "#CCCCCC",
            linewidth=1.4 if is_header_separator else 0.6,
            clip_on=False,
            zorder=10,
        )

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


def create_toc_page(entries: List[Tuple[str, int]], output_dir: str) -> str:
    """
    Render a table of contents PDF page, with dot leaders connecting each section to its page.

    Rendered as a monospaced block rather than a table, so the dot leaders line up perfectly
    across rows regardless of how long each section title is.

    Parameters:
    entries (List[Tuple[str, int]]): (section title, starting page number) pairs, in report order.
    output_dir (str): The path to the output directory.

    Returns:
    str: The file path of the created PDF.
    """
    file = f"{output_dir}/table_of_contents.pdf"
    doc = SimpleDocTemplate(file, pagesize=letter, topMargin=72, bottomMargin=72)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TOCTitle",
        parent=styles["Title"],
        fontSize=22,
        spaceAfter=28,
        textColor=colors.HexColor(ACCENT_COLOR),
    )
    entry_style = ParagraphStyle(
        "TOCEntry", fontName="Courier", fontSize=11, leading=24
    )

    # Chosen so a title + dot leader + page number fits within a letter page's usable width
    # (6.5in) at 11pt Courier (~6.6pt per character).
    line_width = 68

    lines = []
    for title, page in entries:
        label = str(page)
        dots = "." * max(3, line_width - len(title) - len(label) - 1)
        lines.append(f"{title} {dots} {label}")

    elements = [
        Paragraph("Table of Contents", title_style),
        Spacer(1, 6),
        Preformatted("\n".join(lines), entry_style),
    ]
    doc.build(elements)
    return file


def merge_pdfs_with_toc(
    title_page: str, sections: List[Tuple[str, List[str]]], output_file: str
) -> None:
    """
    Build a table of contents for a list of report sections, then merge everything into one PDF.

    The table of contents is always exactly one page, so each section's starting page number can
    be computed up front from its page count, without needing a second pass once the table of
    contents itself is generated.

    Parameters:
    title_page (str): File path of the title page PDF.
    sections (List[Tuple[str, List[str]]]): (section title, file paths) pairs, in report order.
    output_file (str): The output file path (string) where the merged PDF file will be saved.
    """
    output_dir = os.path.dirname(output_file)
    page_counts = [
        sum(len(pdfrw.PdfReader(f).pages) for f in files) for _, files in sections
    ]

    page = 3  # after the title page (1) and the table of contents (1)
    entries = []
    for (title, _), count in zip(sections, page_counts):
        entries.append((title, page))
        page += count

    toc_file = create_toc_page(entries, output_dir)

    all_files = [title_page, toc_file] + [f for _, files in sections for f in files]
    merge_pdfs(all_files, output_file)


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
        name="MainTitle",
        parent=styles["Heading1"],
        alignment=1,
        spaceAfter=24,
        textColor=colors.HexColor(ACCENT_COLOR),
    )

    title_style = styles["Heading3"]
    title_style.alignment = 0
    title_style.textColor = colors.HexColor(ACCENT_COLOR)
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
