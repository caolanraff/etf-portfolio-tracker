from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.platypus import SimpleDocTemplate

from src.utils.pdf import (
    add_page_numbers,
    df_to_pdf,
    merge_pdfs,
    save_paragraphs_to_pdf,
)


def test_df_to_pdf(mocker: Any) -> None:
    mocker.patch.object(PdfPages, "savefig")
    mocker.patch.object(PdfPages, "close")

    df = pd.DataFrame(
        {"A": [1, 2, 3, 4], "B": ["3", "4", "-", "10"], "C": ["3", "4", "-", "-3"]}
    )
    output_dir = "/tmp"

    # single page
    result = df_to_pdf(
        "Test Title", df, output_dir, ["B", "C"], [3.0, 4.0], [">", "<"], "red"
    )
    expected = [f"{output_dir}/test_title_1.pdf"]
    assert result == expected

    # multi page
    result = df_to_pdf(
        "Test Title", df, output_dir, ["B", "C"], [3.0, 4.0], [">", "<"], "red", 3
    )
    expected = [f"{output_dir}/test_title_1.pdf", f"{output_dir}/test_title_2.pdf"]
    assert result == expected

    # column doesn't exist
    with pytest.raises(ValueError):
        df_to_pdf("Test Title", df, output_dir, ["Z"], [3.0], [">"], "red")


def test_merge_pdfs(mocker: Any) -> None:
    input_files = ["file1.pdf", "file2.pdf"]
    output_file = "merged.pdf"

    mock_pdf_writer = mocker.patch("pdfrw.PdfWriter")

    mock_pdf_reader = mocker.patch("pdfrw.PdfReader")
    pdf_reader_instance = mock_pdf_reader.return_value
    pdf_reader_instance.pages = [MagicMock(), MagicMock()]

    mock_os_remove = mocker.patch("os.remove")
    mock_add_page_numbers = mocker.patch("src.utils.pdf.add_page_numbers")

    merge_pdfs(input_files, output_file)

    # Check if PdfWriter was called with the correct output file
    mock_pdf_writer.assert_called_once_with()
    mock_pdf_writer.return_value.write.assert_called_once_with(output_file)

    # Check if PdfReader was called for each input file
    assert mock_pdf_reader.call_count == len(input_files)
    for call in mock_pdf_reader.call_args_list:
        assert call[0][0] in input_files

    # Check if os.remove was called for each input file
    assert mock_os_remove.call_count == len(input_files)
    for call in mock_os_remove.call_args_list:
        assert call[0][0] in input_files

    # Check page numbers were stamped onto the merged writer before saving
    mock_add_page_numbers.assert_called_once_with(mock_pdf_writer.return_value)


def test_add_page_numbers(mocker: Any) -> None:
    page1 = MagicMock()
    page1.MediaBox = ["0", "0", "612", "792"]
    page2 = MagicMock()
    page2.MediaBox = ["0", "0", "612", "792"]

    pdf_output = MagicMock()
    pdf_output.pagearray = [page1, page2]

    mock_page_merge = mocker.patch("pdfrw.PageMerge")
    mock_pdf_reader = mocker.patch("pdfrw.PdfReader")
    mock_pdf_reader.return_value.pages = [MagicMock()]

    add_page_numbers(pdf_output)

    assert mock_page_merge.call_count == 2
    mock_page_merge.assert_any_call(page1)
    mock_page_merge.assert_any_call(page2)
    assert mock_page_merge.return_value.add.return_value.render.call_count == 2


def test_save_paragraphs_to_pdf(mocker: Any) -> None:
    mocker.patch("src.utils.pdf.convert_to_snake_case", return_value="test_title")
    mocker.patch("reportlab.platypus.SimpleDocTemplate.build")

    title = "Test Title"
    headings = ["Heading 1", "Heading 2"]
    paragraphs = ["Paragraph 1", "Paragraph 2"]
    output_dir = "/tmp"

    file_path = save_paragraphs_to_pdf(title, headings, paragraphs, output_dir)

    assert file_path == "/tmp/test_title.pdf"
    SimpleDocTemplate.build.assert_called_once()
