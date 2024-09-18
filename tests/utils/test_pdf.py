from typing import Any

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from src.utils.pdf import df_to_pdf, merge_pdfs


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


def test_merge_pdfs(mocker: Any) -> None:
    input_files = ["file1.pdf", "file2.pdf"]
    output_file = "merged.pdf"

    mock_pdf_writer = mocker.patch("pdfrw.PdfWriter")
    mock_pdf_reader = mocker.patch("pdfrw.PdfReader")
    mock_os_remove = mocker.patch("os.remove")

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
