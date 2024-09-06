from typing import Any

import pandas as pd

from src.utils.pdf import df_to_pdf, saved_pdf_files


def test_df_to_pdf_without_highlighting(mocker: Any) -> None:
    mocker.patch("matplotlib.backends.backend_pdf.PdfPages", autospec=True)
    mocker.patch("src.utils.pdf.saved_pdf_files", [])

    df = pd.DataFrame({"Column1": [1, 2, 3], "Column2": [4, 5, 6]})
    df_to_pdf("Test Title", df, "test_output.pdf")

    assert len(saved_pdf_files) == 1
    assert saved_pdf_files[0] == "test_output.pdf"
