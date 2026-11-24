"""
Utility: File Handler
Provides in-memory binary/text encoding for seamless Streamlit download buttons:
- CSV download buffer
- Excel (.xlsx) download buffer using openpyxl
- JSON download buffer
"""

import io
import json
import pandas as pd
from typing import Optional


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Encodes a DataFrame into UTF-8 CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "PreprocessedData") -> bytes:
    """Encodes a DataFrame into Excel .xlsx bytes using OpenPyXL engine."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    output.seek(0)
    return output.getvalue()


def dataframe_to_json_str(df: pd.DataFrame) -> str:
    """Converts DataFrame to formatted JSON string."""
    return df.to_json(orient="records", indent=2, date_format="iso")
