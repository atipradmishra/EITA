import pandas as pd
import os

def load_business_context(excel_path: str = "ETAI Business domain Dictionary PW.xlsx") -> dict:
    """
    Dynamically load business context definitions from an Excel workbook with arbitrary sheet
    and column names, using positional inference:

    - Two-column sheets: interprets first column as key, second as value.
    - Four-column sheets: interprets first column as key, third as value.
    - Other formats: falls back to list of row dictionaries.

    Args:
        excel_path: Path to the Excel file.

    Returns:
        A tuple containing:
            - business_dict: A dict mapping normalized sheet names to code-to-label mappings or records.
            - excel_file_name: The name of the uploaded Excel file.
    """
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file '{excel_path}' not found.")

    # Extract the filename from the path
    excel_file_name = os.path.basename(excel_path)

    # Read all sheets
    sheets = pd.read_excel(excel_path, sheet_name=None)
    business_dict = {}

    for sheet_name, df in sheets.items():
        # Drop entirely empty rows
        df = df.dropna(how="all").copy()
        # Reset column names to positional labels
        df.columns = [f"col_{i}" for i in range(len(df.columns))]

        # Determine mapping by column count
        if df.shape[1] >= 2:
            if df.shape[1] == 2:
                # Two-column: key->value
                mapping = dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))
            elif df.shape[1] >= 4:
                # Four or more columns: use first as key, third as meaning/value
                mapping = dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 2].astype(str)))
            else:
                # Three columns: assume second is value
                mapping = dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))
        else:
            # Fallback: list of records
            mapping = df.to_dict(orient="records")

        # Normalize sheet name to uppercase key without spaces or special chars
        key = sheet_name.strip().upper().replace(" ", "_")
        business_dict[key] = mapping

    return business_dict, excel_file_name  # Return both the business dictionary and the file name
