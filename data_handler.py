import pandas as pd
import shutil
import os

def load_df():
    file_path = "staffrecruitment.xlsx"
    temp_path = "temp_staffrecruitment.xlsx"
    
    # Try reading from a copy to bypass Windows Excel file locks
    try:
        if os.path.exists(file_path):
            shutil.copy(file_path, temp_path)
            df = pd.read_excel(temp_path, header=5)
            try:
                os.remove(temp_path)
            except Exception:
                pass
            return df
    except Exception as e:
        print(f"Error loading copy of excel file (header=5): {e}")
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass

    try:
        if os.path.exists(file_path):
            shutil.copy(file_path, temp_path)
            df = pd.read_excel(temp_path)
            try:
                os.remove(temp_path)
            except Exception:
                pass
            return df
    except Exception as e:
        print(f"Error loading copy of excel file (standard): {e}")
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass

    # Direct fallback if copying fails or files don't exist
    try:
        return pd.read_excel(file_path, header=5)
    except Exception as e:
        print(f"Error loading excel file directly (header=5): {e}")
        return pd.read_excel(file_path)

def get_shortlisted():
    df = load_df()
    if "Status" in df.columns and "Full Name" in df.columns:
        shortlisted = df[df["Status"] == "Shortlisted"]
        return shortlisted["Full Name"].tolist()
    return []

def get_selected():
    df = load_df()
    if "Status" in df.columns and "Full Name" in df.columns:
        selected = df[df["Status"] == "Selected"]
        return selected["Full Name"].tolist()
    return []

def get_all_candidates():
    try:
        df = load_df().fillna("")
        records = df.to_dict(orient="records")
        cleaned_records = []
        for r in records:
            cleaned_row = {str(k): str(v) for k, v in r.items()}
            cleaned_records.append(cleaned_row)
        return cleaned_records
    except Exception as e:
        print(f"Error in get_all_candidates: {e}")
        return []