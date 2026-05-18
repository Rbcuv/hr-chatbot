import pandas as pd

df = pd.read_excel("staffrecruitment.xlsx")

def get_shortlisted():
    shortlisted = df[df["Status"] == "Shortlisted"]
    return shortlisted["Full Name"].tolist()

def get_selected():
    selected = df[df["Status"] == "Selected"]
    return selected["Full Name"].tolist()