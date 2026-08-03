import pandas as pd

df = pd.read_csv("data/bitext_dataset/processed.csv")

print("Rows:", len(df))
print("\nNull values:\n", df.isna().sum())
print("\nEmpty responses:", (df["response"].str.strip() == "").sum())
print("\nRemaining {{...}} tags in response:", df["response"].str.contains(r"\{\{.*?\}\}", regex=True, na=False).sum())
print("Remaining {{...}} tags in instruction:", df["instruction"].str.contains(r"\{\{.*?\}\}", regex=True, na=False).sum())
print("\nExact duplicate rows:", df.duplicated().sum())
print("\nResponse length stats:\n", df["response"].str.len().describe())
print("\nCategories:", df["category"].nunique(), "| Intents:", df["intent"].nunique())