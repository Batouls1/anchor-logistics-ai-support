from datasets import load_dataset
import pandas as pd
import re

COMPANY_NAME = "Anchor Logistics"
SUPPORT_EMAIL = "support@anchorlogisticslb.com"

dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
df = dataset["train"].to_pandas()

# Fill in the template placeholders with the company's identity
df["response"] = (
    df["response"]
    .str.replace("{{Company Name}}", COMPANY_NAME, regex=False)
    .str.replace("{{Customer Support Email}}", SUPPORT_EMAIL, regex=False)
    .str.replace("{{Customer Support Phone Number}}", "+961 1 234 5672", regex=False)
    .str.replace("{{Website URL}}", "www.anchorlogisticslb.com", regex=False)
    .str.replace("{{Customer Support Hours}}", "Monday–Saturday, 9 AM–8 PM (Beirut time)", regex=False)
    .str.replace("{{Online Company Portal Info}}", "portal.anchorlogisticslb.com", regex=False)
)


def humanize_placeholder(match):
    tag = match.group(1)
    return f"your {tag.lower()}"


# Clean up remaining dynamic placeholders (order number, tracking number,
# delivery country, etc.) in BOTH columns. `response` is what gets shown to
# the user, so it can't contain raw template syntax. `instruction` is what
# gets embedded and searched at retrieval time
for col in ["response", "instruction"]:
    df[col] = df[col].str.replace(r"\{\{(.*?)\}\}", humanize_placeholder, regex=True)
    df[col] = df[col].str.replace(r"\{\{(.*?)\}\}", humanize_placeholder, regex=True)  # safety net pass

df.to_csv("data/bitext_dataset/processed.csv", index=False)
print(f"Saved {len(df)} rows to data/bitext_dataset/processed.csv")