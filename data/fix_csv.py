import pandas as pd

# Try different encodings if you see errors
encodings = ['utf-8', 'latin1', 'ISO-8859-1']

for enc in encodings:
    try:
        df = pd.read_csv("phishing_emails.csv", encoding=enc)
        break
    except Exception as e:
        print(f"Encoding {enc} failed: {e}")
else:
    raise Exception("Could not read file with tried encodings.")

# Keep only first two columns, drop others
df_clean = df.iloc[:, :2]
df_clean.columns = ['label', 'email_content']

# Drop rows where email_content or label is missing
df_clean = df_clean.dropna(subset=['label', 'email_content'])

# Save to clean csv
df_clean.to_csv("phishing_emails.cleaned.csv", index=False)
print("Cleaned CSV saved as phishing_emails.cleaned.csv")
