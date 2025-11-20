import pandas as pd

df = pd.read_excel("dataset.xlsx", header=None)

rows = []
for i, row in df.iterrows():
    user = str(row.iloc[0]).strip()
    values = row.iloc[1:].dropna().tolist()
    for j in range(0, len(values), 2):
        try:
            title = str(values[j]).strip()
            rating = float(values[j+1])
        except Exception:
            continue
        rows.append((user, title, rating))

ratings = pd.DataFrame(rows, columns=["user", "title", "rating"])
print(ratings.head())

ratings.to_csv("ratings.csv", index=False)
