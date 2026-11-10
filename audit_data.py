
import pandas as pd

DATA_PATH = "data/Crop_recommendation.csv"

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

features = [
    "N", "P", "K", "temperature",
    "humidity", "ph", "rainfall"
]
target = "label"

print("=" * 55)
print("CROP DATASET QUALITY AUDIT")
print("=" * 55)

# 1. Dataset dimensions
print("\n1. DATASET SIZE")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Column names:", df.columns.tolist())

# 2. Missing values
print("\n2. MISSING VALUES")
print(df.isnull().sum())

# 3. Duplicate records
print("\n3. DUPLICATE CHECK")
print("Exact duplicate rows:", df.duplicated().sum())

# Same input measurements appearing more than once
print(
    "Duplicate feature rows:",
    df.duplicated(subset=features).sum()
)

# 4. Crop class distribution
print("\n4. CROP CLASS DISTRIBUTION")
print(df[target].value_counts().sort_index())

# 5. Feature ranges
print("\n5. FEATURE SUMMARY")
print(df[features].describe().round(2).to_string())

# 6. Identical measurements assigned different crop labels
print("\n6. CONFLICTING LABEL CHECK")
feature_label_counts = (
    df.groupby(features, dropna=False)[target]
      .nunique()
)
conflicts = int((feature_label_counts > 1).sum())

print(
    "Unique feature combinations assigned multiple labels:",
    conflicts
)

# 7. Numeric validity
print("\n7. INVALID NUMERIC VALUES")
numeric = df[features].apply(pd.to_numeric, errors="coerce")

print(
    "Values that are missing or non-numeric per feature:"
)
print(numeric.isna().sum())

print(
    "\nNegative values per feature "
    "(review before deciding whether invalid):"
)
print((numeric < 0).sum())

# 8. Possible feature ranges to review
print("\n8. OBSERVED MINIMUM AND MAXIMUM")
for column in features:
    print(
        f"{column}: "
        f"min={numeric[column].min():.3f}, "
        f"max={numeric[column].max():.3f}"
    )

print("\nAudit completed. No rows have been changed.")
