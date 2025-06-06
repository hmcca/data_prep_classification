import polars as pl
import json

# File paths
INPUT_PATH = "/gpfs/arx2/med116_mde/proj-shared/RITM0276466/new_data/raw_new_IE_data_with_train_test_val.parquet"
OUTPUT_JSON = "/gpfs/arx2/med116_mde/proj-shared/RITM0276466/new_data/label_mappings.json"
OUTPUT_PARQUET = "/gpfs/arx2/med116_mde/proj-shared/RITM0276466/new_data/raw_new_IE_data_label_mapped.parquet"

# Columns of interest
cols = ['site', 'subsite', 'laterality', 'histology', 'behavior']

# Load data
print(f"Reading data from {INPUT_PATH}")
df = pl.read_parquet(INPUT_PATH)

# Build label mappings
def build_mapping(series):
    unique_labels = series.unique().to_list()
    return {label: idx for idx, label in enumerate(sorted(unique_labels))}

label_mappings = {}
for col in cols:
    if col in df.columns:
        unique_vals = df[col].unique().to_list()
        print(f"{col}: {len(unique_vals)} unique values BEFORE mapping")
        label_mappings[col] = build_mapping(df[col])
        print(f"{col}: {len(label_mappings[col])} unique labels in mapping")
    else:
        print(f"Warning: Column '{col}' not found in data.")

# Save to JSON
with open(OUTPUT_JSON, 'w') as f:
    json.dump(label_mappings, f, indent=2)
print(f"Label mappings saved to {OUTPUT_JSON}")

# Replace original columns with mapped integer values using replace
for col in cols:
    if col in df.columns:
        df = df.with_columns(
            pl.col(col).replace(label_mappings[col], default=-1).alias(col)
        )
        mapped_unique_vals = df[col].unique().to_list()
        print(f"{col}: {len(mapped_unique_vals)} unique values AFTER mapping")

# Save the updated DataFrame
print(f"Saving label-mapped data to {OUTPUT_PARQUET}")
df.write_parquet(OUTPUT_PARQUET)
print("Done.") 