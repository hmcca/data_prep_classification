import polars as pl

# Load datasets
old_df = pl.read_parquet("/gpfs/arx2/med116_mde/proj-shared/RITM0276466/clc_API_step1_window_499_preds_20240404213444.parquet")
new_df = pl.read_parquet("/gpfs/arx2/med116_mde/proj-shared/RITM0276466/raw_new_IE_data.parquet")

# Identify test set in old data (if "val" maps to new "test")
old_test = old_df.filter(pl.col("split") == "val").select([
    'record_document_id', 'patient_id_number', 'tumor_record_number'
]).unique()

# Add a "test" column to help with mapping
old_test = old_test.with_columns(pl.lit("test").alias("split"))

# Left join to propagate "test" label to new data
merged = new_df.join(
    old_test,
    on=['record_document_id', 'patient_id_number', 'tumor_record_number'],
    how='left'
)

# The merged 'split' column will be "test" where matched, else null
merged = merged.with_columns([
    pl.col("split").fill_null("none")  # Or leave as null if you want
])

# Save for next steps
merged.write_parquet("/gpfs/arx2/med116_mde/proj-shared/RITM0276466/new_data/raw_new_IE_data_with_split.parquet")

# Basic counts
total_records = merged.height
test_count = merged.filter(pl.col("split") == "test").height
none_count = merged.filter(pl.col("split") == "none").height
null_count = merged.filter(pl.col("split").is_null()).height  # if you didn't use fill_null

print(f"Total records: {total_records}")
print(f"Test records (split == 'test'): {test_count}")
print(f"Unsplit records (split == 'none'): {none_count}")
print(f"Null split values (if not filled): {null_count}")


print("Column names in merged DataFrame:")
print(merged.columns)
