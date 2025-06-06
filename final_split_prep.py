import polars as pl
import numpy as np

# File paths
INPUT_PATH = "/gpfs/arx2/med116_mde/proj-shared/RITM0276466/new_data/raw_new_IE_data_with_split.parquet"
OUTPUT_PATH = "/gpfs/arx2/med116_mde/proj-shared/RITM0276466/new_data/raw_new_IE_data_with_train_test_val.parquet"

# 1. Load full data (test + unsplit records)
df = pl.read_parquet(INPUT_PATH)

# 2. Separate test and to-be-split (train/val) sets
test_df = df.filter(pl.col("split") == "test")
to_split_df = df.filter(pl.col("split") == "none")

# --- FIX: Exclude test groups from train/val split ---
group_cols = ["patient_id_number", "_meta_registry"]
test_groups = test_df.select(group_cols).unique()
to_split_df = to_split_df.join(test_groups, on=group_cols, how="anti")
# --- END FIX ---

# 3. Get unique groups for splitting (from non-test only)
groups = to_split_df.select(group_cols).unique()
groups = groups.sample(fraction=1.0, seed=42)  # Shuffle for random split

# 4. Assign train/val split to groups
n_groups = groups.height
n_train = int(np.round(n_groups * 0.90))
split_labels = ["train"] * n_train + ["val"] * (n_groups - n_train)
groups = groups.with_columns(pl.Series("split", split_labels))

# 5. Join group split back to all to-be-split rows
to_split_df = to_split_df.join(groups, on=group_cols, how="left").with_columns([
    pl.col("split_right").alias("split")
]).drop("split_right")

# 6. Combine all splits
final_df = pl.concat([test_df, to_split_df])

# 7. Save output
final_df.write_parquet(OUTPUT_PATH)
print(f"Saved split data to: {OUTPUT_PATH}")

# 8. Split counts
print("\nSplit counts:")
print(final_df["split"].value_counts())

# 9. Group leakage check (should all be 1)
group_leakage = final_df.group_by(group_cols).agg(pl.col("split").n_unique().alias("nunique_split"))
n_leaked = group_leakage.filter(pl.col("nunique_split") > 1).height
if n_leaked == 0:
    print("\nAll groups are contained in a single split (no leakage).")
else:
    print(f"\n{n_leaked} group(s) are split across train/val/test! Inspect:")
    print(group_leakage.filter(pl.col("nunique_split") > 1))

# 10. (Optional) Overlap check between splits for group columns
def get_group_tuples(df, split_value):
    return set(tuple(x) for x in df.filter(pl.col("split") == split_value).select(group_cols).unique().rows())

train_groups = get_group_tuples(final_df, "train")
val_groups = get_group_tuples(final_df, "val")
test_groups = get_group_tuples(final_df, "test")

print("\nGroup overlap checks (should all be set()):")
print("Train ∩ Val:", train_groups & val_groups)
print("Train ∩ Test:", train_groups & test_groups)
print("Val ∩ Test:", val_groups & test_groups)
