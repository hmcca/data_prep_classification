import polars as pl
from transformers import AutoTokenizer
from tqdm import tqdm

# File paths
INPUT_PATH = "/gpfs/arx2/med116_mde/proj-shared/RITM0276466/new_data/raw_new_IE_data_label_mapped.parquet"
OUTPUT_PATH = "/gpfs/arx2/med116_mde/proj-shared/RITM0276466/new_data/raw_new_IE_data_tokenized.parquet"
MODEL_PATH = "/sw/summit/mde/med116_mde/krawczukp/Llama-3.1-8B"

# Tokenizer parameters
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    model_max_length=4096,
    pad_token='<|end_of_text|>'
)

print(f"Reading data from {INPUT_PATH}")
df = pl.read_parquet(INPUT_PATH)

texts = df["text"].to_list()

# Tokenize with progress bar
tokenized = []
for text in tqdm(texts, desc="Tokenizing text column"):
    tokenized.append(tokenizer.encode(text, truncation=True, max_length=4096))

df = df.with_columns(
    pl.Series("text_tokenized", tokenized)
)

print(f"Saving tokenized data to {OUTPUT_PATH}")
df.write_parquet(OUTPUT_PATH)
print("Done.") 