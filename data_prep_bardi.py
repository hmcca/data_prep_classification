import pandas as pd
import json
import os
from pathlib import Path

from bardi.data import data_handlers
from bardi.nlp_engineering import CPULabelProcessor, CPUTokenizerEncoder
from bardi.pipeline import Pipeline, DataWriteConfig
from bardi.nlp_engineering.splitter import NewSplit
from bardi.nlp_engineering.splitter import CPUSplitter

from transformers import AutoTokenizer

# DATASET_PATH = "datasets/medical_abstracts_BARDI_ready.parquet"
INPUT_PATH = "/gpfs/arx2/med116_mde/proj-shared/RITM0276466/new_data/raw_new_IE_data_label_mapped.parquet"
OUTPUT_PATH = "/gpfs/arx2/med116_mde/proj-shared/RITM0276466/new_data_bardi/"

text_fields = ['text']

def main():

    # repo_path = Path().resolve()

    # Prepare for pipeline outputs
    output_directory = OUTPUT_PATH
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # read in dataset
    dataset = pd.read_parquet(INPUT_PATH)
    # register a dataset
    dataset = data_handlers.from_pandas(dataset)
    # initialize a pipeline
    pipeline = Pipeline(dataset=dataset, write_outputs=False)

    hf_data_write_config: DataWriteConfig = {
        "data_format": "hf-dataset",
        "data_format_args": {},
    }

    # Create a pipeline object
    pipeline = Pipeline(
        dataset=dataset,
        write_path=output_directory,
        write_outputs="pipeline-outputs",
        data_write_config=hf_data_write_config
    )

    # adding the label processor step to the pipeline
    pipeline.add_step(CPULabelProcessor(fields=['site', 'subsite', 'laterality', 'histology', 'behavior']))


    pipeline.add_step(
        CPUTokenizerEncoder(
            fields=text_fields,
            hf_cache_dir="/sw/summit/mde/med116_mde/gounley1",
            model_name = 'Meta-Llama-3.1/Meta-Llama-3-8B',
            cores=32,
            tokenizer_params={
                "model_max_length": 4096,
                "pad_token" : '<|end_of_text|>'}
        )
    )

    pipeline.run_pipeline()
    # Get pipeline metadata
    pipeline_params = pipeline.get_parameters(condensed=True)
    with open(f"{output_directory}/metadata.json", "w") as f:
        json.dump(pipeline_params, f, indent=4)
    f.close()

if __name__ == "__main__":
    main()