import datasets
from datasets import load_dataset

ds_builder = datasets.load_dataset_builder(
    "code_search_net", name="java", token=True, cache_dir="D:\\", trust_remote_code=True
)

ds_builder.download_and_prepare(output_dir="D:\\CSN")
