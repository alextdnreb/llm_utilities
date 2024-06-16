from typing import Union
from datasets import IterableDatasetDict
from tqdm import tqdm
from pymilvus import MilvusClient
import numpy as np
import torch


def evaluate(
    model,
    client: MilvusClient,
    dataset: IterableDatasetDict,
    collection_name,
    size=1000,
):
    for idx, example in enumerate(tqdm(dataset["test"].shuffle(seed=42))):
        encoding = model.encode([example["func_code_string"]])

        data = {
            "text": example["func_code_string"],
            "vector": encoding[0],
        }

        client.insert(collection_name=collection_name, data=data)

        if idx == size:
            print("insert complete")
            break

    ranks = []
    for idx, example in enumerate(tqdm(dataset["test"].shuffle(seed=42))):

        query_vector = model.encode([example["func_documentation_string"]])

        res = client.search(
            collection_name=collection_name,  # target collection
            data=query_vector,  # query vectors
            limit=1000,  # number of returned entities
            output_fields=["text"],  # specifies fields to be returned
        )

        for index, result in enumerate(res[0]):

            if example["func_code_string"] == result["entity"]["text"]:

                ranks.append(1 / (index + 1))

        if idx == size:
            print("Search complete")
            break

    print(np.mean(ranks))
