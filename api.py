from flask import Flask, request, Response, jsonify
from flask_restful import Resource, Api
from llm import T5, StarEncoder
from pymilvus import MilvusClient, utility
import torch
import json

app = Flask(__name__)
api = Api(app)
MAX_INPUT_LEN = 10000
MAX_TOKEN_LEN = 1024
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
config = {
    "max_input_len": 10000,
    "maximum_token_len": 512,
    "device": DEVICE,
}

model = T5(**config)
client = MilvusClient(uri="http://localhost:19530", user="root", password="Milvus")


class EmbeddingService(Resource):
    def __init__(self, model, dim, collection_name):
        app.logger.info("test")
        super()
        self.model = model

        if not collection_name in client.list_collections():
            client.create_collection(
                collection_name=collection_name,
                dimension=dim,
                auto_id=True,
            )
        self.collection_name = collection_name

    def post(self):
        app.logger.info("Incoming Request for embedding")
        data = request.get_json()
        app.logger.info(data)
        print(data)
        code = data["input"]
        name = data["name"]

        encoding = self.model.encode([code])

        data = {"text": code, "vector": encoding[0], "name": name}

        client.insert(collection_name=self.collection_name, data=data)

        return Response(status=201)


class SearchService(Resource):
    def __init__(self, model, collection_name):
        super()
        self.model = model
        self.collection_name = collection_name

    def post(self):
        data = request.get_json()
        code = data["input"]
        query_vector = self.model.encode([code])
        res = client.search(
            collection_name=self.collection_name,
            data=query_vector,
            limit=5,  # number of returned entities
            output_fields=["text"],
        )

        app.logger.info(res[0])
        return Response(
            response=json.dumps(
                {
                    "results": [
                        result["entity"]["text"] for index, result in enumerate(res[0])
                    ]
                }
            ),
            status=200,
        )


api.add_resource(
    EmbeddingService,
    "/embedding",
    resource_class_kwargs={"model": model, "dim": 256, "collection_name": "app"},
)
api.add_resource(
    SearchService,
    "/search",
    resource_class_kwargs={"model": model, "collection_name": "app"},
)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
