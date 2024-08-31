from flask import Flask, request, Response
from flask_restful import Resource, Api
from flask_cors import CORS

from llm import T5, StarEncoder
from pymilvus import MilvusClient
import torch
import json

app = Flask(__name__)
cors = CORS(app)
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
        super(EmbeddingService, self).__init__() 
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

        code = data["code"]
        id = data["id"]
        name = data["name"]

        #prevent duplicates
        res = client.query(
            collection_name=self.collection_name,
            filter=f'name == "{name}"',
            limit=1000,  # number of returned entities
            output_fields=[
                "solr_id"
            ],
        )
    
        if len(res) > 0:
            print("Found duplicate:", res)
            return Response(status=309) # duplicate found

        encoding = self.model.encode([code])

        data = {
            "vector": encoding[0],
            "solr_id": id,
            "name": name
        }

        client.insert(collection_name=self.collection_name, data=data)

        return Response(status=204)


class SearchService(Resource):
    def __init__(self, model, collection_name):
        super(SearchService, self).__init__() 
        self.model = model
        self.collection_name = collection_name

    def post(self):
        data = request.get_json()
        query = data["input"]
        query_vector = self.model.encode([query])

        num_results = request.args.get('numResults', default=100, type=int)

        res = client.search(
            collection_name=self.collection_name,
            data=query_vector,
            limit=num_results,  # number of returned entities
            output_fields=[
                "solr_id",
                "name"
            ],
        )
        return Response(
            response=json.dumps(
                {
                    "total": len(res[0]),
                    "implementations": [
                        {
                            "score": index + 1,
                            "solr_id": result["entity"]["solr_id"],
                            "name": result['entity']["name"]
                        }
                        for index, result in enumerate(res[0])
                    ],
                }
            ),
            status=200,
        )


api.add_resource(
    EmbeddingService,
    "/embedding",
    endpoint="embedding_service_app",  
    resource_class_kwargs={"model": model, "dim": 256, "collection_name": "classEmbedding"},
)
api.add_resource(
    EmbeddingService,
    "/methodEmbedding",
    endpoint="embedding_service_methods",  
    resource_class_kwargs={"model": model, "dim": 256, "collection_name": "methodEmbedding"},
)
api.add_resource(
    SearchService,
    "/search",
    endpoint="search_service_app",  
    resource_class_kwargs={"model": model, "collection_name": "classEmbedding"},
)
api.add_resource(
    SearchService,
    "/searchMethods",
    endpoint="search_service_methods",  
    resource_class_kwargs={"model": model, "collection_name": "methodEmbedding"},
)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4999, debug=True)
