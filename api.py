from flask import Flask, request
from flask_restful import Resource, Api
from llm import StarEncoder


app = Flask(__name__)
api = Api(app)
MAX_INPUT_LEN = 10000
MAX_TOKEN_LEN = 1024
AUTH_TOKEN = "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"

DEVICE = "cuda:0"

starencoder = StarEncoder(DEVICE, MAX_INPUT_LEN, MAX_TOKEN_LEN)


class EmbeddingService(Resource):
    def __init__(self, model):
        super()
        self.model = model

    def post(self):
        data = request.get_json()
        input = data["input"]
        print("hallo", self.model.encode(input))
        return {"hello": "world"}


api.add_resource(
    EmbeddingService, "/embedding", resource_class_kwargs={"model": starencoder}
)


if __name__ == "__main__":
    app.run(debug=True)
