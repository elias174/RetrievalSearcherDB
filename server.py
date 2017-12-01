from flask import Flask
from flask_restful import Resource, Api
from force_brute import LDAModel, BruteForce

app = Flask(__name__)
api = Api(app)


class LDAResource(Resource):
    def __init__(self):
        self.model = LDAModel()
        self.model.init_data()

    def get(self, query):
        result = self.model.make_query(str(query))[:20]
        return {'results': result}


class BruteResource(Resource):
    def __init__(self):
        self.model = BruteForce()
        self.model.init_data()

    def get(self, query):
        result = self.model.make_query(str(query))[:20]
        return {'results': result}


api.add_resource(BruteResource, '/brute/<string:query>')
api.add_resource(LDAResource, '/lda/<string:query>')

if __name__ == '__main__':
    print('Initializing Data..')
    app.run(host= '0.0.0.0', debug=True)