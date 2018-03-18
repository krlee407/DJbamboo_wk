from flask import Flask
from flask_restful import Api, Resource, reqparse

import json
from DJbamboo import comeondata, Djbamboo

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('rec')

'''
@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    rst = "replay"
    if request.method == 'POST':
        try:
            inputStory =  request.form['myStory'];
            rst = Djbamboo(inputStory)
        except Exception as e:
            print(e)
    # return(jsonify(result=rst))
    return json.dumps(rst, ensure_ascii=False).encode('utf8')
'''

result = []

class recommend(Resource):
    def get(self, st):
        print('debug : ', st)
        #print(json.dumps(Djbamboo(st), ensure_ascii=False).encode('utf8'))
        return json.dumps(Djbamboo(st), ensure_ascii=False)

    def post(self, st):
        return result

api.add_resource(recommend, '/<string:st>/')

if __name__ == "__main__":
	comeondata()
	app.run()