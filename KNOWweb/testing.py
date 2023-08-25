from flask import Flask, request
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

@app.route('/test')
def test():
    return "first message"


@app.route('/testForm', methods=['POST'])
def testForm():
    data = request.get_json()
    name = data.get('name')
    print(name)
    return f"SUCCESS {name}"


@app.route('/test_param/<my_param>', methods=['POST'])
def test_param(my_param):
    print("myparam=" + my_param)


if __name__ == '__main__':
    app.run()



