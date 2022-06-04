from flask import Flask, jsonify, request
from utilities import predict

app = Flask(__name__)

@app.post('/api')
def pred():
    try:
        url = request.values.get('url')
    except KeyError:
        return jsonify({'error': 'No url found'})
    
    prediction = predict(url)

    try:
        result = jsonify(prediction)
    except TypeError as e:
        result = jsonify({'error': str(e)})
    
    return result

if(__name__ == '__main__'):
    app.run(host='0.0.0.0', debug=True)