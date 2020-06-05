import os
import deeplearning
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def home():
    return app.send_static_file('home.html')

@app.route('/analise', methods=['POST'])
def analise():
    ia = deeplearning.DeepLearning()
    resultado = ia.preditc_IA(request.form['base64'])
    return jsonify({"message": resultado})

@app.route('/train')
def train():
    ia = deeplearning.DeepLearning()
    ia.training_IA()
    return jsonify({"message": "ss"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)