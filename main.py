from flask import Flask

app = Flask(__name__)

@app.route("/")

def hello():
    return "Ops!! Voce nao deveria estar aqui."

if __name__ == "__main__":
    app.run()