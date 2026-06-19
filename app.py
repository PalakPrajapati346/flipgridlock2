from flask import Flask
app = Flask(__name__)

@app.route("/")
def home():
    return "Gridlock 2.0 Command Center Running"

if __name__ == "__main__":
    app.run(debug=True)