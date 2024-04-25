from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/hello', methods=['POST'])
def hello():
    print("Hello World")
    return jsonify(message="Hello World")

if __name__ == '__main__':
    app.run(debug=True)