from flask import Flask, render_template
import pickle
import numpy as np

model = pickle.load(open(''))

app = Flask(__name__)






@app.route('/home')
def home():
    return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)
