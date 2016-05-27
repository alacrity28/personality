from flask import Flask, render_template, request
from jinja2 import Template
import os
from model import model
import pandas as pd

app = Flask(__name__)


@app.route('/')
def welcome():
    return render_template('portfolio.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/bigfive')
def bigfive():
    return render_template('bigfive.html')

@app.route('/generalanalysis')
def generalanalysis():
    return render_template('generalanalysis.html')

if __name__ == '__main__':
    main_ = model()
    data, questions, country_dict, questions_dict, answers_clean, answers_messy = main_.load_data()
    app.run(host='0.0.0.0',port=8000, debug=True)
