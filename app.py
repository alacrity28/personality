from flask import Flask, render_template, request
from jinja2 import Template
import os
from model import model

app = Flask(__name__)


@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/portfolio')
def portfolio():
    return render_template('portfolio.html')

@app.route('/bigfive')
def bigfive():
    return render_template('bigfive.html')

@app.route('/generalanalysis')
def generalanalysis():
    return render_template('generalanalysis.html')


@app.route('/fullquiz')
def fullquiz():
    question_pairs = {}
    for i in questions.iterrows():
        question_pairs[i[1][0]] = i[1][1]
    return render_template('fullquiz.html', question_pairs = question_pairs)

@app.route('/shortquiz')
def shortquiz():
    return render_template('shortquiz.html')


@app.route('/results', methods = ['GET', 'POST'])
def results():

    fullquiz_results = {}
    for i in request.form:
        fullquiz_results[i] = request.form[i]
    # for i in questions.iterrows():
        # fullquiz_results[i[1][0]] = request.form[i[1][0]]

    return render_template('results.html', fullquiz_results = fullquiz_results)


@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    main_ = model()
    data, questions, country_dict, questions_dict, answers_clean = main_.load_data()
    app.run(host='0.0.0.0',port=8000, debug=True)
