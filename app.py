from flask import Flask, render_template, request, redirect
from jinja2 import Template
import os
from model import model
import pandas as pd

app = Flask(__name__)


@app.route('/')
def welcome():
    return render_template('portfolio.html')

@app.route('/linkedin')
def linkedin():
    return redirect('https://www.linkedin.com/in/erinwolpert')

@app.route('/github')
def github():
    return redirect('https://github.com/alacrity28/personality')

@app.route('/bigfive')
def bigfive():
    return render_template('bigfive.html')

@app.route('/generalanalysis')
def generalanalysis():
    return render_template('generalanalysis.html')

@app.route('/trait1')
def trait1():
    return render_template('Trait1.html')

@app.route('/trait2')
def trait2():
    return render_template('Trait2.html')

@app.route('/trait3')
def trait3():
    return render_template('Trait3.html')

@app.route('/trait4')
def trait4():
    return render_template('Trait4.html')

@app.route('/trait5')
def trait5():
    return render_template('Trait5.html')

@app.route('/trait6')
def trait6():
    return render_template('Trait6.html')

@app.route('/trait7')
def trait7():
    return render_template('Trait7.html')

@app.route('/trait8')
def trait8():
    return render_template('Trait8.html')


if __name__ == '__main__':
    main_ = model()
    data, questions, country_dict, questions_dict, answers_clean, answers_messy = main_.load_data()

    app.run(host='0.0.0.0', port=8082, debug=False)
