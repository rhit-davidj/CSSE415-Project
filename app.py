from flask import Flask, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('coffee_analysis.csv')

@app.route('/')
def index():
    return render_template('templates/index.html')