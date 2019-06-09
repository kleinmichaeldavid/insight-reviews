from flaskexample import app
from flask import request, render_template

### for plot
import plotly
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import json


@app.route('/', methods=["GET"])
def index():

    return render_template('index.html',slider_num = -2,review_text = '')


@app.route("/", methods=["POST"])
def submit_review():
    review_text = request.form["review_input"]
    print('doo doo doo')
    return render_template('index.html',review_text = review_text)


if __name__ == '__main__':
    app.run(host="0.0.0.0")