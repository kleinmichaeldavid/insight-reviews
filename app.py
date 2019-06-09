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

    def create_plot(addition):


        N = 40
        x = np.linspace(0, 1, N)
        y = np.random.randn(N)
        df = pd.DataFrame({'x': x, 'y': y}) # creating a sample dataframe


        data = [
            go.Bar(
                x=df['x'], # assign x as the dataframe column 'x'
                y=df['y']+addition
            )
        ]

        graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

        return graphJSON

    bar = create_plot(0)
    bar2 = create_plot(2)
    #slider_num = request.form["name_of_slider"]
    return render_template('index.html',slider_num = 0,review_text = '', plot=bar, plot2=bar2)

# @app.route("/", methods=["POST"])
# def test():
#     slider_num = request.form["name_of_slider"]
#     return render_template('index.html',slider_num = slider_num)

@app.route("/", methods=["POST"])
def submit_review():
    review_text = request.form["review_input"]
    print('doo doo doo')
    return render_template('index.html',review_text = review_text)


# @app.route('/')
# def index():

#     def create_plot():


#         N = 40
#         x = np.linspace(0, 1, N)
#         y = np.random.randn(N)
#         df = pd.DataFrame({'x': x, 'y': y}) # creating a sample dataframe


#         data = [
#             go.Bar(
#                 x=df['x'], # assign x as the dataframe column 'x'
#                 y=df['y']
#             )
#         ]

#         graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

#         return graphJSON

#     bar = create_plot()
#     return render_template('index.html', plot=bar)

if __name__ == '__main__':
    app.run(host="0.0.0.0")