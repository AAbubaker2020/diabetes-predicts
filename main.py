import pickle
import numpy as np
from flask import Flask, request, render_template
from sklearn.linear_model import LogisticRegression


# create Flask object
app = Flask(__name__)

# Read the model using pickle
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict():

    
    to_predict_list = [int(x) for x in request.args.values()]
    to_predict_list = list(to_predict_list)
    to_predict_list = list(map(int, to_predict_list))
    to_predict = np.array(to_predict_list).reshape(1, 11)
    prediction = model.predict(to_predict)


    
    if prediction == 0:

        prediction = "No diabetes"

    elif prediction == 1:

        prediction = "Pre-diabetes" 

    else:

        prediction = "Diabetes"


    
    return render_template('predict.html',prediction_text=f'            Based on this data the model predcit your result is:  "{prediction}"' )




@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/plot')
def plot():
    return render_template('plot.html')


@app.route('/new_predict')
def new_predict():
    return render_template('predict.html')


if __name__== '__main__':
    app.run(debug=True)