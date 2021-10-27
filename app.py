import pickle
import numpy as np
import os
import joblib
from joblib import dump,load
import numpy as np
from flask import Flask, request, jsonify, render_template



from flask import Flask, request



app = Flask(__name__)

def lr_prediction(loaded,var_1,var_2,var_3,var_4,var_5,var_6,var_7,var_8):
      pred_arr=np.array([var_1,var_2,var_3,var_4,var_5,var_6,var_7,var_8])
      preds=pred_arr.reshape(1,-1)
      #preds=preds.astype(int)
      model_prediction=loaded.predict(preds)
      
      return model_prediction

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    init_features = [float(x) for x in request.form.values()]
    new_record =  [np.array(init_features)]
    print(new_record)
    model= load('lr_model.pkl')
    predict_result = model.predict(new_record)
    print(predict_result)
    # return the result back

    return "<h3> The Housing Price is:- <h3>" +  str(predict_result)

if __name__ == '__main__':

    app.run(host='0.0.0.0')
