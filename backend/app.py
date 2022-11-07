from flask import Flask, request
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import pickle

app = Flask(__name__)


## Import saved model
recipe_df = pd.read_csv("df_recipes.csv")

model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(model_url)

nnModel = pickle.load(open("rec.sav","rb"))

## API  
@app.route('/',methods=['POST'])
def index():

    query = request.get_json()
    print(query['query'])
    
    emb = model([str(query['query'])])
    neighbors = nnModel.kneighbors(emb,return_distance=False)[0]
    return str(recipe_df['recipe_name'].iloc[neighbors].tolist())



if __name__ == "__main__":
    app.run(debug=True)