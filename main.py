#!/usr/bin/env python
# encoding: utf-8

from fastapi import FastAPI, Form, Depends
from pydantic import BaseModel
import pickle
import json

app = FastAPI()


model_file = open('insurance_model.pkl', 'rb')
model = pickle.load(model_file, encoding='bytes')

class Msg(BaseModel):
    msg: str

class Req(BaseModel):
    age: int
    sex: int
    smoker: int
    bmi: float
    children: int
    region: int

@app.get("/")
async def root():
    return {"message": "Hello World. Welcome to FastAPI!"}

def form_req(age: str = Form(...), sex: str = Form(...), smoker: str = Form(...)):
    return Req(age=age, sex=sex, smoker=smoker, bmi=20.0, children=0, region=0)

@app.get("/path")
async def demo_get():
    return {"message": "This is /path endpoint, use a post request to transform the text to uppercase"}


@app.post("/path")
async def demo_post(inp: Msg):
    return {"message": inp.msg.upper()}


@app.get("/path/{path_id}")
async def demo_get_path_id(path_id: int):
    return {"message": f"This is /path/{path_id} endpoint, use post request to retrieve result"}

@app.get("/predict/{path_id}")
async def predict(path_id: int):
     return {"message":  f"This is /predict/{path_id} endpoint, use post request to retrieve result"}

taxonomies = [["Normal", "EBO non dysplasique", "EBO en dysplasie de bas grade", "EBO en dysplasie de haut grade", "adénocarcinome superficiel", "adénocarcinome invasif", "Dysplasie de haut grade sur muqueuse épidermoïde", "Carcinome épidermoïde superficiel", "Carcinome épidermoïde invasif"],
              ["Normal", "Adénome", "Ampullome bénin", "ampullome dégénéré superficiel", "ampullome dégénéré invasif"],
              ["Normal", "polype glandulokystique", "atrophie", "métaplasie intestinale", "dysplasie de bas grade", "dysplasie de haut grade", "adénocarcinome superficiel", "adénocarcinome invasif", "pancréas aberrant", "GIST"],
              ["Normal", "Polype hyperplasique", "Adénome festonné sessile", "adénome en dysplasie de bas grade", "adénome en dysplasie de haut grade", "adénocarcinome superficiel", "adénocarcinome invasif"]]

anatomyName = ["Oesophage","Duodénum","Estomac" , "Colon rectum"]

relation_taxo = [[[1,4,4,4,4,4,4,4,4],[4,1,2,4,4,4,4,4,4], [4,2,1,3,4,4,4,4,4], [4,4,3,1,3,4,4,4,4], [4,4,4,3,1,3,4,4,4], [4,4,4,4,3,1,4,4,4], [4,4,4,4,4,4,1,2,3], [4,4,4,4,4,4,2,1,4], [4,4,4,4,4,4,3,4,1]],
                  [[1,4,4,4,4], [4,1,4,4,4], [4,4,1,3,4], [4,4,3,1,3], [4,4,4,3,1]],
                  [[1,4,4,4,4,4,4,4,4,4], [4,1,4,4,4,4,4,4,4,4], [4,4,1,2,2,4,4,4,4,4], [4,4,2,1,2,3,4,4,4,4], [4,4,2,2,1,2,4,4,4,4], [4,4,4,3,2,1,3,4,4,4], [4,4,4,4,4,3,1,3,4,4], [4,4,4,4,4,4,3,1,4,4], [4,4,4,4,4,4,4,4,1,4], [4,4,4,4,4,4,4,4,4,1]],
                  [[1,2,4,4,4,4,4], [2,1,2,4,4,4,4], [4,2,1,4,2,4,4], [4,4,4,1,1,4,4], [4,4,2,1,1,2,4], [4,4,4,4,2,1,3], [4,4,4,4,4,3,1]]]  

feedback = ["", "Les taxonomies peuvent être confondus", "Erreur de conformité (les taxonomies ont une texture et une forme visuelle proche)", "Erreur grave (l'erreur entre deux taxonomies peut causer un risque pour le patient)", "Inacceptable"]

@app.get("/taxonomy/{anatomy}&{prediction}&{anapath}")
async def giveFeedback(anatomy: str, prediction: str, anapath: str):
    #Find index
    anatomy_idx = anatomyName.index(anatomy)
    prediction_idx = taxonomies[anatomy_idx].index(prediction)
    anapath_idx = taxonomies[anatomy_idx].index(anapath)

    #Result
    result = feedback[relation_taxo[anatomy_idx][prediction_idx][anapath_idx]]
    jsonStr = json.dumps(result, ensure_ascii=False)
    return {jsonStr}    

@app.get("/taxonomy/{anatomy}")
async def taxonomyList(anatomy: str):
    jsonStr = json.dumps(taxonomies[anatomyName.index(anatomy)], ensure_ascii=False)
    return {jsonStr}

@app.get("/anatomyList")
async def anatomyList():
    jsonStr = json.dumps(anatomyName, ensure_ascii=False)
    return {jsonStr}

@app.post("/predict")
async def predict(requess: Req = Depends(form_req)):
    '''
    Predict the insurance cost based on user inputs
    and render the result to the html page
    '''
    age = requess.age
    sex = requess.sex
    smoker = requess.smoker
    bmi = requess.bmi
    children = requess.children
    region = requess.region
    data = []

    data.append(int(age))
    data.extend([int(sex)])
    data.extend([float(bmi)])
    data.extend([int(children)])
    data.extend([int(smoker)])
    data.extend([int(region)])  
    # if sex == 'Male':
    #     data.extend([0, 1])
    # else:
    #     data.extend([1, 0])

    # if smoker == 'Yes':
    #     data.extend([0, 1])
    # else:
    #     data.extend([1, 0])
    
    prediction = model.predict([data])
    output = round(prediction[0], 2)
    return {"message": f"Your annual insurance is: {output} USD"}        
    
#     #return render_template('index.html', insurance_cost=output, age=age, sex=sex, smoker=smoker)