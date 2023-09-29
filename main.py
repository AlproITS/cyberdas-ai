#!/usr/bin/env python
# encoding: utf-8

from fastapi import FastAPI, Form, Depends, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pickle
import json

app = FastAPI()

# Menentukan direktori templates
templates = Jinja2Templates(directory="templates")

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


def form_req(
        age: str = Form(...),
        sex: str = Form(...),
        smoker: str = Form(...),
        bmi: str = Form(...),
        children: str = Form(...),
        region: str = Form(...)
):
    return Req(age=age, sex=sex, smoker=smoker, bmi=float(bmi), children=int(children), region=int(region))


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


@app.post("/predict")
async def predict(request: Request, requess: Req = Depends(form_req)):
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

    prediction = model.predict([data])
    output = round(prediction[0], 2)

    # Merender index.html dengan data hasil prediksi
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "insurance_cost": output,
            "age": requess.age,
            "sex": "Laki-laki" if requess.sex == 1 else "Perempuan",
            "smoker": "Ya" if requess.smoker == 1 else "Tidak",
            "bmi": requess.bmi,  # Menambahkan ini
            "children": requess.children,  # Menambahkan ini
            "region": get_region_name(requess.region)  # Menambahkan ini
        }
    )

def get_region_name(region_code):
    region_mapping = {
        0: "Northeast",
        1: "Northwest",
        2: "Southeast",
        3: "Southwest"
    }
    return region_mapping.get(region_code, "Unknown")

