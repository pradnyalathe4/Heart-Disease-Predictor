from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import pickle, io
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

app = FastAPI(title="Heart Disease Prediction API")

MODEL_PATH = "../models/heart_model.pkl"
SCALER_PATH = "../models/heart_scaler.pkl"

FEATURES = [
    'Age','Cholesterol','Blood Pressure','Heart Rate','Exercise Hours',
    'Stress Level','Blood Sugar','Gender','Alcohol Intake','Family History',
    'Diabetes','Obesity','Exercise Induced Angina','Smoking','Chest Pain Type'
]

# ---------------- ENCODING ----------------
def encode(df):
    df['Gender'] = df['Gender'].map({'Male':0,'Female':1})
    df['Alcohol Intake'] = df['Alcohol Intake'].map({'None':0,'Moderate':1,'Heavy':2})
    df['Family History'] = df['Family History'].map({'No':0,'Yes':1})
    df['Diabetes'] = df['Diabetes'].map({'No':0,'Yes':1})
    df['Obesity'] = df['Obesity'].map({'No':0,'Yes':1})
    df['Exercise Induced Angina'] = df['Exercise Induced Angina'].map({'No':0,'Yes':1})
    df['Smoking'] = df['Smoking'].map({'Never':0,'Former':1,'Current':2})
    df['Chest Pain Type'] = df['Chest Pain Type'].map({
        'Atypical Angina':0,
        'Typical Angina':1,
        'Non-anginal Pain':2,
        'Asymptomatic':3
    })
    return df

# ---------------- TRAIN ----------------
@app.post("/train")
async def train(file: UploadFile = File(...)):
    df = pd.read_csv(io.BytesIO(await file.read()))
    df = encode(df)

    X = df[FEATURES]
    y = df['Heart Disease']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )
    model.fit(X_scaled, y)

    pickle.dump(model, open(MODEL_PATH, "wb"))
    pickle.dump(scaler, open(SCALER_PATH, "wb"))

    return {"message": "Model trained successfully"}

# ---------------- EVALUATE ----------------
@app.post("/evaluate")
async def evaluate(file: UploadFile = File(...)):
    if not (MODEL_PATH and SCALER_PATH):
        return {"error": "Train model first"}

    model = pickle.load(open(MODEL_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))

    df = pd.read_csv(io.BytesIO(await file.read()))
    df = encode(df)

    X = df[FEATURES]
    y = df['Heart Disease']

    preds = model.predict(scaler.transform(X))

    report = classification_report(y, preds, output_dict=True)
    acc = accuracy_score(y, preds)

    return {"accuracy": acc, "report": report}

# ---------------- PREDICT ----------------
class Patient(BaseModel):
    Age:int
    Cholesterol:float
    Blood_Pressure:float
    Heart_Rate:float
    Exercise_Hours:float
    Stress_Level:float
    Blood_Sugar:float
    Gender:str
    Alcohol_Intake:str
    Family_History:str
    Diabetes:str
    Obesity:str
    Exercise_Induced_Angina:str
    Smoking:str
    Chest_Pain_Type:str

@app.post("/predict")
def predict(p: Patient):
    model = pickle.load(open(MODEL_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))

    df = pd.DataFrame([{
        "Age":p.Age,
        "Cholesterol":p.Cholesterol,
        "Blood Pressure":p.Blood_Pressure,
        "Heart Rate":p.Heart_Rate,
        "Exercise Hours":p.Exercise_Hours,
        "Stress Level":p.Stress_Level,
        "Blood Sugar":p.Blood_Sugar,
        "Gender":p.Gender,
        "Alcohol Intake":p.Alcohol_Intake,
        "Family History":p.Family_History,
        "Diabetes":p.Diabetes,
        "Obesity":p.Obesity,
        "Exercise Induced Angina":p.Exercise_Induced_Angina,
        "Smoking":p.Smoking,
        "Chest Pain Type":p.Chest_Pain_Type
    }])

    df = encode(df)
    X = df[FEATURES]

    pred = model.predict(scaler.transform(X))[0]
    prob = model.predict_proba(scaler.transform(X)).max()

    return {
        "prediction": int(pred),
        "confidence": round(prob, 2)
    }
