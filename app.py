import gradio
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Response

save_file_name = "xgboost-model.pkl"
model = joblib.load(save_file_name)


# FastAPI object
app = FastAPI()

# Function for prediction
def predict_death_event(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, 
                        high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time):
    # Create a DataFrame from user inputs
    input_data = pd.DataFrame([{
        "age": age,
        "anaemia": anaemia,
        "creatinine_phosphokinase": creatinine_phosphokinase,
        "diabetes": diabetes,
        "ejection_fraction": ejection_fraction,
        "high_blood_pressure": high_blood_pressure,
        "platelets": platelets,
        "serum_creatinine": serum_creatinine,
        "serum_sodium": serum_sodium,
        "sex": sex,
        "smoking": smoking,
        "time": time
    }])
    # Predict using the trained model
    prediction = model.predict(input_data)
    
    # Optional: Return a human-readable result
    return "Death Event" if prediction[0] == 1 else "No Death Event"

# Gradio interface to generate UI link
title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

iface = gradio.Interface(fn = predict_death_event,
                        inputs=[
                                    gradio.Number(label="Age"),
                                    gradio.Radio([0, 1], label="Anaemia (0=No, 1=Yes)"),
                                    gradio.Number(label="Creatinine Phosphokinase"),
                                    gradio.Radio([0, 1], label="Diabetes (0=No, 1=Yes)"),
                                    gradio.Number(label="Ejection Fraction"),
                                    gradio.Radio([0, 1], label="High Blood Pressure (0=No, 1=Yes)"),
                                    gradio.Number(label="Platelets"),
                                    gradio.Number(label="Serum Creatinine"),
                                    gradio.Number(label="Serum Sodium"),
                                    gradio.Radio([0, 1], label="Sex (0=Female, 1=Male)"),
                                    gradio.Radio([0, 1], label="Smoking (0=No, 1=Yes)"),
                                    gradio.Number(label="Follow-up Time")
                                ],
                        outputs = gradio.Textbox(type="text", label='Prediction', elem_id="out_textbox"),
                         title = title,
                         description = description,
                         allow_flagging='never'
                        )

# Mount gradio interface object on FastAPI app at endpoint = '/'
app = gradio.mount_gradio_app(app, iface, path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)