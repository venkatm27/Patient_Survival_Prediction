# pull python base image
FROM python:3.10

# specify working directory
WORKDIR /Patient_Survival_Prediction

ADD /Patient_Survival_Prediction/requirements.txt .
ADD /Patient_Survival_Prediction/app.py .
ADD /Patient_Survival_Prediction/xgboost-model.pkl .

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt



# # copy application files
# ADD /titanic_model_api/app/* ./app/

# expose port for application
EXPOSE 8001

# start fastapi application
CMD ["python", "app.py"]
