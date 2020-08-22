# Hepatitis-Mortality-Prediction

This is a project where you can check the mortality rate with Hepatitis B

### Prerequisites
You must have Scikit Learn, Pandas (for Machine Leraning Model), joblib (to load model) and Streamlit.

### Project Structure
This project major parts are:

- app.py - This contains main file that receives Hepatitis test details through GUI and computes the precited value based on our model and returns it.
- manag_db.py - This file contain database related code, creating table and retriving information
- data - Contain the cleaned data
- train-model/Hepatitis-Mortality-prediction.ipynb - This file contain all the code about EDA, Feature selection, Model evaluation etc and finally save the model.
- model - This folder contains all the train model ready to use

### Running the project 
1. Download the zip file
2. Go to project folder and run `pip install -r requirements.txt`
3. Run `streamlit run app.py`
