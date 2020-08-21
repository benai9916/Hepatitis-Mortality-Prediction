# load packages
import streamlit as st 

# EDA package
import pandas as pd 
import numpy as np 


# Utils
import os
import joblib 
import hashlib
# passlib,bcrypt
from PIL import Image

# Data Visualization Pkgs
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')

# DB
from manage_db import *

# Password 
def generate_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()


def verify_hashes(password,hashed_text):
	if generate_hashes(password) == hashed_text:
		return hashed_text
	return False


best_feature =  ['ascites', 'bilirubin', 'albumin', 'spiders','alk_phosphate','protime','age','varices','malaise','histology']

feature_dict = {"No":1,"Yes":2}


def get_fvalue(val):
	feature_dict = {"No":1,"Yes":2}
	for key,value in feature_dict.items():
		if val == key:
			return value 

# Load ML Models
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


html_temp = """
		<div style="background-color:{};padding:10px;border-radius:10px;margin-bottom:30px">
		<h1 style="color:white;text-align:center;">Hepatitis Mortality Prediction</h1>
		<h5 style="color:white;text-align:center;">Hepatitis B </h5>
		</div>
		"""

prescriptive_message_temp ="""
	<div style="background-color:lightgray;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Recommended Life style modification</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		<li style="text-align:justify;color:black;padding:10px">Get Plenty of Rest</li>
		<li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		<li style="text-align:justify;color:black;padding:10px">Avoid Alchol</li>
		<li style="text-align:justify;color:black;padding:10px">Proper diet</li>
		<ul>
		<h3 style="text-align:justify;color:black;padding:10px">Medical Mgmt</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Consult your doctor</li>
		<li style="text-align:justify;color:black;padding:10px">Take your interferons</li>
		<li style="text-align:justify;color:black;padding:10px">Go for checkups</li>
		<ul>
	</div>
	"""


descriptive_message_temp ="""
	<div style="background-color:lightgray;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;margin-bottom:20px">
		<h3 style="text-align:justify;color:black;padding:10px; padding-left:0px">Definition</h3>
		<p>Hepatitis B is a viral infection that attacks the liver and can cause both acute and chronic disease.</p>
	</div>
	"""

@st.cache
def load_image(img):
	im = Image.open(os.path.join(img))
	return im


def main():
	"""Hep Mortality Prediction App"""
	st.markdown(html_temp.format('purple'),unsafe_allow_html=True)

	menu = ["Home","Login","Signup"]
	sub_menu = ["Plot","Prediction"]

	choice = st.sidebar.selectbox("Menu",menu)
	if choice == "Home":
		st.subheader("Home")
		st.markdown(descriptive_message_temp,unsafe_allow_html=True)
		
		st.image(load_image('image/hepatitis.jpg'))


	elif choice == "Login":
		username = st.sidebar.text_input("Username")
		password = st.sidebar.text_input("Password",type='password')
		if st.sidebar.checkbox("Login"):
			create_usertable()
			hashed_pswd = generate_hashes(password)
			result = login_user(username,verify_hashes(password,hashed_pswd))

			if result:
				st.success("Welcome {}".format(username))

				activity = st.selectbox('Select Action',sub_menu)
				if activity == "Plot":
					st.subheader("DataFrame")
					df = pd.read_csv("data/clean_hepatitis.csv")
					st.dataframe(df)

					st.header("Count of Live and Death")
					st.subheader("1 = Die, 2 = Lives")
					df['class'].value_counts().plot(kind='bar')
					plt.xlabel('Live vs Death')
					plt.ylabel('Count')
					st.pyplot()

					# Freq Dist Plot
					freq_df = pd.read_csv("data/age_group_infection_rate.csv")
					freq_df.plot(kind='bar', title='Age Group count')
					plt.xlabel('Count')
					plt.ylabel('Age')
					st.pyplot()

					sns.countplot(x='sex', hue='class', data=df)
					st.pyplot()


					if st.checkbox("Area Chart"):
						all_columns = df.columns.to_list()
						feat_choices = st.multiselect("Choose a Feature",all_columns)
						new_df = df[feat_choices]
						st.area_chart(new_df)
						


				elif activity == "Prediction":
					st.subheader('Predictive Analysis')

					age = st.number_input("Age",7,80)
					malaise = st.radio("Do You Have Malaise",tuple(feature_dict.keys()))
					spiders = st.radio("Presence of Spider Naeve",tuple(feature_dict.keys()))
					ascites = st.selectbox("Ascities",tuple(feature_dict.keys()))
					varices = st.selectbox("Presence of Varices",tuple(feature_dict.keys()))
					bilirubin = st.number_input("Bilirubin Content",0.0,8.0)
					alk_phosphate = st.number_input("Alkaline Phosphate Content",0.0,296.0)
					sgot = st.number_input("Sgot",0.0,648.0)
					albumin = st.number_input("Albumin",0.0,6.4)
					protime = st.number_input("Prothrombin Time",0.0,100.0)

					feature_list = [age,get_fvalue(malaise),get_fvalue(spiders),get_fvalue(ascites),get_fvalue(varices),bilirubin,alk_phosphate,sgot,albumin,int(protime)]
					st.write(len(feature_list))
					st.write(feature_list)
					pretty_result = {"age":age,"malaise":malaise,"spiders":spiders,"ascites":ascites,"varices":varices,"bilirubin":bilirubin,"alk_phosphate":alk_phosphate,"sgot":sgot,"albumin":albumin,"protime":protime}
					
					# show the field option in json formate
					st.json(pretty_result)
					single_sample = np.array(feature_list).reshape(1,-1)

					# Machine learnig prediction
					model_choice = st.selectbox("Select Model",["LogisticRegression","DecisionTree", "RandomForest", "Support Vector"])
					if st.button("Predict"):
						if model_choice == "LogisticRegression":
							loaded_model = load_model("models/logistic_regression.pkl")
							prediction = loaded_model.predict(single_sample)
							pred_prob = loaded_model.predict_proba(single_sample)

						elif model_choice == "DecisionTree":
							loaded_model = load_model("models/decision_tree_clf_model.pkl")
							prediction = loaded_model.predict(single_sample)
							pred_prob = loaded_model.predict_proba(single_sample)

						elif model_choice == "RandomForest":
							loaded_model = load_model("models/Random_forest_model.pkl")
							prediction = loaded_model.predict(single_sample)
							pred_prob = loaded_model.predict_proba(single_sample)

						else:
							loaded_model = load_model("models/support_vector_model.pkl")
							prediction = loaded_model.predict(single_sample)
							pred_prob = loaded_model.predict_proba(single_sample)


						if prediction == 1:
							st.warning("Patient Dies")
							pred_probability_score = {"Die":pred_prob[0][0]*100,"Live":pred_prob[0][1]*100}
							st.subheader("Prediction Probability Score using {}".format(model_choice))
							st.json(pred_probability_score)
							st.subheader("Prescriptive Analytics")
							st.markdown(prescriptive_message_temp,unsafe_allow_html=True)
							
						else:
							st.success("Patient Lives")
							pred_probability_score = {"Die":pred_prob[0][0]*100,"Live":pred_prob[0][1]*100}
							st.subheader("Prediction Probability Score using {}".format(model_choice))
							st.json(pred_probability_score)
					

			else:
				st.warning("Incorrect Username/Password")


	else:
		new_username = st.text_input("User name")
		new_password = st.text_input("Password", type='password')

		confirm_password = st.text_input("Confirm Password",type='password')

		if st.button("Submit"):
			if new_password == confirm_password:

				create_usertable()
				hashed_new_password = generate_hashes(new_password)
				add_userdata(new_username,hashed_new_password)
				st.success("You have successfully created a new account")
				st.info("Login to Get Started")

			else:
				st.warning("Passwords does not match..!!")



if __name__ == '__main__':
	main()