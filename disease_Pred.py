# import necessary libraries
import pickle 
import streamlit as st 
import pandas as pd 
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu

# Loading models 
dia_model = pickle.load(open("Diabetes.sav", 'rb'))
heart_model = pickle.load(open("Heart.sav", 'rb'))


# sidebar for navigation 
with st.sidebar: 

    selected = option_menu('Disease Prediction System.',
                           
                           ['Diabetes Prediction', 
                            'Heart Disease Prediction'],
                            icons = ['activity', 'heart'],
                            default_index = 0)
    
# Diabetes Prediction  Page
if(selected == 'Diabetes Prediction'):

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the inputs data from the user
    col1, col2, col3 = st.columns(3)


    with col1:
        Pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1)

    with col2:
        Glucose = st.number_input("Glucose Level", min_value=0, max_value=300)

    with col3:
        BloodPressure = st.number_input("Blood Pressure value", min_value=0, max_value=200)

    with col1:
        SkinThickness = st.number_input("Skin-Thickness value", min_value=0, max_value=100)

    with col2:
        Insulin = st.number_input("Insulin Level", min_value=0, max_value=900)

    with col3:
        BMI = st.number_input("BMI value", min_value=0.0, max_value=70.0, step=0.1)

    with col1:
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function value", min_value=0.0, max_value=3.0, step=0.01)

    with col2:
        Age = st.number_input("Age of the Person", min_value=1, max_value=120, step=1)


    # code for Prediction 
    diabetes_diagnosis = ''

    # creating a button for Prediction
    if st.button('Diabetes Test Result'): 
        diabetes_prediction = dia_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    
        if(diabetes_prediction[0] == 1): 
            diabetes_diagnosis = 'The person is diabetic'
        else: 
            diabetes_diagnosis = 'The person is not diabetic'


        # graph
        prob = dia_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                Insulin, BMI, DiabetesPedigreeFunction, Age]])[0]
        risk = prob * 100
        st.metric("Diabetes Risk %", f"{risk:.2f}%")

    st.success(diabetes_diagnosis)



# HEART DISEASES PAGE

if(selected == 'Heart Disease Prediction'):

    # page title
    st.title('❤️Heart Disease Prediction using ML')

    # getting the inputs data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120)

    with col2:
        sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])

    with col3:
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])

    with col1:
        trestbps = st.number_input("Resting BP", min_value=50, max_value=200)

    with col2:
        chol = st.number_input("Cholesterol", min_value=100, max_value=500)

    with col3:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

    with col1:
        restecg = st.selectbox("Rest ECG", [0, 1, 2])

    with col2:
        thalach = st.number_input("Max Heart Rate", min_value=50, max_value=250)

    with col3:
        examg = st.selectbox("Exercise Angina", [0, 1])

    with col1:
        oldpeak = st.number_input("Oldpeak (float)", min_value=0.0, max_value=10.0)

    with col2:
        slope = st.selectbox("Slope", [0, 1, 2])

    with col3:
        ca = st.selectbox("CA (Major vessels)", [0, 1, 2, 3, 4])

    with col1:
        thal = st.selectbox("Thal", [1, 2, 3])


    # code for Prediction 
    heart_diagnosis = ''

    # creating a button for Prediction
    if st.button('Heart Disease Test Result'): 

        heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, examg, oldpeak, slope, ca, thal]])[0]
    
        if(heart_prediction == 1): 
            heart_diagnosis = 'The person is having heart disease'
        else:    
            heart_diagnosis = 'The person does not have any heart disease'

        #graph 
        prob = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, examg, oldpeak, slope, ca, thal]])[0]
        risk = prob * 100
        st.metric("Diabetes Risk %", f"{risk:.2f}%")

    st.success(heart_diagnosis)

