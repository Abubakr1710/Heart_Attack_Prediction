# import libraries
import numpy as np
from joblib import load
#import argparse

model = load('./data/best_model.joblib')
scaler = load('./data/scaler.joblib')

features = ['age','sex','cp','trtbps','chol','fbs','restecg','thalachh','exng','oldpeak','slp','caa','thall']
# features_dic = {'age':'age', 'sex':'sex', 'cp':'Chest Pain type', 'trtbps':'Resting blood pressure (in mm Hg)',\
#     'chol':'Cholestoral in mg/dl fetched via BMI sensor','fbs':'(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)',\
#         'restecg':'Resting electrocardiographic results','thalachh':'maximum heart rate achieved',\
#             'exng':'Exercise induced angina (1 = yes; 0 = no)','oldpeak':'oldpeak','slp':'slp','caa':'caa','thall':'thall'}

# while True:
#     age = int(input("How old are you? \n"))
#     sex = int(input("Gender? 0 for Female, 1 for Male \n"))
#     cp = int(input("Chest pain type? 0 for Absent, 1 for light pain, 2 for moderate pain, 3 for extreme pain \n"))
#     trtbps = int(input("Resting blood pressure in mm Hg \n"))
#     chol = int(input("Serum cholestrol in mg/dl \n"))
#     fbs = int(input("Fasting Blood Sugar? 0 for < 120 mg/dl, 1 for > 120 mg/dl \n"))
#     restecg = int(input("Resting ecg? (0,1,2) \n"))
#     thalachh = int(input("Maximum Heart Rate achieved? \n"))
#     exng = int(input("Exercise Induced Angina? 0 for no, 1 for yes \n"))
#     oldpeak = float(input("Old Peak? ST Depression induced by exercise relative to rest \n"))
#     slp = int(input("Slope of the peak? (0,1,2) \n"))
#     caa = int(input("Number of colored vessels during Floroscopy? (0,1,2,3) \n"))
#     thall = int(input("thal: 3 = normal; 6 = fixed defect; 7 = reversable defect \n"))


# parser = argparse.ArgumentParser()
# for item in features:
#     parser.add_argument(item, type=float, help=item)

# args = parser.parse_args()
x_features = [int(input("How old are you? \n")),int(input("Gender? 0 for Female, 1 for Male \n")),int(input("Chest pain type? 0 for Absent, 1 for light pain, 2 for moderate pain, 3 for extreme pain \n")),int(input("Resting blood pressure in mm Hg \n")),int(input("Serum cholestrol in mg/dl \n")),int(input("Fasting Blood Sugar? 0 for < 120 mg/dl, 1 for > 120 mg/dl \n")),int(input("Resting ecg? (0,1,2) \n")),int(input("Maximum Heart Rate achieved? \n")),int(input("Exercise Induced Angina? 0 for no, 1 for yes \n")),float(input("Old Peak? ST Depression induced by exercise relative to rest \n")),int(input("Slope of the peak? (0,1,2) \n")),int(input("Number of colored vessels during Floroscopy? (0,1,2,3) \n")),int(input("thal: 3 = normal; 6 = fixed defect; 7 = reversable defect \n")) ]

# get and transform data
data = np.array(x_features).reshape(1, -1)
data = scaler.transform(data)


#predict with saved model
prediction = model.predict(data)

deci = {1:'YOU HAVE A HEART ATTACK', 0:'You don\'t have Heart Attack'}
print("\n{}\n".format(deci[int(prediction[0])]))

