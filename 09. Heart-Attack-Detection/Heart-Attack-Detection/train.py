# import libraries
import data_handler as dh
from joblib import dump
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# get data
data_path = './data/heart.csv'
x_train, x_test, y_train, y_test = dh.get_data(data_path)

#Normalize data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#fit different models
model = svm.SVC()
#model = GradientBoostingClassifier()
#model = RandomForestClassifier()
model.fit(x_train,y_train)

#check initial accuracy
acc_train = model.score(x_train,y_train)
acc_test = model.score(x_test,y_test)
y_pred = model.predict(x_test)
print(" ")
print("Train Accuracy: {},\tTest Accuracy: {}".format(acc_train,acc_test))
print("F1 Score: {},\tPrecision  Score: {},\tRecall Score: {}".format( f1_score(y_test, y_pred),\
     precision_score(y_test, y_pred), recall_score(y_test, y_pred) ) )

#save model
dump(model, './data/best_model.joblib')
dump(scaler, './data/scaler.joblib')

#plot confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
disp.plot()
plt.show()
