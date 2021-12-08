import pandas as pd
from sklearn.model_selection import train_test_split

def get_data(pth):

    data = pd.read_csv(pth)
    X = data.values[:,:-1]
    y = data.values[:,:-1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
    
    return x_train, x_test, y_train, y_test