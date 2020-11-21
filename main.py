def train_model():
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import MinMaxScaler
    import pickle
    from sklearn.metrics import accuracy_score


    df = pd.read_csv('../input/breast-cancer-dataset/dataR2.csv')
    df.drop(['Leptin'], axis = 1, inplace = True)
    y = df['Classification']
    df.drop(['Classification'], axis = 1, inplace = True)
    scaler = MinMaxScaler()
    pre_df = df.copy()
    x = scaler.fit_transform(pre_df)
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(x,y)
    y_pred=clf.predict(x)
    score = accuracy_score(y, y_pred)
    print("Accuracy:",score)
    pickle.dump(clf, open('model.pkl','wb'))