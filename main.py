import http.server
import socketserver
PORT = 8000
Handler = http.server.CGIHTTPRequestHandler
httpd = socketserver.TCPServer(("", PORT), Handler)
httpd.serve_forever()

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
def get_form():
    #!C:\Python27\python.exe
    # Import modules for CGI handling
    import cgi, cgitb
    # Create instance of FieldStorage
    form = cgi.FieldStorage()
    # Get data from fields
    first_name = form.getvalue('height')
    last_name  = form.getvalue('weight')
    print("Content-type:text/html")
    print("")
    print("")
    print("Hello - Second CGI Program")
    print("")
    print("")
    print("Hello %s %s" % (first_name, last_name))
    print("")
    print("")

get_form()