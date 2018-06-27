import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import time
from dask.distributed import Client


client = Client('tcp://172.17.0.2:8786')
df = pd.read_csv("train.csv")

def demo(df):
    

    X = df.drop(labels = 'Activity',axis = 1)

    y = df['Activity']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25)

    gbm = GradientBoostingClassifier(learning_rate = 0.05,max_features = 106, n_estimators = 300)
    gbm.fit(X_train,y_train)
    result = gbm.predict(X_test)
    score=accuracy_score(y_test,result)
    return score

start = time.time()
big_future = client.scatter(data = df)
output = client.submit(demo,big_future)
outcome=output.result()
# outcome = client.gather(output)
print(outcome)
print("Time_taken:", (time.time()-start)%60)
