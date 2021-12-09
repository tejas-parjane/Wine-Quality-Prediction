import pandas as pd

df = pd.read_csv('winequality-red.csv')

X = df.drop(['quality'],axis=1)
y =  df['quality']

#print(X.shape)
#print(y.shape)
#print(X.isnull())
#print(X.info())

#for i in X:
#    print(X[i].describe())

from matplotlib import pyplot as plt
#for i in X:
#   plt.title(i)
#     plt.scatter(range(X.shape[0]),X[i])
#     plt.show()
     
#plt.scatter(range(X.shape[0]), y)
#plt.show()


#print(df.corr()) 

 #free sulphur dioxide 
 #total sulphur dioxide

X = X.drop(['pH', 'free sulfur dioxide', 'total sulfur dioxide',
           'residual sugar', 'chlorides', 'density', 'fixed acidity'], axis=1)
#print(X.shape)


#import seaborn as sb   

#dataplot = sb.heatmap(X.corr(), cmap="YlGnBu", annot=True)
#plt.show()
#print(X)
from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()
X = scaler.fit_transform(X)

y = pd.get_dummies(y)
#chlorides
#density
#print(df.shape)
#print(X.shape)
#print(y.shape)
from sklearn.model_selection import train_test_split as tts
X_train,X_test,y_train,y_test = tts(X,y,test_size=0.2,random_state=40)

#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)
