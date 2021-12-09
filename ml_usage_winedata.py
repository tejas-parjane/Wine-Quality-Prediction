from case_study_wine import X_train, X_test, y_train, y_test

from sklearn.svm import SVC as m 
model = m()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
y_test = list(y_test.values)

correct = 0
total = len(y_pred)

for i in range(total):
    if y_pred[i] == y_test[i]:
        correct += 1
        
print(100*correct/total)

