from keras import layers
from case_study_wine import X_train, X_test, y_train, y_test

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10,input_dim=4,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,input_dim=4,activation='relu'))
model.add(Dense(6,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=100,batch_size=4)

_,accuracy = model.evaluate(X_test,y_test)
print(accuracy*100)
