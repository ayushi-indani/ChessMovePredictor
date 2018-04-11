import numpy as np
import pandas as pd
from sklearn import preprocessing,cross_validation
from sklearn.neighbors import KNeighborsClassifier
import cPickle as pickle
df=pd.read_csv('chess.csv')
#print df.head()
y=np.array(df['label'])
x=np.array(df.drop(['label'],axis=1))
x_train,x_test,y_train,y_test=cross_validation.train_test_split(x,y,test_size=0.2)
#print df_x.head()
clf=KNeighborsClassifier(100)
clf.fit(x_train,y_train)
with open('chessEng.pickle','wb') as file:
	pickle.dump(clf,file)

acc=clf.score(x_test,y_test)
print acc*100
pos=np.array([['1',1,'2',3,'3',2]])
pred=clf.predict(pos)
print pred