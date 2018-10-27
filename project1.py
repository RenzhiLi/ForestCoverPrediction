# import libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier

train_df=pd.read_csv('train.csv')
train=np.array(train_df)
train_x=train[:,1:11]
train_x=preprocessing.scale(train_x)
train_x=np.hstack((train_x[:],train[:,11:-1]))
train_x=np.delete(train_x,[11,15],axis=1)
train_y=train[:,-1]
x_train,x_test,y_train,y_test=train_test_split(train_x,train_y,test_size=0.1,random_state=0)

def singlevalview():
	feature=['Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow','Aspen','Douglas-fir','Krummholz']
	def featuretrans(a):
		return feature[a-1]
	train_df['Cover_Type']=train_df['Cover_Type'].apply(featuretrans)
	plt.subplots(2,5,figsize=(18,10))
	for i in range(1,11):
		plt.subplot(2,5,i)
		group=train_df.groupby([list(train_df.columns)[i],"Cover_Type"])
		s1=group.size().unstack().fillna(0)
		s1.index=[int(x) for x in s1.index]
		s1=s1.sort_index()
		for j in range(7):
			z=np.polyfit(list(s1.index),list(s1[featuretrans(j+1)]),11)
			#plt.plot(s1.index,s1[featuretrans(j+1)],label=feature[j])
			plt.plot(list(s1.index),np.polyval(z,list(s1.index)),label=feature[j])
		plt.xlabel(train_df.columns[i])
		plt.legend()
			
	plt.show()

def binvalview():
	def color(a):
		color=['red','green','yellow','blue','orange','pink','brown']
		return color[a-1]
	train_df['Cover_Type']=train_df['Cover_Type'].apply(color)
	plt.subplots(5,9,figsize=(18,8))
	a=1
	for i in range(1,10):
		for j in range(i+1,11):
			plt.subplot(5,9,a)
			plt.scatter(train_df[list(train_df.columns)[i]],train_df[list(train_df.columns)[j]],s=0.2,c=train_df['Cover_Type'])
			a=a+1
	plt.show()

def dummyview():
	plt.subplots(5,9,figsize=(18,11))
	plt.subplots_adjust(hspace=0.5)
	for i in range(44):
		plt.subplot(5,9,i+1)
		group=train_df.groupby(["Cover_Type",list(train_df.columns)[i+11]])
		s1=group.size().unstack().fillna(0)
		s1.index=[int(x) for x in s1.index]
		s1=s1.sort_index()
		if len(s1.columns)==1:
			plt.bar(s1.index,[0]*7,tick_label=s1.index)
		else:
			plt.bar(s1.index,s1[1],tick_label=s1.index)
		plt.xlabel(train_df.columns[i+11])
		#print(train_df[train_df.columns[i+11]])
	plt.show()
	


rfc=RandomForestClassifier(n_estimators=50,max_features=10,class_weight=None,n_jobs=-1,random_state=0)
rfc.fit(x_train,y_train)
y_predit=rfc.predict(x_test)
print(metrics.accuracy_score(y_test,y_predit))





'''

y=np.vstack((y_test,y_predit))
y=pd.DataFrame(y)
y.to_csv('1.csv')

#71
svmodel=SVC()
svmodel.fit(x_train,y_train)
y_predit=svmodel.predict(x_test)
print(metrics.accuracy_score(y_test,y_predit))

#47
gnb=GaussianNB()
gnb.fit(x_train,y_train)
y_predit=gnb.predict(x_test)
print(metrics.accuracy_score(y_test,y_predit))

#82
mlp=MLPClassifier()
mlp.fit(x_train,y_train)
y_predit=mlp.predict(x_test)
print(metrics.accuracy_score(y_test,y_predit))

from sklearn.ensemble import AdaBoostClassifier
#43
ada=AdaBoostClassifier(n_estimators=100)
ada.fit(x_train,y_train)
y_predit=ada.predict(x_test)
print(metrics.accuracy_score(y_test,y_predit))


#76
gbc=GradientBoostingClassifier(n_estimators=50)
gbc.fit(x_train,y_train)
y_predit=gbc.predict(x_test)
print(metrics.accuracy_score(y_test,y_predit))


#86
rfc=RandomForestClassifier(n_estimators=50)
rfc.fit(x_train,y_train)
y_predit=rfc.predict(x_test)
print(metrics.accuracy_score(y_test,y_predit))

rfc=RandomForestClassifier(n_estimators=50)
mlp=MLPClassifier()
svmodel=SVC(probability=True)
gbc=GradientBoostingClassifier(n_estimators=50)


from sklearn.ensemble import VotingClassifier
vtc=VotingClassifier(estimators=[('rfc',rfc),('mlp',mlp),('gbc',gbc)],voting='hard')
vtc.fit(x_train,y_train)
y_predit=vtc.predict(x_test)
print(metrics.accuracy_score(y_test,y_predit))


lsvc=OneVsRestClassifier(LinearSVC(random_state=0)).fit(x_train,y_train)
y_predit=lsvc.predict(x_test)
print(metrics.accuracy_score(y_test,y_predit))

lsvc=OneVsOneClassifier(LinearSVC(random_state=0)).fit(x_train,y_train)
y_predit=lsvc.predict(x_test)
print(metrics.accuracy_score(y_test,y_predit))
'''
#86



'''
y=np.vstack((y_test,y_predit))
y=pd.DataFrame(y)
y.to_csv('1.csv')
'''