import pickle
import pandas as pd

data = pd.read_csv("creditcard.csv")
data1 = pd.read_csv("creditcard.csv")
print(data.shape)
print(data.head())

numberFrauds = len(data[data["Class"] == 1])
numberNormal = len(data[data["Class"] == 0])

print("Number of frauds",numberFrauds,"\nNumber of Normal transaction",numberNormal,"\n")

from sklearn.preprocessing import StandardScaler 
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))

print(data.head())

x = data.drop(['Time','Amount'], axis = 1)
y = data['Class']
x.drop(['Class'],axis = 1)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = .4, random_state = 0)

print("length of xtrain ",len(xtrain))
print("length of xtest ",len(xtest))
print("length of ytrain ",len(ytrain))
print("length of ytest ",len(ytest))

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix 
from sklearn.metrics import accuracy_score,classification_report

#Train using N_B
gbModel = GaussianNB()
gbModel.fit(xtrain,ytrain)

ypredNB = gbModel.predict(xtest)


print(accuracy_score(ytest,ypredNB))
print(classification_report(ytest,ypredNB))

# Train using SVM
#print("Shashi checking")
from sklearn.svm import SVC

svm = SVC(C = 1, kernel = 'rbf',random_state = 0)
#svm.fit(xtrain,ytrain)

filename = 'finalized_model.sav'
#pickle.dump(svm, open(filename, 'wb'))

svm = pickle.load(open(filename, 'rb'))


ypred  = svm.predict(xtest)
yactual = ytest

accuracy_svm  = svm.score(xtest,ytest)*100
print("accuracy of svm algorithm is " , accuracy_svm)

cm = confusion_matrix(ytest,ypred)
print(cm)
print(accuracy_score(ytest,ypred))
print(classification_report(ytest,ypred))
def confusionMatrix(cm):
    fig, cx = plot_confusion_matrix(conf_mat = cm)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    print("The accuracy is ",((cm[1,1]+cm[0,0])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]))*100,"%")
    print("The recall is ",((cm[1,1])/(cm[1,0]+cm[1,1]))*100)

confusionMatrix(cm)