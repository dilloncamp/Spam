library(e1071)
library(R.matlab)
library(rhdf5)

traindata <- readMat('spamTrain.mat')
X <- traindata$X
Y <- traindata$y
Xtest <- h5read('spamTest2.mat','Xtest')
ytest <- h5read('spamTest2.mat','ytest')

#Using cost determined in Python
clf <- svm(cost = .03, x=X,y=Y,kernel="linear",scale=FALSE,type="C-classification")
trainPredict <- fitted(clf)
trainingAccuracy <- mean(trainPredict == Y)
pred <- predict(clf, Xtest)
testingAccuracy <- mean(pred == ytest)

#f1 score
tp = pred== 1 & ytest == 1
p = ytest == 1
recall = sum(tp)/sum(p)
fp = pred == 1 & ytest == 0
precision = sum(tp)/(sum(tp)+sum(fp))
f1 = (2*(precision*recall))/(precision+recall)

print(testingAccuracy)
print(f1)