library(kernlab)

library(MASS)
library(caret)
#library(devtools)
#install_github('caretEnsemble', 'zachmayer') #Install zach's caretEnsemble package
library(caretEnsemble)

#Split train/test
#train <- runif(nrow(X)) <= .66
X <- data.frame(train)
Y <- label_s[,1]
#Setup CV Folds
#returnData=FALSE saves some space
folds=5
repeats=1
myControl <- trainControl(method='cv', number=folds, repeats=repeats, returnResamp='none', 
                          returnData=FALSE, savePredictions=TRUE, 
                          verboseIter=TRUE, allowParallel=TRUE,
                          index=createMultiFolds(Y, k=folds, times=repeats))
PP <- c('center', 'scale')
 
# train the lm
model7 <- train(X, Y, method='glm', trControl=myControl, preProcess=PP)



svm1 <- ksvm(train,label_s,type="C-svc",kernel=sk,C=100,scaled=c())
# General summary
svm1
# Attributes that you can access
attributes(svm1)
# For example, the support vectors
alpha(svm1)
alphaindex(svm1)
b(svm1)
# Use the built-in function to pretty-plot the classifier
plot(svp,data=train)
 
## get fitted values
fitted(svm1)
 
## Test on the training set with probabilities as svm1put
probpreds <- predict(svm1, label, type="probabilities")
## Test on the training set with response as svm1put
responsepreds <- predict(svm1, label, type = "r")

## Compare
results <- data.frame(x = label, 
                      response = responsepreds, 
                      P.x.0 = probpreds[,1], 
                      P.x.1 = probpreds[,2])
 
## Further Comparison

predict(svm1, newdata = label, type = "response")
predict(svm1, newdata = label, type = "decision")
predict(svm1, newdata = label, type = "votes")
predict(svm1, newdata = label, type = "prob")