library(kernlab)

# train the SVM
svm1 <- ksvm(train,label_senti,type="C-svc",kernel=sk,C=100,scaled=c())
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
 
## Test on the training set with probabilities as output
predict(svm1, label, type="probabilities")
 
