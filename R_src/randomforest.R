#Setup
rm(list = ls(all = TRUE)) 
gc(reset=TRUE)
set.seed(42) #From random.org
 
#Libraries
library(caret)
library(devtools)
install_github('caretEnsemble', 'zachmayer') #Install zach's caretEnsemble package
library(caretEnsemble)
 
#Data
library(mlbench)
data(BostonHousing2)
X <- model.matrix(cmedv~crim+zn+indus+chas+nox+rm+age+dis+
                    rad+tax+ptratio+b+lstat+lat+lon, BostonHousing2)[,-1]
X <- data.frame(X)
Y <- BostonHousing2$cmedv
 
#Split train/test
train <- runif(nrow(X)) <= .66
 
#Setup CV Folds
#returnData=FALSE saves some space
folds=5
repeats=1
myControl <- trainControl(method='cv', number=folds, repeats=repeats, returnResamp='none', 
                          returnData=FALSE, savePredictions=TRUE, 
                          verboseIter=TRUE, allowParallel=TRUE,
                          index=createMultiFolds(Y[train], k=folds, times=repeats))
PP <- c('center', 'scale')
 
#Train some models
model1 <- train(X[train,], Y[train], method='gbm', trControl=myControl,
                tuneGrid=expand.grid(.n.trees=500, .interaction.depth=15, .shrinkage = 0.01))
model2 <- train(X[train,], Y[train], method='blackboost', trControl=myControl)
model3 <- train(X[train,], Y[train], method='parRF', trControl=myControl)
model4 <- train(X[train,], Y[train], method='mlpWeightDecay', trControl=myControl, trace=FALSE, preProcess=PP)
model5 <- train(X[train,], Y[train], method='ppr', trControl=myControl, preProcess=PP)
model6 <- train(X[train,], Y[train], method='earth', trControl=myControl, preProcess=PP)
model7 <- train(X[train,], Y[train], method='glm', trControl=myControl, preProcess=PP)
model8 <- train(X[train,], Y[train], method='svmRadial', trControl=myControl, preProcess=PP)
model9 <- train(X[train,], Y[train], method='gam', trControl=myControl, preProcess=PP)
model10 <- train(X[train,], Y[train], method='glmnet', trControl=myControl, preProcess=PP)
 
#Make a list of all the models
all.models <- list(model1, model2, model3, model4, model5, model6, model7, model8, model9, model10)
names(all.models) <- sapply(all.models, function(x) x$method)
sort(sapply(all.models, function(x) min(x$results$RMSE)))
 
#Make a greedy ensemble - currently can only use RMSE
greedy <- caretEnsemble(all.models, iter=1000L)
sort(greedy$weights, decreasing=TRUE)
greedy$error
 
#Make a linear regression ensemble
linear <- caretStack(all.models, method='glm', trControl=trainControl(method='cv'))
summary(linear$ens_model$finalModel)
linear$error
 
#Predict for test set:
preds <- data.frame(sapply(all.models, predict, newdata=X[!train,]))
preds$ENS_greedy <- predict(greedy, newdata=X[!train,])
preds$ENS_linear <- predict(linear, newdata=X[!train,])
sort(sqrt(colMeans((preds - Y[!train]) ^ 2)))