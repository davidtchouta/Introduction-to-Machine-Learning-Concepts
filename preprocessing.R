scores<-c(45, 66,66,55,55,52,61,64,65,49)
mean(scores)
sd(scores)
scale(scores)
plot(scale(scores))
hist(scale(scores))

library(caret)
repeated <- c(rep(100,9999), 10)
random<- sample(100, 10000, T)
data<- data.frame(random=random, repeated=repeated)
nearZeroVar(data)
names(data)[nearZeroVar(data)]


install.packages("mlbench")
install.packages("corrplot")
library(mlbench)
library(corrplot)
library(caret)
data("PimaIndiansDiabetes")

diab<- PimaIndiansDiabetes
corrplot(cor(diab[,-ncol(diab)]), method ="color", type = "upper")
corrplot(cor(diab[,-ncol(diab)]), method ="number", type = "upper")


correlated_col <- findCorrelation(cor(diab[,-ncol(diab)]), cutoff = 0.5)
names(diab)[correlated_col]

#Imbalanced samples
diabsim <- diab
diabrows <- nrow(diabsim)
negrows<-floor(.95*diabrows)
posrows<-(diabrows-negrows)

diabsim$diabetes[1:729] <- as.factor("neg")
diabsim$diabetes[-c(1:729)] <- as.factor("pos")

mean(diabsim$diabetes=="neg")
mean(diabsim$diabetes=="pos")

table(diabsim$diabetes)
#techn1: Upsampled

upsampled <- upSample(diabsim[,-ncol(diabsim)], diabsim$diabetes)
table(upsampled$Class)

#techn2: Downsampled

downsampled <- downSample(diabsim[,-ncol(diabsim)], diabsim$diabetes)
table(downsampled$Class)

library(tidyverse)
##Rename a column
upsampled$diabetes <- upsampled$Class
upsampled$Class <- NULL
table(upsampled$diabetes)

# techn 3 : SMOTE (Synthetic Minority OverSampling Technique)
install.packages(c("zoo","xts","quantmod", "abind", "ROCR"))
install.packages("C:\\Users\\dvid\\Documents\\R_ML\\Livre_practical_big_data_Analytics\\DMwR_0.4.1.tar.gz")
library(DMwR)
rm(diabsyn)
diabsyn <- SMOTE(diabetes ~ ., diabsim, perc.over=500, perc.under=150)
table(diabsyn$diabetes)
mean(diabsyn$diabetes=="pos")
mean(diabsyn$diabetes=="neg")

# techn 4 : ROSE (Randomly OverSampling Examples)
install.packages("ROSE")
library(ROSE)
diabrose <- ROSE(diabetes ~ ., data=diabsim)
table(diabrose$data$diabetes)

##Data imputation (mean, knnimputation and others page 246)
# a chaque fois on calcule le RMSE afin de choisir la meilleure méthode





##########Machine Learning
# importance of variables
training_index <- createDataPartition(diab$diabetes, times=1,p=.8, list=FALSE)
#table(diab$diabetes)
diab_train <- diab[training_index,]
diab_test <- diab[-training_index,]

#create train Control parameters for model
diab_control <- trainControl("repeatedcv", number = 3, repeats = 2, classProbs = TRUE, summaryFunction = twoClassSummary)

# build the model
install.packages("RandomForest")
library(randomForest)
rf_model <- train(diabetes ~ ., data=diab_train, method="rf", preProc=c("center", "scale"), tuneLenght=5,
                  trControl=diab_control, metric="ROC")

varImp(rf_model)
plot(varImp(rf_model))

##predictions
install.packages("e1071")
library(e1071)
predictions <- predict(rf_model, diab_test[-ncol(diab_test)])
head(predictions)

##confusionMatrix
cf<- confusionMatrix(predictions, diab_test$diabetes)
cf
plot(rf_model)
fourfoldplot(cf$table)

#Decision trees
install.packages("rpart")
library(rpart)
install.packages("rpart.plot")
library(rpart.plot)

rpart_model <- rpart(diabetes ~ glucose + insulin + mass + age, data = PimaIndiansDiabetes)

rpart_model
rpart.plot(rpart_model,extra=102, nn=TRUE)

## Random Forest = bagging = sample data set = egal weights
install.packages("randomForest")
library(randomForest)
rf_model1 <- randomForest(diabetes ~ ., data=PimaIndiansDiabetes)
rf_model1

##Boosting algo = assign lower weights for correctly classified and future learner focus on misclassified cases
#Adaboost, GBM(Stochastic Gradient Bossting), xGBoost(most prefered : accurate, fast)
install.packages("xgboost")
library(xgboost)
library(Matrix)
set.seed(123)
train_ind <- sample(nrow(PimaIndiansDiabetes), as.integer(nrow(PimaIndiansDiabetes)*.80))

training_diab <- PimaIndiansDiabetes[train_ind,]
test_diab <- PimaIndiansDiabetes[-train_ind,]

diab_train <- sparse.model.matrix( ~.-1, data=training_diab[,-ncol(training_diab)])
diab_train_dmatrix<- xgb.DMatrix(data=diab_train, label=training_diab$diabetes=="pos")

diab_test <- sparse.model.matrix( ~.-1, data=test_diab[,-ncol(test_diab)])
diab_test_dmatrix<- xgb.DMatrix(data=diab_test, label=test_diab$diabetes=="pos")

param_diab <- list(objective="binary:logistic", eval_metric="error", booster="gbtree", max_depth=5, eta=0.1)

xgb_model <- xgb.train(data = diab_train_dmatrix, param_diab, nrounds = 1000, watchlist = list(train=diab_train_dmatrix,
                      test=diab_test_dmatrix), print_every_n = 10)

predicted <- predict(xgb_model, diab_test_dmatrix)
predicted <- predicted > 0.5
actual <- test_diab$diabetes=="pos"
confusionMatrix(as.factor(actual),as.factor(predicted))

rf_model1

###SVM : 

diab_train <- diab[training_index,]
diab_test <- diab[-training_index,]
install.packages("mlbench")
library(mlbench)
library(e1071)
svm_model <- svm(diabetes ~ ., data=diab_train)
plot(svm_model, diab_train,glucose ~ mass)

svm_predict <- predict(svm_model, diab_test[, -ncol(diab_test)])

confusionMatrix(svm_predict, diab_test$diabetes)

## neural networs
nnet_grid <- expand.grid(.decay=c(0.5,0.1), .size=c(3,5,7))

nnet_model <- train(diabetes ~ ., data = diab_train, method="nnet", metric="Accuracy", maxit=500, tuneGrid=nnet_grid)

plot(nnet_model)
nnet_predicted <- predict(nnet_model, diab_test)
confusionMatrix(nnet_predicted, diab_test$diabetes)

                                            



