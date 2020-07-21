set.seed(1)
Detroit = read.csv("detroit_data.csv")
Detroit = Detroit[, 4:31]
Detroit = Detroit[which(!is.na(Detroit$TotGrad)),]
Detroit = na.omit(Detroit)
names(Detroit)
train = sample(0.65*nrow(Detroit), rep = FALSE)
library(glmnet)
grid = 10^seq(10,-2, length=100)
X = model.matrix(PctEnrl ~ ., Detroit)[, -1]
Y = Detroit$PctEnrl
lasso.mod =glmnet(X[train,], Y[train], alpha=1, lambda=grid)
par(mfrow =c(1, 2))
plot(lasso.mod)# plot w/ L1 norm
plot(lasso.mod, xvar = "lambda", label = TRUE, xlim = c(-5, 0))
set.seed(1)
cv.out =cv.glmnet(X[train,], Y[train], alpha=1)
plot(cv.out)
bestlam = cv.out$lambda.min
lasso.coef =predict(lasso.mod, type="coefficients", s=bestlam)
lasso_vars_lm = lm(PctEnrl ~ TotEnrl + FivePlus_Tot + PctUndr5ER +  PctOvr5ER  + CntTested + EBLL + Under6EBLL + Under18CntTested + PctUnder6EBLL, data = Detroit[train,])
lasso.pred_train =predict(lasso.mod,s=bestlam,newx=X[train,])
mean((lasso.pred_train-Y[train])^2)
lasso.pred_test = predict(lasso.mod, s = bestlam, newx = X[-train,])
mean((lasso.pred_test - Y[-train])^2)
lm.pred.train = predict.lm(lasso_vars_lm, Detroit[train,])
lm.train.mse = mean((lm.pred.train - Detroit$PctEnrl[train])^2)
lm.pred.test = predict.lm(lasso_vars_lm, Detroit[-train,])
lm.test.mse = mean((lm.pred.test - Detroit$PctEnrl[-train])^2)
plot(lasso_vars_lm)
Detroit$EnrlLvl = factor(ifelse(Detroit$PctEnrl < 0.35, "low", ifelse(Detroit$PctEnrl < 0.65, "medium", "high")))
plot(Detroit$EnrlLvl, xlab = "Enrollment Level", ylab = "Number of Observations", main = "Number of Observations with Each Enrollment Level")
library("FNN")
Detroit_train = Detroit[train,]
Detroit_test = Detroit[-train,]
mean_train = colMeans(Detroit_train[,1:28])
std_train = sqrt(diag(var(Detroit_train[,1:28])))
X_Detroit_train = scale(Detroit_train[,1:28], center = mean_train, scale = std_train)
X_Detroit_train = X_Detroit_train[,c(1:12, 16:18, 20:28)]
Y_Detroit_train = Detroit_train$EnrlLvl
X_Detroit_test = scale(Detroit_test[,1:28], center = mean_train, scale = std_train)
X_Detroit_test = X_Detroit_test[,c(1:12, 16:18, 20:28)]
Y_Detroit_test = Detroit_test$EnrlLvl
Detroit_pred = knn(train = X_Detroit_train, test = X_Detroit_test, cl = Y_Detroit_train, k = 10)
mean(Detroit_pred != Y_Detroit_test)
conf_table = table(Detroit_pred, Y_Detroit_test)
set.seed(1)
k_to_try = 1:length(Detroit_train)
err_Detroit_train = rep(0, times = length(k_to_try))
err_Detroit_test = rep(0, times = length(k_to_try))
for(i in 1:length(k_to_try))
{
    pred_train = knn(train = X_Detroit_train, cl = Y_Detroit_train, test = X_Detroit_train, k = k_to_try[i])
    pred_train = factor(pred_train, levels = levels(Y_Detroit_train))
    err_Detroit_train[i] = mean(Y_Detroit_train != pred_train)
    pred_test = knn(train = X_Detroit_train, cl = Y_Detroit_train, test = X_Detroit_test, k = k_to_try[i])
    pred_test = factor(pred_test, levels = levels(Y_Detroit_train))
    err_Detroit_test[i] = mean(Y_Detroit_test != pred_test)    
}
plot(k_to_try, err_Detroit_train, type = "b", col = "blue", cex = 1, pch = 20,xlab = "Number of neighbors (K)", ylab = "classification error",ylim =c(0,1),main = "Classification error for different K")
lines(k_to_try, err_Detroit_test, type = "b", lwd = 2, col = "red")
legend("bottomright", legend =c("training error", "test error"),col =c("blue", "red"), cex = .75, lwd =c(2, 2), pch =c(1, 1), lty =c(1, 1))
plot(Detroit[-train,]$PctAsthma, Detroit[-train,]$PctUndr5ER, col = Detroit$EnrlLvl, xlab = "Percent Asthma", ylab = "Percentage Under 5 ER", main = "True class vs Predicted class by Logistic Regression (Testing Data)")
points(Detroit$PctAsthma, Detroit$PctUndr5ER, pch = c(4,14)[pred_test])
legend("bottomright", c("True Level = low","True Level = medium", "True Level = high", "Pred Level = low", "Pred Level = medium", "Pred Level = high"), 
col=c("orange", "blue", "black", "black"), pch=c(1,1,4,14))
ggplot(Detroit[-train], aes(x = Detroit[train,]$PctAsthma, y = Detroit[train,]$PctUndr5ER)) +
    geom_point(aes(color=Detroit[train,]$EnrlLvl), shape = pred_train)+
    labs(title="PctAsthma vs. PctUndr5ER", x="Asthma", y = "Under 5", color = "Levels") +
    theme_minimal() +
    geom_jitter(width = 0.005, height = 0.005, aes(color = Detroit[train,]$EnrlLvl), shape = pred_train)
min(err_Detroit_test)
err_Detroit_test
log1 = glm(PctEnrl ~ TotEnrl + FivePlus_Tot + PctUndr5ER +  PctOvr5ER  + CntTested + EBLL + Under6EBLL + Under18CntTested + PctUnder6EBLL, data = Detroit[train,], family = binomial)
summary(log1)
plot(log1)
pred = predict(log1, Detroit[-train,])
predProbs = binomial()$linkinv(pred)
testPrediction = rep("high", nrow(Detroit[-train,]))
testPrediction[predProbs < 0.65] = "medium"
testPrediction[predProbs < 0.35] = "low"
table(testPrediction, Detroit[-train,]$EnrlLvl, dnn =c("Predicted", "Actual"))
round(mean(testPrediction != Detroit[-train,]$EnrlLvl),2)
simple = lm(PctEnrl ~ PctUndr5ER, data = Detroit[train,])
simple_pred = predict.lm(simple, Detroit[train,])
mean((simple_pred - Detroit[train,]$PctEnrl)^2)
plot(Detroit$PctUndr5ER, Detroit$PctEnrl, xlab = "Percentage Enrollment", ylab = "Percentage Under 5 ER", xlim = c(0,1), ylim = c(0.1,0.9), main = "Percentage Enrollment vs Percentage Under 5 ER")
abline(a = simple$coefficients[1], b = simple$coefficients[2], col = 'blue')
abline(a = lasso_vars_lm$coefficients[1], b = lasso_vars_lm$coefficients[2], col = 'blue')
