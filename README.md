---
# Boston Project
### Samantha Knox


## Introduction

This project analyses the Boston data set. It concerns the median value of housing in the Boston Standard Metropolitan Statistical Area in 1970 and related variables such as socio-economic conditions, environmental conditions, educational facilities and some other similar factors.

There are 506 observations in the data for 14 variables including the natural logarithm of the per capita crime rate by town in Boston. There are 12 numerical variables in our dataset and 1 categorical variable. The aim of this project is to build and interpret linear regression models for predicting the natural logarithm of the per capita crime rate (LCRIM) in terms of the other variables. 

```{r include=FALSE}
#Partition 80% of the data into a training and test set
data(Boston, package="nclSLR", type="source")

index <- sample(nrow(Boston),nrow(Boston)*0.80)
train <- Boston[index,]
test <- Boston[-index,]
```
## Variable Description

The 14 variables included within the data are: 

 - LCRIM - Natural logarithm of the per capita crime rate by town.
 
 - ZN - Proportion of residential land zoned for lots over 25,000 sq.ft.
 
 - INDUS - Proportion of non-retail business acres per town
 
 - CHAS - Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
 
 - NOX - Nitrogen oxides concentration (parts per 10 million)
 
 - RM - Average number of rooms per dwelling
 
 - AGE - Proportion of owner-occupied units built prior to 1940
 
 - DISF - A numerical vector representing an ordered categorical variable with four levels depending on the weighted mean of the distances to five Boston employment centres (=1 if distance < 2.5, =2 if 2.5 <= distance < 5, =3 if 5 <= distance < 7.5, =4 if distance >= 7.5).
 
 - RAD - Index of accessibility to radial highways
 
 - TAX - Full-value property-tax rate per $10,000
 
 - PTRATIO - Pupil-teacher ratio by town
 
 - B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
 
 - LSTAT - % lower status of the population
 
 - MEDV - Median value of owner-occupied homes in $1000's
 
## Data Exploration

Firstly, the summary statistics and distribution of the observations in each variable were identified. There are no missing values in the data. However the summary statistics suggest that there might be outliers in some of the variables.
```{r include=FALSE}
summary(train)
head(Boston)
```

```{r include=FALSE}
pairs.panels(Boston[c(-1,-4)], cex=2)
```

#### Boxplot

In order to identify variation in the variables, a boxplot was used. This showed that there are outliers in the variables ‘LSTAT’, 'MEDV', ‘ZN’ and most distinctly, ‘BLACK’. Highest variability is observed in the full-value property tax rates.

```{r, echo=FALSE}
ggplot(stack(train), aes(x = ind, y = values)) + geom_boxplot() + theme(axis.line = element_line(colour = "black", size = 0.25)) + geom_boxplot(fill = "cyan", colour = "#1F3552", outlier.colour = "red")
```

#### Correlation Matrix

Numerous predictors (INDUS, NOX, AGE, DISF, RAD, TAX & LSTAT)	have	high	correlation	with the	outcome	variable.

```{r, echo=FALSE, include=FALSE}
corr <- round(cor(train), 1)
ggcorrplot(corr, hc.order = TRUE, type = "lower", lab = TRUE,
   outline.col = "white",
   ggtheme = ggplot2::theme_gray,
   colors = c("cyan", "white", "#1F3552"))
```

```{r, echo=FALSE}
corr <- round(cor(train), 1)
corrplot(cor(train))
```

Linear regression was performed on the dataset. The variable of our interest here is ‘LCRIM’. Our main objective is to predict the value of ‘LCRIM’ based on other independent variables present in the dataset. 

## Best Model Identification
### Subset Selection
```{r, include=FALSE}
set.seed(123)
fit <- lm(lcrim~.,data = train)
sum.fit <- summary(fit)
print(sum.fit)
```

From the summary statistics of the model (see Appendix) we observe that RAD, NOX and ZN have the most significance as they have the lowest p values ((Pr(>|t|)<0.001). The coefficient estimate of ZN variable is negative indicating that the value of response variable LCRIM decreases as the percentage of people with residential land zoned for lots over 25,000sq ft increases. 


```{r, include=FALSE}
set.seed(123)
fit.mse <- (sum.fit$sigma)^2
fit.rsq <- sum.fit$r.squared
fit.arsq <- sum.fit$adj.r.squared
test.pred.fit <- predict(fit, newdata=test) 
fit.mpse <- mean((test$lcrim-test.pred.fit)^2)
fit.aic <- AIC(fit)
fit.bic <- BIC(fit)

#stats.fit <- c("FULL", fit.mse, fit.rsq, fit.arsq, fit.mpse, fit.aic, fit.bic)
#comparison_table <- c("model type", "MSE", "R-Squared", "Adjusted R-Squared", "Test MSPE", "AIC", "BIC")
#data.frame(cbind(comparison_table, stats.fit))
```

We will now apply different techniques of variable selection to decide the best fit model.

### Best Subset Selection
We wish to predict the crime rate by town based on the available predictors by applying the best subset selection approach. We decide on the number of variables that are required in the model using BIC, r squared and adjusted r squared.

```{r, include=FALSE}
regs_model <- regsubsets(lcrim~., train,nvmax = 13)
regs_summ <- summary(regs_model)
cbind(regs_summ$which,bic=regs_summ$bic,adj.r2=regs_summ$adjr2)
```

The BIC values in the Subset Selection table (see Appendix) suggests that the model with 6 variables is the best model, however the adjusted r squared suggest the model with 10 variables is the best. As training data has been used to fit the model, model selection using these metrics is possibly subject to overfitting and may not perform as well when applied to new data.
A better approach would be to compare the model stats with the stats obtained from the full model in the previous section. 

```{r, echo=FALSE}
model.ss <- lm(lcrim ~ . -indus -age, data=train)
sum.model.ss <- summary(model.ss)
sum.model.ss
```

```{r, echo=FALSE}
model.ss.mse <- (sum.model.ss$sigma)^2
model.ss.rsq <- sum.model.ss$r.squared
model.ss.arsq <- sum.model.ss$adj.r.squared
test.pred.model.ss <- predict(model.ss, newdata=test) 
model.ss.mpse <- mean((test$lcrim-test.pred.model.ss)^2)
modelss.aic <- AIC(model.ss)
modelss.bic <- BIC(model.ss)

stats.model.ss <- c("best_subset", model.ss.mse, model.ss.rsq, model.ss.arsq, model.ss.mpse, modelss.aic, modelss.bic)

#data.frame(cbind(comparison_table, stats.fit, stats.model.ss))
```

### Forward Selection
Forward stepwise selection is a computationally efficient alternative to best subset selection. This model has same goal as the best subset method, but the algorithm and variables selected are different. First a model is created with no predictors. Then a predictor is added to the model which increases its adjusted r square. Next another predictor is added to further increase our adjusted R square. This process is continued till the r squared reaches the maximum value.
```{r, echo=FALSE}
set.seed(123)
nullmodel <- lm(lcrim~1, data = train)
fullmodel <- lm(lcrim~., data = train)

model.step.f<- step(nullmodel, scope=list(lower=nullmodel, upper=fullmodel), direction='forward', trace=FALSE)
summary(model.step.f)

#Comparison Table
#model.step.f.mse <- (sum.model.step.f$sigma)^2
#model.step.f.rsq <- sum.model.step.f$r.squared
#model.step.f.arsq <- sum.model.step.f$adj.r.squared
#test.pred.model.step.f <- predict(model.step.f, newdata=test) 
#model.step.f.mpse <- mean((test$lcrim-test.pred.model.step.f)^2)
#modelstep.f.aic <- AIC(model.step.f)
#modelstep.f.bic <- BIC(model.step.f)

#stats.model.step.f <- c("best_subset", model.step.f.mse, model.step.f.rsq, model.step.f.arsq, model.step.f.mpse, modelstep.f.aic, modelstep.f.bic)

#data.frame(cbind(comparison_table, stats.fit, stats.model.step.f))
```

### Backward Elimination

Like forward stepwise selection, backward stepwise selection provides an efficient alternative to best subset selection. However, unlike forward stepwise selection, it begins with the full least squares model containing all p predictors, and then iteratively removes the least useful predictor, one-at-a-time. This process is continued till the r squared reaches the maximum value.

```{r, echo=FALSE}
#backward elimination
set.seed(123)
model.step.b<- step(fullmodel,direction='backward',trace=FALSE)
summary(model.step.b)
```

### Stepwise Selection

Stepwise selection has the flexibility to both add/remove variables in the process of selecting the best regression model. 

```{r, echo=FALSE}
#stepwise(mixed)
set.seed(123)
model.step.s<- step(nullmodel, scope=list(lower=nullmodel, upper=fullmodel), direction='both',trace=FALSE)
summary(model.step.s)
```

Subset Selection: 0.8709
Best Subset Selection: 0.871
Forward Selection: 0.8739
Backward Selection: 0.8717
Stepwise Selection: 0.8717

Looking at the adjusted R^2 results, Forward Selection appears to be the best model, which considers 7 variable; NOX being the most significant. 

### Ridge Regression
Ridge regression does not perform variable selection. However, it 'shrinks' the coefficients towards zero. Repeated cross-validation will be used, where CV = 10, therefore the model will be made from 9 parts and 1 part will be used for error estimation. This is reapeated 10 times with a different part used for error estimation each time. It will be repeated 5 times. 

```{r, include=FALSE}
set.seed(1234)
custom <- trainControl(method = "repeatedcv",
                       number = 10,
                       repeats = 5,
                       verboseIter = T)

#Linear Model
set.seed(1234)
lm <- train(lcrim ~ .,
            train,
            method = 'lm',
            trControl = custom)
```
```{r, include=FALSE}
#Results
lm$results
summary(lm)
plot(lm$finalModel)
```

RMSE (Root Mean Squared Error) = 0.782

R^2 = 0.874, i.e. more than 87% of variability seen in response is because of the model. 

```{r, include=FALSE}
#Ridge Regression
set.seed(1234)
ridge <- train(lcrim ~ .,
               train,
               method = 'glmnet',
               tuneGrid = expand.grid(alpha = 0,
                                      lambda = seq(0.0001, 1, length=5)),
               trControl = custom)

```  
Fitting alpha = 0, the best value of lambda = 1e-04 on full training set. 
For higher values of lambda, error increases. 

```{r, include=FALSE}
plot(ridge)
```
```{r, echo=FALSE}
plot(ridge$finalModel, xvar = "lambda", label = T)
```

When lambda is approximately 8, all coefficients are close to zero. At every point, we have all 13 independent variables in the model. 

```{r, echo=FALSE}
plot(ridge$finalModel, xvar = 'dev', label = T)
```

After fractional deviance of 0.8, the coefficients become inflated and we start to see overfitting. 

```{r, echo=FALSE}
#Variable Importance
plot(varImp(ridge, scale = F))
```

Plotting variable importance shows that NOX is the most important variable 

### The LASSO
We will now use Lasso (least absolute shrinkage and selection operator) to build the regression model. This method shrinks regression coefficients, with some shrunk to zero. Thus, it helps with feature selection. 

```{r, include=FALSE}
#LASSO
set.seed(1234)
lasso <- train(lcrim ~ .,
               train,
               method = 'glmnet',
               tuneGrid = expand.grid(alpha = 1,
                                      lambda = seq(0.0001, 1, length =5)),
               trControl = custom)
```

Fitting alpha = 1, the best value of lambda = 1e-04 on full training set, so we get a similar output to Ridge Regression.

```{r, include=FALSE}
#Plot Results
plot(lasso)
```
```{r, echo=FALSE}
plot(lasso$finalModel, xvar = 'lambda', label = T)
plot(lasso$finalModel, xvar = 'dev', label = T)
```
```{r, echo=FALSE}
plot(varImp(lasso, scale = T))
```

### Elastic Net Regression

```{r, include=FALSE}

set.seed(1234)
en <- train(lcrim ~ .,
            train,
            method = 'glmnet',
            tuenGrid = expand.grid(lpha = seq(0,1, length = 10),
                                   lambda = seq(0.001, 0.2, length = 5)),
            trControl = custom)

```
Fitting alpha = 1, the best value of lambda = 0.0383 on full training set

## Comparing Model Performances

```{r, include=FALSE}
plot(en)
```
```{r, include=FALSE}
#Plot Results
plot(en$finalModel, xvar = 'lambda', label=T)
plot(en$finalModel, xvar = 'dev', label=T)
plot(varImp(en))
```
```{r, echo=FALSE}
#Compare Models
model_list <- list(LinearModel = lm, Ridge = ridge, Lasso = lasso, ElasticNet = en)
res <- resamples(model_list)
summary(res)
```
```{r, include=FALSE}
bwplot(res)
xyplot(res, metric = 'RMSE')


#Select best model
en$bestTune
best <- en$finalModel
coef(best, s= en$bestTune$lambda)

#Prediction
p1 <- predict(en, train)
sqrt(mean((train$lcrim)^2-p1))

p2 <- predict(en, train)
sqrt(mean((test$lcrim)^2-p2))
```
Ridge Regression: 1e-04

LASSO: 1e-04

Elastic Net Regression: 0.0383


Comparing the lambda values for the regression models, which measures how well observed outcomes are replicated by the model, the most closely fitted models are the Ridge Regression model and the LASSO, which perform similarly. Ridge Regression considers 13 variables, whereas lambda is only close to zero when considering 4 or 5 in LASSO. This may suggest that Ridge Regression may be overfitting and LASSO underfitting, with the true best selection of variables somewhere in between. 
Both the best-performing selection methods and regression methods identify NOX (Nitrogen oxides concentration (parts per 10 million)) as the most influential variable for LCRIM (Natural logarithm of the per capita crime rate by town), which may indicate that areas with higher levels of air pollution (such a busy city centres) have a much higher crime rate than more rural areas. Subset Selection supports this, as it indicates that crime rates are lower in areas of large plots of land (ZN). This would also make sense if crimes such as car theft/damage were included, where the number of vehicles that contribute to air pollution are located. 



```{r, include=FALSE}

library(psych)
library(mlbench)
library(caret)
library(recipes)
library(rcompanion)
```




