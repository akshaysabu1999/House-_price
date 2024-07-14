
View(data)
summary(data)
dim(data)
str(data)
class(data)
print(head(data))
head(data)
tail(data)


data=read.csv("C:/Users/AKSHAY SABU/Downloads/House price dataset.csv", header=T)
names(data)
attach(data)


sum(is.na(mydata))




#Ridge Regression


# Load necessary libraries
library(glmnet)  # For Lasso and Ridge regression
library(caret)   # For data splitting


# View the first few rows of the dataset
head(data)

# Select relevant features and the target variable
selected_data <- data[, c("Price", "No.of.bedrooms", "No.of.bathrooms", "Renovation.Year", "Built.Year","No.of.floors", "house.condition", "No.of.schools.nearby", "Distance.from.the.airport" )]

# Convert Renovation.Year and Built.Year to numeric if they are not already
selected_data$Renovation.Year <- as.numeric(selected_data$Renovation.Year)
selected_data$Built.Year <- as.numeric(selected_data$Built.Year)

# Create model matrix
X <- model.matrix(Price ~ ., selected_data)[, -1]  # Exclude the intercept term

# Separate the target variable
Y <- selected_data$Price

# Define the lambda sequence
lambda <- 10^seq(10, -2, length = 100)
print(lambda)

# Split the data into training and validation sets
set.seed(567)
part <- sample(2, nrow(X), replace = TRUE, prob = c(0.7, 0.3))
X_train <- X[part == 1, ]
X_cv <- X[part == 2, ]
Y_train <- Y[part == 1]
Y_cv <- Y[part == 2]







# Perform Ridge regression
ridge_reg <- glmnet(X_train, Y_train, alpha = 0, lambda = lambda)
summary(ridge_reg)

# Find the best lambda via cross-validation
ridge_reg1 <- cv.glmnet(X_train, Y_train, alpha = 0)
bestlam_ridge <- ridge_reg1$lambda.min
print(bestlam_ridge)

# Predict on the validation set
ridge_pred <- predict(ridge_reg, s = bestlam_ridge, newx = X_cv)

# Calculate mean squared error
mse_ridge <- mean((Y_cv - ridge_pred)^2)
print(paste("Ridge Mean Squared Error:", mse_ridge))

# Calculate R2 value
sst <- sum((Y_cv - mean(Y_cv))^2)
sse <- sum((Y_cv - ridge_pred)^2)
r2_ridge <- 1 - (sse / sst)
print(paste("Ridge R²:", r2_ridge))

# Get the Ridge regression coefficients
ridge_coef <- predict(ridge_reg, type = "coefficients", s = bestlam_ridge)
print("Ridge Coefficients:")
print(ridge_coef)



#Lasso regression

# Perform Lasso regression
lasso_reg <- glmnet(X_train, Y_train, alpha = 1, lambda = lambda)

# Find the best lambda via cross-validation
lasso_reg1 <- cv.glmnet(X_train, Y_train, alpha = 1)
bestlam_lasso <- lasso_reg1$lambda.min
print(bestlam_lasso)

# Predict on the validation set
lasso_pred <- predict(lasso_reg, s = bestlam_lasso, newx = X_cv)

# Calculate mean squared error
mse_lasso <- mean((Y_cv - lasso_pred)^2)
print(paste("Lasso Mean Squared Error:", mse_lasso))

# Calculate R2 value
sst <- sum((Y_cv - mean(Y_cv))^2)
sse <- sum((Y_cv - lasso_pred)^2)
r2_lasso <- 1 - (sse / sst)
print(paste("Lasso R²:", r2_lasso))

# Get the Lasso regression coefficients
lasso_coef <- predict(lasso_reg, type = "coefficients", s = bestlam_lasso)
print("Lasso Coefficients:")
print(lasso_coef)




#Multiple linear regression

# Load necessary library
library(caret)

# View the first few rows of the dataset
head(data)

# Select relevant features and the target variable
selected_data <- data[, c("Price", "No.of.bedrooms", "No.of.bathrooms", "Renovation.Year", "Built.Year")]

# Convert Renovation.Year and Built.Year to numeric if they are not already
selected_data$Renovation.Year <- as.numeric(selected_data$Renovation.Year)
selected_data$Built.Year <- as.numeric(selected_data$Built.Year)

# Split the data into training and validation sets
set.seed(567)
trainIndex <- createDataPartition(selected_data$Price, p = 0.7, list = FALSE)
train_data <- selected_data[trainIndex, ]
test_data <- selected_data[-trainIndex, ]



# Fit the multiple linear regression model
mlr_model <- lm(Price ~ No.of.bedrooms + No.of.bathrooms + Renovation.Year + Built.Year, data = train_data)

# Summarize the model
summary(mlr_model)



# Predict on the test set
predictions <- predict(mlr_model, newdata = test_data)

# Calculate mean squared error
mse_mlr <- mean((test_data$Price - predictions)^2)
print(paste("Multiple Linear Regression Mean Squared Error:", mse_mlr))

# Calculate R2 value
sst <- sum((test_data$Price - mean(test_data$Price))^2)
sse <- sum((test_data$Price - predictions)^2)
r2_mlr <- 1 - (sse / sst)
print(paste("Multiple Linear Regression R²:", r2_mlr))



# Print the coefficients
print("Multiple Linear Regression Coefficients:")
print(coef(mlr_model))



# Compare the models
model_comparison <- data.frame(
  Model = c("Ridge Regression", "Lasso Regression", "Multiple Linear Regression"),
  MSE = c(mse_ridge, mse_lasso, mse_mlr),
  R2 = c(r2_ridge, r2_lasso, r2_mlr)
)

print(model_comparison)

# Recommend the best model
best_model <- model_comparison[which.min(model_comparison$MSE), "Model"]
print(paste("The best model is:", best_model))




