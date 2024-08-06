# Load required libraries
library(neuralnet)

# Load the dataset
data <- read.csv("C:/Users/Asus/Desktop/ML CW/ExchangeUSDML.csv")

# Rename columns for convenience
colnames(data) <- c("Date", "Weekday", "USD_EUR")

# Extract test and train data
train_data <- data[1:400, "USD_EUR"]
test_data <- data[401:500, "USD_EUR"]

# Apply normalization - min-max
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

train_data_normalized <- normalize(train_data)
test_data_normalized <- normalize(test_data)

# Function to create lagged dataset
create_lagged_dataset <- function(data, lag) {
  lagged_data <- embed(data, lag + 1)[, 1:(lag + 1)]
  colnames(lagged_data) <- c("USD_EUR", paste0("lag", 1:lag))
  return(lagged_data)
}

# Create lagged datasets
lag1_train <- create_lagged_dataset(train_data_normalized, 1)
lag2_train <- create_lagged_dataset(train_data_normalized, 2)
lag3_train <- create_lagged_dataset(train_data_normalized, 3)
lag4_train <- create_lagged_dataset(train_data_normalized, 4)

# Train MLP models
model_lag1 <- neuralnet(USD_EUR ~ ., data = as.data.frame(lag1_train), hidden = c(5, 3), linear.output = TRUE)
model_lag2 <- neuralnet(USD_EUR ~ ., data = as.data.frame(lag2_train), hidden = c(5, 3), linear.output = TRUE)
model_lag3 <- neuralnet(USD_EUR ~ ., data = as.data.frame(lag3_train), hidden = c(5, 3), linear.output = TRUE)
model_lag4 <- neuralnet(USD_EUR ~ ., data = as.data.frame(lag4_train), hidden = c(5, 3), linear.output = TRUE)

# Predict using each model
lag1_test <- create_lagged_dataset(test_data_normalized, 1)
lag2_test <- create_lagged_dataset(test_data_normalized, 2)
lag3_test <- create_lagged_dataset(test_data_normalized, 3)
lag4_test <- create_lagged_dataset(test_data_normalized, 4)

predictions_lag1 <- predict(model_lag1, as.data.frame(lag1_test))
predictions_lag2 <- predict(model_lag2, as.data.frame(lag2_test))
predictions_lag3 <- predict(model_lag3, as.data.frame(lag3_test))
predictions_lag4 <- predict(model_lag4, as.data.frame(lag4_test))

# Function to denormalize predictions
denormalize <- function(x, original_data) {
  return(x * (max(original_data) - min(original_data)) + min(original_data))
}

# Denormalize predictions for each lag
denormalized_predictions_lag1 <- denormalize(predictions_lag1, data$USD_EUR)
denormalized_predictions_lag2 <- denormalize(predictions_lag2, data$USD_EUR)
denormalized_predictions_lag3 <- denormalize(predictions_lag3, data$USD_EUR)
denormalized_predictions_lag4 <- denormalize(predictions_lag4, data$USD_EUR)


# Function to calculate evaluation metrics
evaluation_metrics <- function(predictions, actual) {
  # Ensure predictions and actual have the same length
  min_length <- min(length(predictions), length(actual))
  predictions <- predictions[1:min_length]
  actual <- actual[1:min_length]
  
  # Root Mean Squared Error
  rmse <- sqrt(mean((predictions - actual)^2))
  
  # Mean Absolute Error
  mae <- mean(abs(predictions - actual))
  
  # Mean Absolute Percentage Error
  mape <- mean(abs((actual - predictions) / actual)) * 100
  
  # Symmetric Mean Absolute Percentage Error
  smape <- mean(2 * abs(predictions - actual) / (abs(predictions) + abs(actual))) * 100
  
  return(c(RMSE = rmse, MAE = mae, MAPE = mape, sMAPE = smape))
}

# Function to evaluate models for each lag separately
evaluate_models_for_each_lag <- function(model, lag_test_data, original_test_data) {
  predictions <- predict(model, as.data.frame(lag_test_data))
  denormalized_predictions <- denormalize(predictions, original_test_data)
  evaluation_metrics(denormalized_predictions, original_test_data)
}

# Evaluate models for each lag separately
evaluation_lag1 <- evaluate_models_for_each_lag(model_lag1, lag1_test, test_data)
evaluation_lag2 <- evaluate_models_for_each_lag(model_lag2, lag2_test, test_data)
evaluation_lag3 <- evaluate_models_for_each_lag(model_lag3, lag3_test, test_data)
evaluation_lag4 <- evaluate_models_for_each_lag(model_lag4, lag4_test, test_data)

# Print evaluation metrics for each lag separately
cat("Evaluation Metrics for Lag 1:\n")
print(evaluation_lag1)
cat("\n")

cat("Evaluation Metrics for Lag 2:\n")
print(evaluation_lag2)
cat("\n")

cat("Evaluation Metrics for Lag 3:\n")
print(evaluation_lag3)
cat("\n")

cat("Evaluation Metrics for Lag 4:\n")
print(evaluation_lag4)
cat("\n")

# Create a comparison table
comparison_table <- data.frame(
  Model = c("Lag 1", "Lag 2", "Lag 3", "Lag 4"),
  RMSE = c(evaluation_lag1["RMSE"], evaluation_lag2["RMSE"], evaluation_lag3["RMSE"], evaluation_lag4["RMSE"]),
  MAE = c(evaluation_lag1["MAE"], evaluation_lag2["MAE"], evaluation_lag3["MAE"], evaluation_lag4["MAE"]),
  MAPE = c(evaluation_lag1["MAPE"], evaluation_lag2["MAPE"], evaluation_lag3["MAPE"], evaluation_lag4["MAPE"]),
  sMAPE = c(evaluation_lag1["sMAPE"], evaluation_lag2["sMAPE"], evaluation_lag3["sMAPE"], evaluation_lag4["sMAPE"]),
  Description = c("Input vector: Lag 1, Hidden layers: (5,3)", 
                  "Input vector: Lag 2, Hidden layers: (5,3)", 
                  "Input vector: Lag 3, Hidden layers: (5,3)", 
                  "Input vector: Lag 4, Hidden layers: (5,3)")
)

# Print comparison table
print("Comparison Table:")
print(comparison_table)


# Choose the model with the lowest RMSE
best_model <- model_lag1  # Assuming lag1 model is the best based on RMSE

# Get predictions for test data using the best model
best_predictions <- predict(best_model, as.data.frame(lag1_test))
denormalized_best_predictions <- denormalize(best_predictions, test_data)

# Plot predicted output vs. desired output
plot(test_data, type = "l", col = "blue", xlab = "Index", ylab = "Exchange Rate", main = "MLP Predicted vs. Desired Output")
lines(denormalized_best_predictions, col = "red")
legend("topright", legend = c("Desired Output", "Predicted Output"), col = c("blue", "red"), lty = 1)

