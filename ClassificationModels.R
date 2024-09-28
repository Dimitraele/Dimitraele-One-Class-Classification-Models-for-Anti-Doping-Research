# Load required libraries
library(keras)
library(ggplot2)
library(e1071)
library(caret)
library(ROSE)
library(dplyr)
library(tidyr)

# Set working directory and load data
setwd("")
# Import the dataset
data <- read.csv("data.csv", header = TRUE)

# View the dimensions of the data
dim(data)

# Assign names to the elements
Ion <- paste0("Ion_", 1:dim(data)[1])
data$ID <- Ion
data <- cbind(ID=data[,111], data[,-111])

# Correction: Turn 0s to a very small value
data[data == 0] <- runif(sum(data == 0), min = 0.1, max = 0.5)

# Randomly select 14 samples to be the test set including the true positive samples
set.seed(1234)
testsamples <- sample(c(4:(dim(data)[2]-2)), size=14, replace=FALSE)
test_pos <- c(testsamples, c(6,110))

# Prepare training data
train_data <- as.data.frame(t(data[,-c(1:3, test_pos)]))  # excluding test data
train_data <- log(train_data)
colnames(train_data) <- as.character(data$ID)
rownames(train_data) <- 1:92

# Class variable for training data
train_Class <- rep(TRUE, times=92)

# Prepare test data
test_data <- as.data.frame(t(data[,c(test_pos)]))  
test_data <- log(test_data)
colnames(test_data) <- as.character(data$ID)
rownames(test_data) <- 1:16

# Class variable for test data
test_Class <- c(rep(TRUE, times=14), rep(FALSE, times=2))   # TRUE IS INLIER (NORMAL), FALSE IS OUTLIER (ABNORMAL)

# Duplicate the true positive samples
test_data <- rbind(test_data, test_data[15,], test_data[16,], test_data[15,], test_data[16,])
test_Class <- c(test_Class, FALSE, FALSE, FALSE, FALSE)

# Data oversampling
test_Class2 <- ifelse(test_Class==TRUE, "negative", "positive")
overs_test_set <- data.frame(test_data, Class=test_Class2)

oversample <- function(data) {
  data_balanced <- ovun.sample(Class ~ ., data = data, method = "over", p=0.5)$data
  return(data_balanced)
}

data_oversampled <- oversample(overs_test_set)

# Prepare test data and class labels
test_data <- data_oversampled[, -ncol(data_oversampled)]  # remove class
Class <- data_oversampled[, ncol(data_oversampled)]
test_Class <- ifelse(Class == "negative", TRUE, FALSE)


##################################
# 1. One class classification SVM
##################################

# Train the One-Class SVM model
svm_model <- svm(train_data, 
                 type = "one-classification",   # for novelty detection
                 nu = 0.1,                      # Adjust 'nu' based on your needs
                 scale = TRUE,
                 kernel = "radial")             # Use radial kernel for better results

# Make predictions on test data
all_predictions <- predict(svm_model, test_data)

# Create confusion matrix
pred <- factor(all_predictions, levels = c(TRUE, FALSE), 
               labels = c("negative", "positive"))
true <- factor(test_Class, levels = c(TRUE, FALSE), 
               labels = c("negative", "positive"))
conf_matrix <- confusionMatrix(pred, true, positive = "positive")

# Print classification metrics
metrics <- conf_matrix$byClass %>% as.data.frame() %>% 
           mutate(Measure = rownames(.))
print(metrics)

# Extract additional metrics
accuracy <- conf_matrix$overall["Accuracy"]
print(paste("Accuracy:", accuracy))

# Visualize the confusion matrix
confusion_plot <- as.data.frame(conf_matrix$table) %>%
  ggplot(aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "white", high = "blue") +
  geom_text(aes(label = Freq), color = "black") +
  labs(title = "Confusion Matrix", x = "True Class", y = "Predicted Class") +
  theme_minimal()
print(confusion_plot)

# Define the filename and path where you want to save the PNG file
output_file <- "confusion_matrix_plot_OCC_SVM.png"

# Save the plot using ggsave
ggsave(filename = output_file, plot = confusion_plot, width = 6, height = 4, dpi = 300)

# Optionally, you can also display the plot
print(confusion_plot)


# Load necessary library
library(ggplot2)
library(reshape2)

# Identify the duplicated rows (excluding the first occurrence)
duplicates <- duplicated(test_data[,c(1,2)]) | duplicated(test_data[,c(1,2)], fromLast = TRUE)

# Add a small random value to the first column of the duplicated rows
test_data2 <- test_data
test_data2[rep(TRUE, 27), 1] <- test_data2[rep(TRUE, 27), 1] + 
                                runif(sum(rep(TRUE, 27)), 
                                min =  1, max =  12)

length(test_data2[duplicates, 1])  # 13 OUTLIERS

# Visualize the predictions and anomalies
anomalies <- test_data2[all_predictions == FALSE, ]

test_data_with_Pred_Class <- cbind(test_data2,
                                    Prediction=all_predictions) 

# Prepare data for ggplot
test_data_long <- melt(test_data_with_Pred_Class, variable.name = "Ion", value.name = "Value")
test_data_long$Class <- ifelse(test_Class, "Negative", "Positive")

# Combine test data and anomalies for plotting
combined_data <- rbind(
  data.frame(test_data_long)
)

combined_data$Prediction <- ifelse(combined_data$Prediction, "Negative", "Positive")


# Create plots for Ion comparisons
plot_list <- lapply(1:1, function(i) {
  ggplot(combined_data[combined_data$Ion %in% c(paste0("Ion_", i)), ], 
         aes_string(x = "Value", y = "Class", color = "Class")) +
    geom_point(alpha = 0.6, size = 5.5) + 
    geom_point(data = combined_data[combined_data$Prediction == "Positive" & combined_data$Ion %in% c(paste0("Ion_", i)),], 
               aes(x = Value, y = Class), 
               color = "gold", shape = 19, size = 3, alpha = 0.8) +  # Gold points for anomalies
    geom_point(aes(x = Inf, y = Inf, color = "Anomalies"), shape = 19, size = 3) +  # Dummy point for legend
    labs(title = "OCC-SVM Anomaly Detection",
         x = "Measurement Value of Ion 1", 
         y = "True Class") +
    scale_color_manual(values = c("Negative" = "black", 
                                  "Positive" = "red", 
                                  "Anomalies" = "gold")) +  # Add "Anomalies" to the color scale
    theme_minimal(base_size = 15) +
    theme(legend.title = element_blank(),
          legend.position = "top")
})

# Arrange plots in a grid
do.call(grid.arrange, c(plot_list, ncol = 1))


### plot_list2
combined_data2 <- combined_data[combined_data$Ion == "Ion_1",]
combined_data2$Value2 <- c(1:13, 1:14)

plot_list2 <- lapply(1:1, function(i) {
  ggplot(combined_data2, 
         aes_string(x = "Value2", y = "Class", color = "Class")) +
    geom_point(alpha = 0.6, size = 5.5) + 
    geom_point(data = combined_data2[combined_data2$Prediction == "Positive",], 
               aes(x = Value2, y = Class), 
               color = "gold", shape = 19, size = 3, alpha = 0.8) +  # Gold points for anomalies
    geom_point(aes(x = Inf, y = Inf, color = "Anomalies"), shape = 19, size = 3) +  # Dummy point for legend
    labs(title = "OCC-SVM Anomaly Detection",
         x = "Sample", 
         y = "True Class") +
    scale_color_manual(values = c("Negative" = "black", 
                                  "Positive" = "red", 
                                  "Anomalies" = "gold")) +  # Add "Anomalies" to the color scale
    scale_x_continuous(breaks = seq(min(combined_data2$Value2), max(combined_data2$Value2), by = 1)) +  # Show all numbers on the x-axis
    theme_minimal(base_size = 15) +
    theme(legend.title = element_blank(),
          legend.position = "top")
})

# Arrange plots in a grid
do.call(grid.arrange, c(plot_list2, ncol = 1))

# Load required library
library(gridExtra)

# Define the filename and path where you want to save the PNG file
output_file <- "Classification_plot_OCC_SVM.png"

# Save the plots using ggsave. We use arrangeGrob to combine plots before saving.
g <- do.call(grid.arrange, c(plot_list2, ncol = 1))  # Combine plots into a single object

# Save the combined plot
ggsave(filename = output_file, plot = g, width = 6, height = 6, dpi = 300)

# Optionally, you can display the plot
print(g)

# Additional metrics
sensitivity <- conf_matrix$byClass["Sensitivity"]
specificity <- conf_matrix$byClass["Specificity"]
g_mean <- sqrt(sensitivity * specificity)

# Print additional metrics
print(paste("Sensitivity:", sensitivity))
print(paste("Specificity:", specificity))
print(paste("G-mean:", g_mean))

# Store performance metrics
performance_metrics <- list(
  accuracy = accuracy,
  precision = conf_matrix$byClass["Precision"],
  recall = conf_matrix$byClass["Recall"],
  f1_score = conf_matrix$byClass["F1"],
  sensitivity = sensitivity,
  specificity = specificity,
  g_mean = g_mean
)

# Print performance metrics for all models
print(performance_metrics)

# Identify the duplicated rows (excluding the first occurrence)
duplicates <- duplicated(test_data[,c(1,2)]) | duplicated(test_data[,c(1,2)], fromLast = TRUE)

# Add a small random value to the first column of the duplicated rows
test_data2 <- test_data
test_data2[duplicates, 1] <- test_data2[duplicates, 1] + runif(sum(duplicates), 
                                                               min =  2, max =  4)
test_data2[duplicates, 2] <- test_data2[duplicates, 2] + runif(sum(duplicates), 
                                                               min =  2, max =  4)
test_data2[duplicates, 3] <- test_data2[duplicates, 3] + runif(sum(duplicates), 
                                                               min = 2, max = 4)
test_data2[duplicates, 4] <- test_data2[duplicates, 4] + runif(sum(duplicates), 
                                                               min =  2, max =  4)
test_data2[duplicates, 5] <- test_data2[duplicates, 5] + runif(sum(duplicates), 
                                                               min =  2, max =  4)
length(test_data[duplicates, 1])  # 18 OUTLIERS

# Extract anomalies from the test data
anomalies <- test_data2[all_predictions == FALSE, ]


# Create a data frame for ggplot
plot_data <- data.frame(
  Ion1 = test_data2[, 1],
  Ion2 = test_data2[, 2],
  Class = ifelse(test_Class, "Negative", "Positive"),
  Anomaly = ifelse(all_predictions == FALSE, "Anomaly", "Non-Anomaly")
)

# Create the plot
performance_text <- paste(
  "Accuracy: ", round(performance_metrics$accuracy, 2), "\n",
  "Precision: ", round(performance_metrics$precision, 2), "\n",
  "Recall: ", round(performance_metrics$recall, 2), "\n",
  "F1 Score: ", round(performance_metrics$f1_score, 2), "\n",
  "Sensitivity: ", round(performance_metrics$sensitivity, 2), "\n",
  "Specificity: ", round(performance_metrics$specificity, 2), "\n",
  "G-Mean: ", round(performance_metrics$g_mean, 2)
)

# Define the filename and path where you want to save the PNG file
output_file_2 <- "svm_anomaly_detection_plot_Ion1_Ion2.png"

# Create the plot
svm_plot <- ggplot(plot_data, aes(x = Ion1, y = Ion2)) +
  geom_point(aes(color = Class, shape = Anomaly), size = 3, alpha = 0.7) +
  scale_color_manual(values = c("Negative" = "black", "Positive" = "red")) +
  scale_shape_manual(values = c("Non-Anomaly" = 19, "Anomaly" = 17)) + # Triangle for anomalies
  labs(title = "SVM Anomaly Detection",
       x = "Ion 1",
       y = "Ion 2",
       color = "Class",
       shape = "Prediction") +
  theme_minimal(base_size = 15) +
  theme(legend.position = "bottom") +
  guides(shape = guide_legend(override.aes = list(size = 4))) +  # Increase size of legend points
  annotate("text", x = max(plot_data$Ion1) * 0.8, y = max(plot_data$Ion2) * 0.8, 
           label = performance_text, hjust = 0, vjust = 1.3, size = 6, 
           color = "blue", 
           family = "serif", fontface = "italic")

# Save the plot
ggsave(filename = output_file_2, plot = svm_plot, width = 9, height = 9, dpi = 300)

# Optionally, you can display the plot
print(svm_plot)

# Set the output file for saving the plots
output_file_3 <- "svm_anomaly_detection_visualization.png"

# Open a PNG device
png(filename = output_file_3, width = 800, height = 800)


# Visualize the test data and anomalies
par(mfrow = c(2, 2))
plot(test_data2[, c(1, 2)], col = ifelse(test_Class == TRUE, "black", "red"), 
     cex=1.5, pch = 19, main = "SVM Anomaly Detection")
points(anomalies[, c(1, 2)], col = "gold", pch = 19, cex=0.8)
legend("bottomright", legend = c("Negative", "Positive", "Predicted as anomalies"), 
       col = c("black", "red", "gold"), 
       pch = c(19, 19), cex=0.75)

plot(test_data2[, c(1, 3)], col = ifelse(test_Class == TRUE, "black", "red"), 
     cex=1.3, pch = 19, main = "SVM Anomaly Detection")
points(anomalies[, c(1, 3)], col = "gold", pch = 19, cex=0.8)
legend("topleft", legend = c("True negative", "True positive", "Predicted anomalies"), 
       col = c("black", "red", "gold"), 
       pch = c(19, 19), cex=0.75)

plot(test_data2[, c(1, 4)], col = ifelse(test_Class == TRUE, "black", "red"), 
     cex=1.3, pch = 19, main = "SVM Anomaly Detection")
points(anomalies[, c(1, 4)], col = "gold", pch = 19, cex=0.8)
legend("topleft", legend = c("True negative", "True positive", "Predicted anomalies"),  
       col = c("black", "red", "gold"), 
       pch = c(19, 19), cex=0.75)

plot(test_data2[, c(1, 5)], col = ifelse(test_Class == TRUE, "black", "red"), 
     cex=1.3, pch = 19, main = "SVM Anomaly Detection")
points(anomalies[, c(1, 5)], col = "gold", pch = 19, cex=0.8)
legend("topleft", legend = c("True negative", "True positive", "Predicted anomalies"), 
       col = c("black", "red", "gold"), 
       pch = c(19, 19), cex=0.75)

# Close the PNG device
dev.off()

# Optionally, you can print a message to confirm the saving process
cat("Plots saved to", output_file_3, "\n")


############################################
# 2. Deep Learning Based Autoencoder Model
############################################

# Load required packages
library(h2o)
library(caret)
library(ggplot2)
library(reshape2)

# Initialize H2O
h2o.init()

# Convert datasets to H2O frame
train_set_hf <- as.h2o(train_data)
test_set_hf  <- as.h2o(test_data)

#Set predictors (all features in this case)
predictors <- names(train_set_hf)

# Train the Deep Learning Autoencoder model
dl_model <- h2o.deeplearning(
  x = predictors,
  training_frame = train_set_hf,
  validation_frame = test_set_hf,
  activation = "Tanh",                 # Tanh activation function
  autoencoder = TRUE,                  # Autoencoder mode
  hidden = c(50),                      # Number of hidden neurons
  l1 = 1e-5,                           # L1 regularization
  ignore_const_cols=FALSE,
  epochs = 1,                          # Number of epochs
  seed = 2024,                         # For reproducibility
  variable_importances = TRUE
)

dev.off()
# Visualize variable importance
h2o.varimp_plot(dl_model, num_of_features = 15)

# Make predictions on train and test data using the Autoencoder
train_anomaly_scores <- h2o.anomaly(dl_model, train_set_hf)
test_anomaly_scores  <- h2o.anomaly(dl_model, test_set_hf)

# Convert anomaly scores to matrices
train_anomaly_scores <- as.matrix(train_anomaly_scores)
test_anomaly_scores  <- as.matrix(test_anomaly_scores)

# Define threshold for anomaly detection (e.g., 95th percentile of train set)
threshold <- quantile(train_anomaly_scores, 0.95)

# Classify test data based on the threshold
all_predictions <- factor(
  ifelse(test_anomaly_scores > threshold, FALSE, TRUE)
)

# Create confusion matrix
pred <- factor(all_predictions, levels = c(TRUE, FALSE), 
               labels = c("negative", "positive"))
true <- factor(test_Class, levels = c(TRUE, FALSE), 
               labels = c("negative", "positive"))
conf_matrix <- confusionMatrix(pred, true, positive = "positive")

# Print classification metrics
metrics <- conf_matrix$byClass %>% as.data.frame() %>% 
  mutate(Measure = rownames(.))
print(metrics)

# Extract additional metrics
accuracy <- conf_matrix$overall["Accuracy"]
print(paste("Accuracy:", accuracy))

# Visualize the confusion matrix
confusion_plot <- as.data.frame(conf_matrix$table) %>%
  ggplot(aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "white", high = "blue") +
  geom_text(aes(label = Freq), color = "black") +
  labs(title = "Confusion Matrix", x = "True Class", y = "Predicted Class") +
  theme_minimal()
print(confusion_plot)

# Define the filename and path where you want to save the PNG file
output_file <- "confusion_matrix_plot_DL.png"

# Save the plot using ggsave
ggsave(filename = output_file, plot = confusion_plot, width = 6, height = 4, dpi = 300)

# Optionally, you can also display the plot
print(confusion_plot)

# Load necessary library
library(ggplot2)
library(reshape2)

# Identify the duplicated rows (excluding the first occurrence)
duplicates <- duplicated(test_data[,c(1,2)]) | duplicated(test_data[,c(1,2)], fromLast = TRUE)

# Add a small random value to the first column of the duplicated rows
test_data2 <- test_data
test_data2[rep(TRUE, 27), 1] <- test_data2[rep(TRUE, 27), 1] + 
  runif(sum(rep(TRUE, 27)), 
        min =  1, max =  12)

length(test_data2[duplicates, 1])  # 13 OUTLIERS

# Visualize the predictions and anomalies
anomalies <- test_data2[all_predictions == FALSE, ]

test_data_with_Pred_Class <- cbind(test_data2,
                                   Prediction=all_predictions) 

# Prepare data for ggplot
test_data_long <- melt(test_data_with_Pred_Class, variable.name = "Ion", value.name = "Value")
test_data_long$Class <- ifelse(test_Class, "Negative", "Positive")

# Combine test data and anomalies for plotting
combined_data <- rbind(
  data.frame(test_data_long)
)

combined_data$Prediction <- ifelse(as.logical(combined_data$Prediction), "Negative", "Positive")

# Create plots for Ion comparisons
plot_list <- lapply(1:1, function(i) {
  ggplot(combined_data[combined_data$Ion %in% c(paste0("Ion_", i)), ], 
         aes_string(x = "Value", y = "Class", color = "Class")) +
    geom_point(alpha = 0.6, size = 5.5) + 
    geom_point(data = combined_data[combined_data$Prediction == "Positive" & combined_data$Ion %in% c(paste0("Ion_", i)),], 
               aes(x = Value, y = Class), 
               color = "gold", shape = 19, size = 3, alpha = 0.8) +  # Gold points for anomalies
    geom_point(aes(x = Inf, y = Inf, color = "Anomalies"), shape = 19, size = 3) +  # Dummy point for legend
    labs(title = "Deep Learning Autoencoder Anomaly Detection",
         x = "Measurement Value of Ion 1", 
         y = "True Class") +
    scale_color_manual(values = c("Negative" = "black", 
                                  "Positive" = "red", 
                                  "Anomalies" = "gold")) +  # Add "Anomalies" to the color scale
    theme_minimal(base_size = 15) +
    theme(legend.title = element_blank(),
          legend.position = "top")
})

# Arrange plots in a grid
do.call(grid.arrange, c(plot_list, ncol = 1))

### plot_list2
combined_data2 <- combined_data[combined_data$Ion == "Ion_1",]
combined_data2$Value2 <- c(1:13, 1:14)

plot_list2 <- lapply(1:1, function(i) {
  ggplot(combined_data2, 
         aes_string(x = "Value2", y = "Class", color = "Class")) +
    geom_point(alpha = 0.6, size = 5.5) + 
    geom_point(data = combined_data2[combined_data2$Prediction == "Positive",], 
               aes(x = Value2, y = Class), 
               color = "gold", shape = 19, size = 3, alpha = 0.8) +  # Gold points for anomalies
    geom_point(aes(x = Inf, y = Inf, color = "Anomalies"), shape = 19, size = 3) +  # Dummy point for legend
    labs(title = "Deep Learning Autoencoder Anomaly Detection",
         x = "Sample", 
         y = "True Class") +
    scale_color_manual(values = c("Negative" = "black", 
                                  "Positive" = "red", 
                                  "Anomalies" = "gold")) +  # Add "Anomalies" to the color scale
    scale_x_continuous(breaks = seq(min(combined_data2$Value2), max(combined_data2$Value2), by = 1)) +  # Show all numbers on the x-axis
    theme_minimal(base_size = 15) +
    theme(legend.title = element_blank(),
          legend.position = "top")
})


# Define the filename and path where you want to save the PNG file
output_file <- "Classification_plot_DL.png"

# Save the plots using ggsave. We use arrangeGrob to combine plots before saving.
g <- do.call(grid.arrange, c(plot_list2, ncol = 1))  # Combine plots into a single object

# Save the combined plot
ggsave(filename = output_file, plot = g, width = 8, height = 8, dpi = 300)

# Optionally, you can display the plot
print(g)


# Additional metrics
sensitivity <- conf_matrix$byClass["Sensitivity"]
specificity <- conf_matrix$byClass["Specificity"]
g_mean <- sqrt(sensitivity * specificity)

# Print additional metrics
print(paste("Sensitivity:", sensitivity))
print(paste("Specificity:", specificity))
print(paste("G-mean:", g_mean))

# Store performance metrics
performance_metrics <- list(
  accuracy = accuracy,
  precision = conf_matrix$byClass["Precision"],
  recall = conf_matrix$byClass["Recall"],
  f1_score = conf_matrix$byClass["F1"],
  sensitivity = sensitivity,
  specificity = specificity,
  g_mean = g_mean
)

# Print performance metrics for all models
print(performance_metrics)

# Identify the duplicated rows (excluding the first occurrence)
duplicates <- duplicated(test_data[,c(1,2)]) | duplicated(test_data[,c(1,2)], fromLast = TRUE)

# Add a small random value to the first column of the duplicated rows
test_data2 <- test_data
test_data2[duplicates, 1] <- test_data2[duplicates, 1] + runif(sum(duplicates), 
                                                               min =  2, max =  4)
test_data2[duplicates, 2] <- test_data2[duplicates, 2] + runif(sum(duplicates), 
                                                               min =  2, max =  4)
test_data2[duplicates, 3] <- test_data2[duplicates, 3] + runif(sum(duplicates), 
                                                               min = 2, max = 4)
test_data2[duplicates, 4] <- test_data2[duplicates, 4] + runif(sum(duplicates), 
                                                               min =  2, max =  4)
test_data2[duplicates, 5] <- test_data2[duplicates, 5] + runif(sum(duplicates), 
                                                               min =  2, max =  4)
length(test_data[duplicates, 1])  # 18 OUTLIERS

# Extract anomalies from the test data
anomalies <- test_data2[all_predictions == FALSE, ]


# Create a data frame for ggplot
plot_data <- data.frame(
  Ion1 = test_data2[, 1],
  Ion2 = test_data2[, 2],
  Class = ifelse(test_Class, "Negative", "Positive"),
  Anomaly = ifelse(all_predictions == FALSE, "Anomaly", "Non-Anomaly")
)

# Create the plot
performance_text <- paste(
  "Accuracy: ", round(performance_metrics$accuracy, 2), "\n",
  "Precision: ", round(performance_metrics$precision, 2), "\n",
  "Recall: ", round(performance_metrics$recall, 2), "\n",
  "F1 Score: ", round(performance_metrics$f1_score, 2), "\n",
  "Sensitivity: ", round(performance_metrics$sensitivity, 2), "\n",
  "Specificity: ", round(performance_metrics$specificity, 2), "\n",
  "G-Mean: ", round(performance_metrics$g_mean, 2)
)

# Define the filename and path where you want to save the PNG file
output_file_2 <- "DL_anomaly_detection_plot_Ion1_Ion2.png"


dl_plot <- ggplot(plot_data, aes(x = Ion1, y = Ion2)) +
  geom_point(aes(color = Class, shape = Anomaly), size = 3, alpha = 0.7) +
  scale_color_manual(values = c("Negative" = "black", "Positive" = "red")) +
  scale_shape_manual(values = c("Non-Anomaly" = 19, "Anomaly" = 17)) + # Triangle for anomalies
  labs(title = "Deep Learning Autoencoder Anomaly Detection",
       x = "Ion 1",
       y = "Ion 2",
       color = "Class",
       shape = "Prediction") +
  theme_minimal(base_size = 15) +
  theme(legend.position = "bottom") +
  guides(shape = guide_legend(override.aes = list(size = 4))) +  # Increase size of legend points
  annotate("text", x = max(plot_data$Ion1) * 0.8, y = max(plot_data$Ion2) * 0.8, 
           label = performance_text, hjust = 0, vjust = 1.3, size = 6, 
           color = "blue", 
           family = "serif", fontface = "italic")

# Save the plot
ggsave(filename = output_file_2, plot = dl_plot, width = 9, height = 9, dpi = 300)

# Optionally, you can display the plot
print(dl_plot)

# Set the output file for saving the plots
output_file_3 <- "dl_anomaly_detection_visualization.png"

# Open a PNG device
png(filename = output_file_3, width = 800, height = 800)


# Visualize the test data and anomalies
par(mfrow = c(2, 2))
plot(test_data2[, c(1, 2)], col = ifelse(test_Class == TRUE, "black", "red"), 
     cex=1.5, pch = 19, main = "Deep Learning Autoencoder Anomaly Detection")
points(anomalies[, c(1, 2)], col = "gold", pch = 19, cex=0.8)
legend("bottomright", legend = c("Negative", "Positive", "Predicted as anomalies"), 
       col = c("black", "red", "gold"), 
       pch = c(19, 19), cex=0.75)

plot(test_data2[, c(1, 3)], col = ifelse(test_Class == TRUE, "black", "red"), 
     cex=1.3, pch = 19, main = "Deep Learning Autoencoder Anomaly Detection")
points(anomalies[, c(1, 3)], col = "gold", pch = 19, cex=0.8)
legend("topright", legend = c("True negative", "True positive", "Predicted anomalies"), 
       col = c("black", "red", "gold"), 
       pch = c(19, 19), cex=0.75)

plot(test_data2[, c(1, 4)], col = ifelse(test_Class == TRUE, "black", "red"), 
     cex=1.3, pch = 19, main = "Deep Learning Autoencoder Anomaly Detection")
points(anomalies[, c(1, 4)], col = "gold", pch = 19, cex=0.8)
legend("topleft", legend = c("True negative", "True positive", "Predicted anomalies"),  
       col = c("black", "red", "gold"), 
       pch = c(19, 19), cex=0.75)

plot(test_data2[, c(1, 5)], col = ifelse(test_Class == TRUE, "black", "red"), 
     cex=1.3, pch = 19, main = "Deep Learning Autoencoder Anomaly Detection")
points(anomalies[, c(1, 5)], col = "gold", pch = 19, cex=0.8)
legend("bottomright", legend = c("True negative", "True positive", "Predicted anomalies"), 
       col = c("black", "red", "gold"), 
       pch = c(19, 19), cex=0.75)


# Close the PNG device
dev.off()

# Optionally, you can print a message to confirm the saving process
cat("Plots saved to", output_file_3, "\n")




##############################################
#### 3. One-Class k-Means (manual implementation with thresholding):
##############################################

# Run k-means clustering with only 1 cluster
set.seed(1234)
kmeans_model <- kmeans(train_data, centers = 3, nstart = 10)

# Compute distances to the centroid
distances <- sqrt(rowSums((test_data - kmeans_model$centers[kmeans_model$cluster, ])^2))

# Set a threshold for classification
threshold <- quantile(distances, 0.85)  # for example, 95% quantile

# Classify data points as normal (0) or anomaly (1)
kmeans_predictions <- ifelse(distances > threshold, FALSE, TRUE)
table(kmeans_predictions)

all_predictions <- kmeans_predictions

# Create confusion matrix
pred <- factor(all_predictions, levels = c(TRUE, FALSE), 
               labels = c("negative", "positive"))
true <- factor(test_Class, levels = c(TRUE, FALSE), 
               labels = c("negative", "positive"))
conf_matrix <- confusionMatrix(pred, true, positive = "positive")

# Print classification metrics
metrics <- conf_matrix$byClass %>% as.data.frame() %>% 
  mutate(Measure = rownames(.))
print(metrics)

# Extract additional metrics
accuracy <- conf_matrix$overall["Accuracy"]
print(paste("Accuracy:", accuracy))

# Visualize the confusion matrix
confusion_plot <- as.data.frame(conf_matrix$table) %>%
  ggplot(aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "white", high = "blue") +
  geom_text(aes(label = Freq), color = "black") +
  labs(title = "Confusion Matrix", x = "True Class", y = "Predicted Class") +
  theme_minimal()
print(confusion_plot)

# Define the filename and path where you want to save the PNG file
output_file <- "confusion_matrix_plot_OC-KMEANS.png"

# Save the plot using ggsave
ggsave(filename = output_file, plot = confusion_plot, width = 6, height = 4, dpi = 300)

# Optionally, you can also display the plot
print(confusion_plot)

# Load necessary library
library(ggplot2)
library(reshape2)

# Identify the duplicated rows (excluding the first occurrence)
duplicates <- duplicated(test_data[,c(1,2)]) | duplicated(test_data[,c(1,2)], fromLast = TRUE)

# Add a small random value to the first column of the duplicated rows
test_data2 <- test_data
test_data2[rep(TRUE, 27), 1] <- test_data2[rep(TRUE, 27), 1] + 
  runif(sum(rep(TRUE, 27)), 
        min =  1, max =  12)

length(test_data2[duplicates, 1])  # 13 OUTLIERS

# Visualize the predictions and anomalies
anomalies <- test_data2[all_predictions == FALSE, ]

test_data_with_Pred_Class <- cbind(test_data2,
                                   Prediction=all_predictions) 

# Prepare data for ggplot
test_data_long <- melt(test_data_with_Pred_Class, variable.name = "Ion", value.name = "Value")
test_data_long$Class <- ifelse(test_Class, "Negative", "Positive")

# Combine test data and anomalies for plotting
combined_data <- rbind(
  data.frame(test_data_long)
)

combined_data$Prediction <- ifelse(as.logical(combined_data$Prediction), "Negative", "Positive")

# Create plots for Ion comparisons
plot_list <- lapply(1:1, function(i) {
  ggplot(combined_data[combined_data$Ion %in% c(paste0("Ion_", i)), ], 
         aes_string(x = "Value", y = "Class", color = "Class")) +
    geom_point(alpha = 0.6, size = 5.5) + 
    geom_point(data = combined_data[combined_data$Prediction == "Positive" & combined_data$Ion %in% c(paste0("Ion_", i)),], 
               aes(x = Value, y = Class), 
               color = "gold", shape = 19, size = 3, alpha = 0.8) +  # Gold points for anomalies
    geom_point(aes(x = Inf, y = Inf, color = "Anomalies"), shape = 19, size = 3) +  # Dummy point for legend
    labs(title = "One-Class k-Means Anomaly Detection",
         x = "Measurement Value of Ion 1", 
         y = "True Class") +
    scale_color_manual(values = c("Negative" = "black", 
                                  "Positive" = "red", 
                                  "Anomalies" = "gold")) +  # Add "Anomalies" to the color scale
    theme_minimal(base_size = 15) +
    theme(legend.title = element_blank(),
          legend.position = "top")
})

# Arrange plots in a grid
do.call(grid.arrange, c(plot_list, ncol = 1))

### plot_list2
combined_data2 <- combined_data[combined_data$Ion == "Ion_1",]
combined_data2$Value2 <- c(1:13, 1:14)

plot_list2 <- lapply(1:1, function(i) {
  ggplot(combined_data2, 
         aes_string(x = "Value2", y = "Class", color = "Class")) +
    geom_point(alpha = 0.6, size = 5.5) + 
    geom_point(data = combined_data2[combined_data2$Prediction == "Positive",], 
               aes(x = Value2, y = Class), 
               color = "gold", shape = 19, size = 3, alpha = 0.8) +  # Gold points for anomalies
    geom_point(aes(x = Inf, y = Inf, color = "Anomalies"), shape = 19, size = 3) +  # Dummy point for legend
    labs(title = "One-Class k-Means Anomaly Detection",
         x = "Sample", 
         y = "True Class") +
    scale_color_manual(values = c("Negative" = "black", 
                                  "Positive" = "red", 
                                  "Anomalies" = "gold")) +  # Add "Anomalies" to the color scale
    scale_x_continuous(breaks = seq(min(combined_data2$Value2), max(combined_data2$Value2), by = 1)) +  # Show all numbers on the x-axis
    theme_minimal(base_size = 15) +
    theme(legend.title = element_blank(),
          legend.position = "top")
})

# Define the filename and path where you want to save the PNG file
output_file <- "Classification_plot_OC_kmeans.png"

# Save the plots using ggsave. We use arrangeGrob to combine plots before saving.
g <- do.call(grid.arrange, c(plot_list2, ncol = 1))  # Combine plots into a single object

# Save the combined plot
ggsave(filename = output_file, plot = g, width = 6, height = 6, dpi = 300)

# Optionally, you can display the plot
print(g)

# Additional metrics
sensitivity <- conf_matrix$byClass["Sensitivity"]
specificity <- conf_matrix$byClass["Specificity"]
g_mean <- sqrt(sensitivity * specificity)

# Print additional metrics
print(paste("Sensitivity:", sensitivity))
print(paste("Specificity:", specificity))
print(paste("G-mean:", g_mean))

# Store performance metrics
performance_metrics <- list(
  accuracy = accuracy,
  precision = conf_matrix$byClass["Precision"],
  recall = conf_matrix$byClass["Recall"],
  f1_score = conf_matrix$byClass["F1"],
  sensitivity = sensitivity,
  specificity = specificity,
  g_mean = g_mean
)

# Print performance metrics for all models
print(performance_metrics)

# Identify the duplicated rows (excluding the first occurrence)
duplicates <- duplicated(test_data[,c(1,2)]) | duplicated(test_data[,c(1,2)], fromLast = TRUE)

# Add a small random value to the first column of the duplicated rows
test_data2 <- test_data
test_data2[duplicates, 1] <- test_data2[duplicates, 1] + runif(sum(duplicates), 
                                                               min =  2, max =  4)
test_data2[duplicates, 2] <- test_data2[duplicates, 2] + runif(sum(duplicates), 
                                                               min =  2, max =  4)
test_data2[duplicates, 3] <- test_data2[duplicates, 3] + runif(sum(duplicates), 
                                                               min = 2, max = 4)
test_data2[duplicates, 4] <- test_data2[duplicates, 4] + runif(sum(duplicates), 
                                                               min =  2, max =  4)
test_data2[duplicates, 5] <- test_data2[duplicates, 5] + runif(sum(duplicates), 
                                                               min =  2, max =  4)
length(test_data[duplicates, 1])  # 18 OUTLIERS

# Extract anomalies from the test data
anomalies <- test_data2[all_predictions == FALSE, ]


# Create a data frame for ggplot
plot_data <- data.frame(
  Ion1 = test_data2[, 1],
  Ion2 = test_data2[, 2],
  Class = ifelse(test_Class, "Negative", "Positive"),
  Anomaly = ifelse(all_predictions == FALSE, "Anomaly", "Non-Anomaly")
)

# Create the plot
performance_text <- paste(
  "Accuracy: ", round(performance_metrics$accuracy, 2), "\n",
  "Precision: ", round(performance_metrics$precision, 2), "\n",
  "Recall: ", round(performance_metrics$recall, 2), "\n",
  "F1 Score: ", round(performance_metrics$f1_score, 2), "\n",
  "Sensitivity: ", round(performance_metrics$sensitivity, 2), "\n",
  "Specificity: ", round(performance_metrics$specificity, 2), "\n",
  "G-Mean: ", round(performance_metrics$g_mean, 2)
)

# Define the filename and path where you want to save the PNG file
output_file_2 <- "OC_kmeans_anomaly_detection_plot_Ion1_Ion2.png"

oc_kmeans_plot <- ggplot(plot_data, aes(x = Ion1, y = Ion2)) +
  geom_point(aes(color = Class, shape = Anomaly), size = 3, alpha = 0.7) +
  scale_color_manual(values = c("Negative" = "black", "Positive" = "red")) +
  scale_shape_manual(values = c("Non-Anomaly" = 19, "Anomaly" = 17)) + # Triangle for anomalies
  labs(title = "One-Class k-Means Anomaly Detection",
       x = "Ion 1",
       y = "Ion 2",
       color = "Class",
       shape = "Prediction") +
  theme_minimal(base_size = 15) +
  theme(legend.position = "bottom") +
  guides(shape = guide_legend(override.aes = list(size = 4))) +  # Increase size of legend points
  annotate("text", x = max(plot_data$Ion1) * 0.8, y = max(plot_data$Ion2) * 0.8, 
           label = performance_text, hjust = 0, vjust = 1.3, size = 6, 
           color = "blue", 
           family = "serif", fontface = "italic")

# Save the plot
ggsave(filename = output_file_2, plot = oc_kmeans_plot, width = 9, height = 9, dpi = 300)

# Optionally, you can display the plot
print(oc_kmeans_plot)

# Set the output file for saving the plots
output_file_3 <- "oc_kmeans_anomaly_detection_visualization.png"

# Open a PNG device
png(filename = output_file_3, width = 800, height = 800)



# Visualize the test data and anomalies
par(mfrow = c(2, 2))
plot(test_data2[, c(1, 2)], col = ifelse(test_Class == TRUE, "black", "red"), 
     cex=1.5, pch = 19, main = "One-Class k-Means Anomaly Detection")
points(anomalies[, c(1, 2)], col = "gold", pch = 19, cex=0.8)
legend("bottomright", legend = c("Negative", "Positive", "Predicted as anomalies"), 
       col = c("black", "red", "gold"), 
       pch = c(19, 19), cex=0.75)

plot(test_data2[, c(1, 3)], col = ifelse(test_Class == TRUE, "black", "red"), 
     cex=1.3, pch = 19, main = "One-Class k-Means Anomaly Detection")
points(anomalies[, c(1, 3)], col = "gold", pch = 19, cex=0.8)
legend("topright", legend = c("True negative", "True positive", "Predicted anomalies"), 
       col = c("black", "red", "gold"), 
       pch = c(19, 19), cex=0.75)

plot(test_data2[, c(1, 4)], col = ifelse(test_Class == TRUE, "black", "red"), 
     cex=1.3, pch = 19, main = "One-Class k-Means Anomaly Detection")
points(anomalies[, c(1, 4)], col = "gold", pch = 19, cex=0.8)
legend("topleft", legend = c("True negative", "True positive", "Predicted anomalies"),  
       col = c("black", "red", "gold"), 
       pch = c(19, 19), cex=0.75)

plot(test_data2[, c(1, 5)], col = ifelse(test_Class == TRUE, "black", "red"), 
     cex=1.3, pch = 19, main = "One-Class k-Means Anomaly Detection")
points(anomalies[, c(1, 5)], col = "gold", pch = 19, cex=0.8)
legend("bottomright", legend = c("True negative", "True positive", "Predicted anomalies"), 
       col = c("black", "red", "gold"), 
       pch = c(19, 19), cex=0.75)

# Close the PNG device
dev.off()

# Optionally, you can print a message to confirm the saving process
cat("Plots saved to", output_file_3, "\n")



###########################
### 4. PCA Anomaly Detection 
#############################

# Fit PCA model
pca_model <- prcomp(train_data, scale. = FALSE)

# Reconstruct the data
pca_reconstruction <- predict(pca_model, train_data) %*% t(pca_model$rotation)

# Compute reconstruction error
pca_error <- rowSums((test_data - pca_reconstruction)^2)

# Set a threshold for classification
pca_threshold <- quantile(pca_error, 0.95)  # For example, 95% quantile

# Classify data points as normal (0) or anomaly (1)
pca_predictions <- ifelse(pca_error > pca_threshold, FALSE, TRUE)
table(pca_predictions)


all_predictions <- pca_predictions

# Create confusion matrix
pred <- factor(all_predictions, levels = c(TRUE, FALSE), 
               labels = c("negative", "positive"))
true <- factor(test_Class, levels = c(TRUE, FALSE), 
               labels = c("negative", "positive"))
conf_matrix <- confusionMatrix(pred, true, positive = "positive")

# Print classification metrics
metrics <- conf_matrix$byClass %>% as.data.frame() %>% 
  mutate(Measure = rownames(.))
print(metrics)

# Extract additional metrics
accuracy <- conf_matrix$overall["Accuracy"]
print(paste("Accuracy:", accuracy))

# Visualize the confusion matrix
confusion_plot <- as.data.frame(conf_matrix$table) %>%
  ggplot(aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "white", high = "blue") +
  geom_text(aes(label = Freq), color = "black") +
  labs(title = "Confusion Matrix", x = "True Class", y = "Predicted Class") +
  theme_minimal()
print(confusion_plot)

# Define the filename and path where you want to save the PNG file
output_file <- "confusion_matrix_plot_PCA.png"

# Save the plot using ggsave
ggsave(filename = output_file, plot = confusion_plot, width = 6, height = 4, dpi = 300)

# Optionally, you can also display the plot
print(confusion_plot)

# Load necessary library
library(ggplot2)
library(reshape2)

# Identify the duplicated rows (excluding the first occurrence)
duplicates <- duplicated(test_data[,c(1,2)]) | duplicated(test_data[,c(1,2)], fromLast = TRUE)

# Add a small random value to the first column of the duplicated rows
test_data2 <- test_data
test_data2[rep(TRUE, 27), 1] <- test_data2[rep(TRUE, 27), 1] + 
  runif(sum(rep(TRUE, 27)), 
        min =  1, max =  12)

length(test_data2[duplicates, 1])  # 13 OUTLIERS

# Visualize the predictions and anomalies
anomalies <- test_data2[all_predictions == FALSE, ]

test_data_with_Pred_Class <- cbind(test_data2,
                                   Prediction=all_predictions) 

# Prepare data for ggplot
test_data_long <- melt(test_data_with_Pred_Class, variable.name = "Ion", value.name = "Value")
test_data_long$Class <- ifelse(test_Class, "Negative", "Positive")

# Combine test data and anomalies for plotting
combined_data <- rbind(
  data.frame(test_data_long)
)

combined_data$Prediction <- ifelse(as.logical(combined_data$Prediction), "Negative", "Positive")

# Create plots for Ion comparisons
plot_list <- lapply(1:1, function(i) {
  ggplot(combined_data[combined_data$Ion %in% c(paste0("Ion_", i)), ], 
         aes_string(x = "Value", y = "Class", color = "Class")) +
    geom_point(alpha = 0.6, size = 5.5) + 
    geom_point(data = combined_data[combined_data$Prediction == "Positive" & combined_data$Ion %in% c(paste0("Ion_", i)),], 
               aes(x = Value, y = Class), 
               color = "gold", shape = 19, size = 3, alpha = 0.8) +  # Gold points for anomalies
    geom_point(aes(x = Inf, y = Inf, color = "Anomalies"), shape = 19, size = 3) +  # Dummy point for legend
    labs(title = "PCA",
         x = "Measurement Value of Ion 1", 
         y = "True Class") +
    scale_color_manual(values = c("Negative" = "black", 
                                  "Positive" = "red", 
                                  "Anomalies" = "gold")) +  # Add "Anomalies" to the color scale
    theme_minimal(base_size = 15) +
    theme(legend.title = element_blank(),
          legend.position = "top")
})

# Arrange plots in a grid
do.call(grid.arrange, c(plot_list, ncol = 1))

### plot_list2
combined_data2 <- combined_data[combined_data$Ion == "Ion_1",]
combined_data2$Value2 <- c(1:13, 1:14)

plot_list2 <- lapply(1:1, function(i) {
  ggplot(combined_data2, 
         aes_string(x = "Value2", y = "Class", color = "Class")) +
    geom_point(alpha = 0.6, size = 5.5) + 
    geom_point(data = combined_data2[combined_data2$Prediction == "Positive",], 
               aes(x = Value2, y = Class), 
               color = "gold", shape = 19, size = 3, alpha = 0.8) +  # Gold points for anomalies
    geom_point(aes(x = Inf, y = Inf, color = "Anomalies"), shape = 19, size = 3) +  # Dummy point for legend
    labs(title = "PCA Anomaly Detection",
         x = "Sample", 
         y = "True Class") +
    scale_color_manual(values = c("Negative" = "black", 
                                  "Positive" = "red", 
                                  "Anomalies" = "gold")) +  # Add "Anomalies" to the color scale
    scale_x_continuous(breaks = seq(min(combined_data2$Value2), max(combined_data2$Value2), by = 1)) +  # Show all numbers on the x-axis
    theme_minimal(base_size = 15) +
    theme(legend.title = element_blank(),
          legend.position = "top")
})


# Define the filename and path where you want to save the PNG file
output_file <- "Classification_plot_OCC_PCA.png"

# Save the plots using ggsave. We use arrangeGrob to combine plots before saving.
g <- do.call(grid.arrange, c(plot_list2, ncol = 1))  # Combine plots into a single object

# Save the combined plot
ggsave(filename = output_file, plot = g, width = 6, height = 6, dpi = 300)

# Optionally, you can display the plot
print(g)


# Additional metrics
sensitivity <- conf_matrix$byClass["Sensitivity"]
specificity <- conf_matrix$byClass["Specificity"]
g_mean <- sqrt(sensitivity * specificity)

# Print additional metrics
print(paste("Sensitivity:", sensitivity))
print(paste("Specificity:", specificity))
print(paste("G-mean:", g_mean))

# Store performance metrics
performance_metrics <- list(
  accuracy = accuracy,
  precision = conf_matrix$byClass["Precision"],
  recall = conf_matrix$byClass["Recall"],
  f1_score = conf_matrix$byClass["F1"],
  sensitivity = sensitivity,
  specificity = specificity,
  g_mean = g_mean
)

# Print performance metrics for all models
print(performance_metrics)

# Identify the duplicated rows (excluding the first occurrence)
duplicates <- duplicated(test_data[,c(1,2)]) | duplicated(test_data[,c(1,2)], fromLast = TRUE)

# Add a small random value to the first column of the duplicated rows
test_data2 <- test_data
test_data2[duplicates, 1] <- test_data2[duplicates, 1] + runif(sum(duplicates), 
                                                               min =  2, max =  4)
test_data2[duplicates, 2] <- test_data2[duplicates, 2] + runif(sum(duplicates), 
                                                               min =  2, max =  4)
test_data2[duplicates, 3] <- test_data2[duplicates, 3] + runif(sum(duplicates), 
                                                               min = 2, max = 4)
test_data2[duplicates, 4] <- test_data2[duplicates, 4] + runif(sum(duplicates), 
                                                               min =  2, max =  4)
test_data2[duplicates, 5] <- test_data2[duplicates, 5] + runif(sum(duplicates), 
                                                               min =  2, max =  4)
length(test_data[duplicates, 1])  # 18 OUTLIERS

# Extract anomalies from the test data
anomalies <- test_data2[all_predictions == FALSE, ]


# Create a data frame for ggplot
plot_data <- data.frame(
  Ion1 = test_data2[, 1],
  Ion2 = test_data2[, 2],
  Class = ifelse(test_Class, "Negative", "Positive"),
  Anomaly = ifelse(all_predictions == FALSE, "Anomaly", "Non-Anomaly")
)

# Create the plot
performance_text <- paste(
  "Accuracy: ", round(performance_metrics$accuracy, 2), "\n",
  "Precision: ", round(performance_metrics$precision, 2), "\n",
  "Recall: ", round(performance_metrics$recall, 2), "\n",
  "F1 Score: ", round(performance_metrics$f1_score, 2), "\n",
  "Sensitivity: ", round(performance_metrics$sensitivity, 2), "\n",
  "Specificity: ", round(performance_metrics$specificity, 2), "\n",
  "G-Mean: ", round(performance_metrics$g_mean, 2)
)


# Define the filename and path where you want to save the PNG file
output_file_2 <- "PCA_anomaly_detection_plot_Ion1_Ion2.png"


pca_plot <- ggplot(plot_data, aes(x = Ion1, y = Ion2)) +
  geom_point(aes(color = Class, shape = Anomaly), size = 3, alpha = 0.7) +
  scale_color_manual(values = c("Negative" = "black", "Positive" = "red")) +
  scale_shape_manual(values = c("Non-Anomaly" = 19, "Anomaly" = 17)) + # Triangle for anomalies
  labs(title = "PCA",
       x = "Ion 1",
       y = "Ion 2",
       color = "Class",
       shape = "Prediction") +
  theme_minimal(base_size = 15) +
  theme(legend.position = "bottom") +
  guides(shape = guide_legend(override.aes = list(size = 4))) +  # Increase size of legend points
  annotate("text", x = max(plot_data$Ion1) * 0.8, y = max(plot_data$Ion2) * 0.8, 
           label = performance_text, hjust = 0, vjust = 1.3, size = 6, 
           color = "blue", 
           family = "serif", fontface = "italic")

# Save the plot
ggsave(filename = output_file_2, plot = pca_plot, width = 9, height = 9, dpi = 300)

# Optionally, you can display the plot
print(pca_plot)

# Set the output file for saving the plots
output_file_3 <- "pca_anomaly_detection_visualization.png"

# Open a PNG device
png(filename = output_file_3, width = 800, height = 800)


# Visualize the test data and anomalies
par(mfrow = c(2, 2))
plot(test_data2[, c(1, 2)], col = ifelse(test_Class == TRUE, "black", "red"), 
     cex=1.5, pch = 19, main = "PCA")
points(anomalies[, c(1, 2)], col = "gold", pch = 19, cex=0.8)
legend("bottomright", legend = c("Negative", "Positive", "Predicted as anomalies"), 
       col = c("black", "red", "gold"), 
       pch = c(19, 19), cex=0.75)

plot(test_data2[, c(1, 3)], col = ifelse(test_Class == TRUE, "black", "red"), 
     cex=1.3, pch = 19, main = "PCA")
points(anomalies[, c(1, 3)], col = "gold", pch = 19, cex=0.8)
legend("topright", legend = c("True negative", "True positive", "Predicted anomalies"), 
       col = c("black", "red", "gold"), 
       pch = c(19, 19), cex=0.75)

plot(test_data2[, c(1, 4)], col = ifelse(test_Class == TRUE, "black", "red"), 
     cex=1.3, pch = 19, main = "PCA")
points(anomalies[, c(1, 4)], col = "gold", pch = 19, cex=0.8)
legend("topleft", legend = c("True negative", "True positive", "Predicted anomalies"),  
       col = c("black", "red", "gold"), 
       pch = c(19, 19), cex=0.75)

plot(test_data2[, c(1, 5)], col = ifelse(test_Class == TRUE, "black", "red"), 
     cex=1.3, pch = 19, main = "PCA")
points(anomalies[, c(1, 5)], col = "gold", pch = 19, cex=0.8)
legend("topright", legend = c("True negative", "True positive", "Predicted anomalies"), 
       col = c("black", "red", "gold"), 
       pch = c(19, 19), cex=0.75)


# Close the PNG device
dev.off()

# Optionally, you can print a message to confirm the saving process
cat("Plots saved to", output_file_3, "\n")


### MODEL COMPARISON

# Combine all performance metrics into one data frame
combined_metrics <- rbind(
  data.frame(Method = "SVM", performance_metrics_svm),
  data.frame(Method = "DL", performance_metrics_dl),
  data.frame(Method = "KMeans", performance_metrics_kmeans),
  data.frame(Method = "PCA", performance_metrics_pca)
)

# Print the combined metrics
print(combined_metrics)




