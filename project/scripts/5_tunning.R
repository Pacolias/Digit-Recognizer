# 5_tunning.R
library(caret)
library(rpart)
library(randomForest)
library(e1071)
library(nnet)
source(here::here("scripts", "utils.R"))

set.seed(123)

# Cargar datos (CORREGIDO: Lee el rds, no RData)
train <- readRDS(here::here("data", "processed", "train_pca.rds"))

# Reducimos dataset para que el tuning no tarde horas en la práctica
# Usamos solo 1000 filas para demostración. Quita esto si tienes tiempo y CPU.
train_small_idx <- createDataPartition(train$label, p = 0.1, list=FALSE) 
train_int <- train[train_small_idx, ]

ctrl <- trainControl(method = "cv", number = 3, verboseIter = FALSE)

# Modelo 1: RF
cat("Tuning RF...\n")
grid_rf <- expand.grid(mtry = c(2, 4)) # Grid pequeño
model_rf_tuned <- train(label ~ ., data = train_int, method = "rf", 
                        trControl = ctrl, tuneGrid = grid_rf, ntree = 50)
saveRDS(model_rf_tuned, here::here("models", "model_rf_tuned.rds"))

# Modelo 2: SVM
cat("Tuning SVM...\n")
model_svm_tuned <- train(label ~ ., data = train_int, method = "svmLinear", 
                         trControl = ctrl, tuneLength = 2)
saveRDS(model_svm_tuned, here::here("models", "model_svm_tuned.rds"))

cat(">>> Tuning completado. Modelos guardados como .rds en models/\n")