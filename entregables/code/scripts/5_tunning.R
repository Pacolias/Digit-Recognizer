# 5_tunning.R
library(caret)
library(rpart)
library(randomForest)
library(e1071)
library(nnet)
source(here::here("scripts", "utils.R"))

set.seed(123)

# Cargar datos
train <- readRDS(here::here("data", "processed", "train_pca.rds"))

# Usamos muestra reducida para el tuning
train_small_idx <- createDataPartition(train$label, p = 0.1, list=FALSE) 
train_int <- train[train_small_idx, ]

# IMPORTANTE: classProbs = TRUE para que funcionen en el ensemble
ctrl <- trainControl(method = "cv", number = 3, verboseIter = FALSE, classProbs = TRUE)

# -----------------------------------------------------------------------------
# 1. Random Forest
# -----------------------------------------------------------------------------
cat("Tuning Random Forest...\n")
grid_rf <- expand.grid(mtry = c(2, 4))
model_rf <- train(label ~ ., data = train_int, method = "rf", 
                  trControl = ctrl, tuneGrid = grid_rf, ntree = 50)
saveRDS(model_rf, here::here("models", "model_rf_tuned.rds"))

# -----------------------------------------------------------------------------
# 2. SVM (Support Vector Machine)
# -----------------------------------------------------------------------------
cat("Tuning SVM...\n")
model_svm <- train(label ~ ., data = train_int, method = "svmLinear", 
                   trControl = ctrl, tuneLength = 2)
saveRDS(model_svm, here::here("models", "model_svm_tuned.rds"))

# -----------------------------------------------------------------------------
# 3. Árbol de Decisión (rpart)
# -----------------------------------------------------------------------------
cat("Tuning Árbol de Decisión...\n")
grid_rpart <- expand.grid(cp = c(0.01, 0.001))
model_rpart <- train(label ~ ., data = train_int, method = "rpart",
                     trControl = ctrl, tuneGrid = grid_rpart)
saveRDS(model_rpart, here::here("models", "model_rpart_tuned.rds"))

# -----------------------------------------------------------------------------
# 4. Perceptrón Multicapa (Red Neuronal - nnet)
# -----------------------------------------------------------------------------
cat("Tuning Perceptrón Multicapa (MLP)...\n")
grid_nnet <- expand.grid(size = c(5, 10), decay = c(0.1))
# MaxNWts evita el error de "too many weights"
model_nnet <- train(label ~ ., data = train_int, method = "nnet",
                    trControl = ctrl, tuneGrid = grid_nnet, 
                    MaxNWts = 5000, trace = FALSE)
saveRDS(model_nnet, here::here("models", "model_nnet_tuned.rds"))

cat(">>> Tuning completado. 4 Modelos guardados en models/\n")