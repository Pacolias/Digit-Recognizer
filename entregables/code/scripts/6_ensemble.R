# 6_ensemble.R
library(caret)
library(nnet)
library(tidyverse)
source(here::here("scripts", "utils.R"))

set.seed(123)

# Cargar datos y los 4 modelos
train <- readRDS(here::here("data", "processed", "train_pca.rds"))
model_rf <- readRDS(here::here("models", "model_rf_tuned.rds"))
model_svm <- readRDS(here::here("models", "model_svm_tuned.rds"))
model_rpart <- readRDS(here::here("models", "model_rpart_tuned.rds"))
model_nnet <- readRDS(here::here("models", "model_nnet_tuned.rds"))

# Muestra fresca para el ensemble
train_idx <- createDataPartition(train$label, p = 0.2, list=FALSE)
train_meta <- train[train_idx, ]

cat("Generando predicciones de los 4 modelos...\n")
pred_rf    <- predict(model_rf, train_meta, type = "prob")
pred_svm   <- predict(model_svm, train_meta, type = "prob")
pred_rpart <- predict(model_rpart, train_meta, type = "prob")
pred_nnet  <- predict(model_nnet, train_meta, type = "prob")

# Combinar todo
meta_train <- bind_cols(pred_rf, pred_svm, pred_rpart, pred_nnet)
colnames(meta_train) <- make.names(colnames(meta_train), unique = TRUE)
meta_train$label <- train_meta$label
meta_train <- na.omit(meta_train)

# Entrenar Meta-modelo
cat("Entrenando Meta-Modelo (Stacking)...\n")
meta_model <- multinom(label ~ ., data = meta_train, trace = FALSE, MaxNWts = 5000)

saveRDS(meta_model, here::here("models", "meta_model.rds"))
cat(">>> Ensemble con 4 modelos guardado.\n")