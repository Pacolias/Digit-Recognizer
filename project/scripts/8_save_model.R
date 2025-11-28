# 8_save_model.R
library(caret)
library(nnet)
library(tidyverse)
source(here::here("scripts", "utils.R"))

cat("Cargando todo para el modelo final reducido...\n")
train <- readRDS(here::here("data", "processed", "train_pca.rds"))
model_rf <- readRDS(here::here("models", "model_rf_tuned.rds"))
model_svm <- readRDS(here::here("models", "model_svm_tuned.rds"))
model_rpart <- readRDS(here::here("models", "model_rpart_tuned.rds"))
model_nnet <- readRDS(here::here("models", "model_nnet_tuned.rds"))

# 100 casos
train_small <- train[1:100, ]

# Predicciones
pred_rf    <- predict(model_rf, train_small, type = "prob")
pred_svm   <- predict(model_svm, train_small, type = "prob")
pred_rpart <- predict(model_rpart, train_small, type = "prob")
pred_nnet  <- predict(model_nnet, train_small, type = "prob")

# Meta dataset
meta_small <- bind_cols(pred_rf, pred_svm, pred_rpart, pred_nnet)
colnames(meta_small) <- make.names(colnames(meta_small), unique = TRUE)
meta_small$label <- train_small$label

# Entrenar final
meta_model_small <- multinom(label ~ ., data = meta_small, trace = FALSE, MaxNWts = 5000)

output_path <- here::here("models", "final_model_100.rds")
saveRDS(meta_model_small, output_path)
cat(">>> MODELO FINAL (100 casos) guardado correctamente con los 4 algoritmos.\n")