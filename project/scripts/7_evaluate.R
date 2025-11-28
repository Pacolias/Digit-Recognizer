# 7_evaluate.R
library(caret)
library(nnet)
library(tidyverse)
source(here::here("scripts", "utils.R"))

# Cargar validación y todos los modelos
valid <- readRDS(here::here("data", "processed", "valid_pca.rds"))
model_rf <- readRDS(here::here("models", "model_rf_tuned.rds"))
model_svm <- readRDS(here::here("models", "model_svm_tuned.rds"))
model_rpart <- readRDS(here::here("models", "model_rpart_tuned.rds"))
model_nnet <- readRDS(here::here("models", "model_nnet_tuned.rds"))
meta_model <- readRDS(here::here("models", "meta_model.rds"))

cat("Evaluando...\n")

# Predicciones base
pred_rf    <- predict(model_rf, valid, type = "prob")
pred_svm   <- predict(model_svm, valid, type = "prob")
pred_rpart <- predict(model_rpart, valid, type = "prob")
pred_nnet  <- predict(model_nnet, valid, type = "prob")

# Meta-features
meta_valid <- bind_cols(pred_rf, pred_svm, pred_rpart, pred_nnet)
colnames(meta_valid) <- make.names(colnames(meta_valid), unique = TRUE)

# Predicción Final
final_probs <- predict(meta_model, meta_valid, type = "class")

# Resultados
cm <- confusionMatrix(factor(final_probs), valid$label)
cat(">>> ACCURACY FINAL DEL ENSEMBLE:", cm$overall["Accuracy"], "\n")

sink(here::here("results", "evaluation.txt"))
print(cm)
sink()