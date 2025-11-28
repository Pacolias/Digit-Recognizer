# 7_evaluate.R
library(caret)
library(nnet)
source(here::here("scripts", "utils.R"))

# Cargar Validacion y Modelos
valid <- readRDS(here::here("data", "processed", "valid_pca.rds"))
model_rf <- readRDS(here::here("models", "model_rf_tuned.rds"))
model_svm <- readRDS(here::here("models", "model_svm_tuned.rds"))
meta_model <- readRDS(here::here("models", "meta_model.rds"))

cat("Evaluando en set de validación...\n")

# Generar predicciones base
pred_rf  <- predict(model_rf, valid, type = "prob")
pred_svm <- predict(model_svm, valid, type = "prob")

# Crear dataset meta
meta_valid <- data.frame(pred_rf, pred_svm)

# Predicción Final
final_probs <- predict(meta_model, meta_valid, type = "class")

# Matriz Confusión
cm <- confusionMatrix(factor(final_probs), valid$label)
print(cm$overall["Accuracy"])

# Guardar resultado
sink(here::here("results", "evaluation.txt"))
print(cm)
sink()