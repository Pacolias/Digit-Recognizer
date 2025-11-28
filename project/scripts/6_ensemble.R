# 6_ensemble.R
library(caret)
library(nnet)
source(here::here("scripts", "utils.R"))

set.seed(123)

# Cargar datos y modelos
train <- readRDS(here::here("data", "processed", "train_pca.rds"))
model_rf <- readRDS(here::here("models", "model_rf_tuned.rds"))
model_svm <- readRDS(here::here("models", "model_svm_tuned.rds"))

# Usamos una muestra para entrenar el meta-modelo
train_idx <- createDataPartition(train$label, p = 0.2, list=FALSE)
train_meta <- train[train_idx, ]

# Predicciones (Meta-features)
cat("Generando predicciones para stacking...\n")
pred_rf  <- predict(model_rf, train_meta, type = "prob")
pred_svm <- predict(model_svm, train_meta, type = "prob")

# Dataframe para el meta-modelo
meta_train <- data.frame(pred_rf, pred_svm, label = train_meta$label)

# Entrenar Meta-modelo
cat("Entrenando Meta-Modelo...\n")
meta_model <- multinom(label ~ ., data = meta_train, trace = FALSE)

saveRDS(meta_model, here::here("models", "meta_model.rds"))
cat(">>> Ensemble guardado.\n")