# 8_save_model.R
# Guardar el modelo final usando SOLO los primeros 100 dígitos
# Requisito obligatorio de la práctica

library(caret)
library(nnet)
library(tidyverse)
source(here::here("scripts", "utils.R"))

# 1. Cargar datos y modelos (Usando .rds y here)
# -----------------------------------------------------------------------------
cat("Cargando datos y modelos para el entrenamiento final...\n")

train <- readRDS(here::here("data", "processed", "train_pca.rds"))
model_rf <- readRDS(here::here("models", "model_rf_tuned.rds"))
model_svm <- readRDS(here::here("models", "model_svm_tuned.rds"))

# 2. Crear dataset reducido con los primeros 100 dígitos
# -----------------------------------------------------------------------------
cat("Seleccionando los primeros 100 casos...\n")
train_small <- train[1:100, ]

# 3. Generar meta-features para estos 100 casos
# -----------------------------------------------------------------------------
cat("Generando predicciones base...\n")
pred_rf  <- predict(model_rf, train_small, type = "prob")
pred_svm <- predict(model_svm, train_small, type = "prob")

# Crear dataframe de meta-features (Igual que en script 6 y 7)
meta_small <- bind_cols(pred_rf, pred_svm)

# FIX: Usar make.names para evitar columnas duplicadas
colnames(meta_small) <- make.names(colnames(meta_small), unique = TRUE)
meta_small$label <- train_small$label

# 4. Re-entrenar el meta-modelo SOLO con estos 100 casos
# -----------------------------------------------------------------------------
cat("Entrenando modelo final reducido (Stacking)...\n")
meta_model_small <- multinom(label ~ ., data = meta_small, 
                             trace = FALSE, 
                             MaxNWts = 5000)

# 5. Guardar el modelo final
# -----------------------------------------------------------------------------
output_path <- here::here("models", "final_model_100.rds")
saveRDS(meta_model_small, output_path)

cat(">>> ÉXITO: Modelo final guardado en:", output_path, "\n")