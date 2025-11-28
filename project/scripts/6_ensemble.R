# 6_ensemble.R
library(caret)
library(nnet)
source(here::here("scripts", "utils.R"))

set.seed(123)

# Cargar datos y modelos
train <- readRDS(here::here("data", "processed", "train_pca.rds"))
model_rf <- readRDS(here::here("models", "model_rf_tuned.rds"))
model_svm <- readRDS(here::here("models", "model_svm_tuned.rds"))

# Usamos una muestra fresca para entrenar el meta-modelo
train_idx <- createDataPartition(train$label, p = 0.2, list=FALSE)
train_meta <- train[train_idx, ]

# Predicciones (Meta-features)
cat("Generando predicciones RF...\n")
pred_rf  <- predict(model_rf, train_meta, type = "prob")

cat("Generando predicciones SVM...\n")
pred_svm <- predict(model_svm, train_meta, type = "prob")

# Dataframe para el meta-modelo
# FIX: Usamos bind_cols para mayor seguridad con nombres
meta_train <- bind_cols(pred_rf, pred_svm)
# Renombrar columnas para evitar conflictos
colnames(meta_train) <- make.names(colnames(meta_train), unique = TRUE)
meta_train$label <- train_meta$label

# FIX: Eliminar filas con NA si el SVM falló en alguna predicción
meta_train <- na.omit(meta_train)

if(nrow(meta_train) == 0) stop("Error crítico: El dataframe de ensemble está vacío. Revisa Script 5.")

# Entrenar Meta-modelo
cat("Entrenando Meta-Modelo con", nrow(meta_train), "filas...\n")
meta_model <- multinom(label ~ ., data = meta_train, trace = FALSE, MaxNWts = 5000)

saveRDS(meta_model, here::here("models", "meta_model.rds"))
cat(">>> Ensemble guardado.\n")