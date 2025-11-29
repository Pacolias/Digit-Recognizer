# 4_models.R
# Entrenamiento rápido de modelos base (Baseline)
# Se entrenan los 4 algoritmos obligatorios con parámetros por defecto

library(tidyverse)
library(caret)
library(randomForest)
library(nnet)
library(e1071)
library(rpart) # Necesario para el árbol
source(here::here("scripts", "utils.R"))

set_seed(42)

# Cargar datos PCA
train <- readRDS(here::here("data", "processed", "train_pca.rds"))
valid <- readRDS(here::here("data", "processed", "valid_pca.rds"))

# Control simple (sin CV costoso, solo entrenamiento)
trctrl <- trainControl(method = "none", classProbs = TRUE)

# -----------------------------------------------------------------------------
# 1. Árbol de Decisión (rpart)
# -----------------------------------------------------------------------------
cat("Entrenando Árbol de Decisión (Baseline)...\n")
model_rpart_base <- train(label ~ ., data = train, 
                          method = "rpart", 
                          trControl = trctrl, 
                          tuneLength = 1) # Usar defecto

# -----------------------------------------------------------------------------
# 2. Random Forest
# -----------------------------------------------------------------------------
cat("Entrenando Random Forest (Baseline)...\n")
# ntree=50 para rapidez en esta fase inicial
model_rf_base <- train(label ~ ., data = train, 
                       method = "rf", 
                       trControl = trctrl, 
                       tuneGrid = data.frame(mtry = 4), # Valor fijo simple
                       ntree = 50) 

# -----------------------------------------------------------------------------
# 3. SVM Lineal
# -----------------------------------------------------------------------------
cat("Entrenando SVM (Baseline)...\n")
model_svm_base <- train(label ~ ., data = train, 
                        method = "svmLinear", 
                        trControl = trctrl, 
                        tuneGrid = data.frame(C = 1)) # C=1 por defecto

# -----------------------------------------------------------------------------
# 4. Perceptrón Multicapa (NNet)
# -----------------------------------------------------------------------------
cat("Entrenando Perceptrón Multicapa (Baseline)...\n")
model_nnet_base <- train(label ~ ., data = train, 
                         method = "nnet", 
                         trControl = trctrl, 
                         tuneGrid = data.frame(size = 5, decay = 0.1), # Config básica
                         MaxNWts = 5000, 
                         trace = FALSE)

# -----------------------------------------------------------------------------
# Evaluación Rápida (Opcional, para ver cómo van)
# -----------------------------------------------------------------------------
cat("\n--- Resultados Preliminares (Accuracy en Validación) ---\n")
preds <- list(
  Tree = predict(model_rpart_base, valid),
  RF   = predict(model_rf_base, valid),
  SVM  = predict(model_svm_base, valid),
  MLP  = predict(model_nnet_base, valid)
)

for(m in names(preds)){
  acc <- mean(preds[[m]] == valid$label)
  cat(m, ":", round(acc, 4), "\n")
}

# Guardar
saveRDS(list(rpart=model_rpart_base, rf=model_rf_base, svm=model_svm_base, nnet=model_nnet_base), 
        here::here("models", "models_base.rds"))

cat(">>> Modelos base guardados. Ahora ejecuta 5_tunning.R para optimizarlos.\n")