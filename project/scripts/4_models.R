# 4_models.R
library(tidyverse)
library(caret)
library(randomForest)
library(nnet)
library(e1071)
library(xgboost)
source(here::here("scripts", "utils.R"))
set_seed(42)

# Cargar datos PCA (Rutas corregidas con here)
# Nota: Asegúrate de que 3_feature_engineering se ejecutó bien antes
train <- readRDS(here::here("data", "processed", "train_pca.rds"))
valid <- readRDS(here::here("data", "processed", "valid_pca.rds"))

# Control de entrenamiento simple
trctrl <- trainControl(method = "none", classProbs = TRUE)

# -----------------------------------------------------------------------------
# 1) Random Forest
# -----------------------------------------------------------------------------
cat("Entrenando Random Forest...\n")
# ntree=50 es suficiente para pruebas. Súbelo a 100-500 para el modelo final.
rf_model <- randomForest(x = train %>% select(-label), y = train$label, ntree = 50)


# -----------------------------------------------------------------------------
# 2) Multinomial (glmnet)
# -----------------------------------------------------------------------------
cat("Entrenando multinomial...\n")
# MaxNWts añadido para evitar el error "too many weights"
glmnet_model <- train(label ~ ., data = train, 
                      method = "multinom", 
                      trace = FALSE, 
                      trControl = trctrl,
                      MaxNWts = 10000) # <--- FIX AQUI


# -----------------------------------------------------------------------------
# 3) NNet (Red Neuronal Simple)
# -----------------------------------------------------------------------------
cat("Entrenando nnet...\n")
# MaxNWts añadido aquí también
nnet_model <- nnet::multinom(label ~ ., data = train, 
                             maxit = 100, 
                             trace = FALSE, 
                             MaxNWts = 10000) # <--- FIX AQUI

# -----------------------------------------------------------------------------
# Guardar modelos
# -----------------------------------------------------------------------------
cat("Guardando modelos en models/models_base.rds ...\n")
saveRDS(list(rf = rf_model, glmnet = glmnet_model, nnet = nnet_model), 
        here::here("models", "models_base.rds"))

cat(">>> Script 4 Completado con éxito.\n")