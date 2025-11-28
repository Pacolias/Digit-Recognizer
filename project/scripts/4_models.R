# 4_models.R
library(tidyverse)
library(caret)
library(randomForest)
library(nnet)
library(e1071)
library(xgboost)
source(here::here("scripts", "utils.R"))
set_seed(42)

# Cargar datos PCA (Rutas corregidas)
train <- readRDS(here::here("data", "processed", "train_pca.rds"))
valid <- readRDS(here::here("data", "processed", "valid_pca.rds"))

trctrl <- trainControl(method = "none", classProbs = TRUE)

# 1) Random Forest
cat("Entrenando Random Forest...\n")
rf_model <- randomForest(x = train %>% select(-label), y = train$label, ntree = 50) # Reducido a 50 para rapidez

# 2) Multinomial (glmnet)
cat("Entrenando multinomial...\n")
glmnet_model <- train(label ~ ., data = train, method = "multinom", trace = FALSE, trControl = trctrl)

# 3) NNet
cat("Entrenando nnet...\n")
nnet_model <- nnet::multinom(label ~ ., data = train, maxit = 100, trace=FALSE)

# Guardar modelos base (Ruta corregida)
saveRDS(list(rf = rf_model, glmnet = glmnet_model, nnet = nnet_model), 
        here::here("models", "models_base.rds"))

cat(">>> Modelos base entrenados y guardados en models/models_base.rds\n")