# 3_feature_engineering.R
library(tidyverse)
library(caret)
source(here::here("scripts", "utils.R"))
set_seed(42)

# Cargar datos procesados
train_set <- readRDS(here::here("data", "processed", "train_processed.rds"))
valid_set <- readRDS(here::here("data", "processed", "valid_processed.rds"))

# 1) Selección por varianza
pixel_cols <- names(train_set)[names(train_set) != "label"]
variances <- apply(train_set %>% select(-label), 2, var)
keep_idx <- which(variances > 1e-6)
keep_pixels <- pixel_cols[keep_idx]

train_var <- train_set %>% select(label, any_of(keep_pixels))
valid_var <- valid_set %>% select(label, any_of(keep_pixels))

# Guardar VAR
readr::write_rds(train_var, here::here("data", "processed", "train_var.rds"))
readr::write_rds(valid_var, here::here("data", "processed", "valid_var.rds"))

# 2) PCA
pca_ctrl <- preProcess(train_var %>% select(-label), method = c("center","scale","pca"), thresh = 0.95)
train_pca <- predict(pca_ctrl, train_var %>% select(-label))
valid_pca <- predict(pca_ctrl, valid_var %>% select(-label))

train_pca <- bind_cols(label = train_var$label, as.data.frame(train_pca))
valid_pca <- bind_cols(label = valid_var$label, as.data.frame(valid_pca))

# Guardar PCA
saveRDS(pca_ctrl, here::here("models", "pca_ctrl.rds"))
readr::write_rds(train_pca, here::here("data", "processed", "train_pca.rds"))
readr::write_rds(valid_pca, here::here("data", "processed", "valid_pca.rds"))

cat(">>> Feature Engineering completado y guardado.\n")