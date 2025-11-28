# 1_data_prep.R
library(tidyverse)
library(caret)
source(here::here("scripts", "utils.R"))

set_seed(42)

# 1. Cargar datos (Usando here para rutas seguras)
train_raw <- read.csv(here::here("data", "raw", "train.csv"))
# test_raw  <- read.csv(here::here("data", "raw", "test.csv")) # Descomentar si usas Kaggle submission

# 2. Comprobaciones básicas
# Nota: train.csv tiene 785 columnas (label + 784 pixels)
stopifnot(ncol(train_raw) == 785)

# Normalizar nombre de la etiqueta
names(train_raw)[1] <- "label"

# Convertir label a factor
train_raw$label <- as.factor(train_raw$label)

# 3. Separar y Escalar
pixels <- train_raw %>% select(starts_with("pixel"))
labels <- train_raw$label

# Escalado a [0,1]
pixels_scaled <- pixels / 255

# Combinar de nuevo
train_proc <- bind_cols(label = labels, pixels_scaled)

# 4. Particionar: train (80%) / validation (20%)
idx <- createDataPartition(train_proc$label, p = 0.8, list = FALSE)
train_set <- train_proc[idx, ]
valid_set <- train_proc[-idx, ]

cat("Dimensiones train:", dim(train_set), "\n")
cat("Dimensiones valid:", dim(valid_set), "\n")

# 5. Guardar (IMPORTANTE: Usamos .rds y here::here)
saveRDS(train_set, here::here("data", "processed", "train_processed.rds"))
saveRDS(valid_set, here::here("data", "processed", "valid_processed.rds"))

cat(">>> Datos procesados guardados en data/processed/\n")