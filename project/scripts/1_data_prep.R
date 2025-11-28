# 1_data_prep.R
library(tidyverse)
library(caret)
source(here::here("scripts", "utils.R"))

set_seed(42)

# 1. Cargar datos
train_raw <- read.csv(here::here("data", "raw", "train.csv"))

# 2. Comprobaciones básicas
stopifnot(ncol(train_raw) == 785)

# Normalizar nombre de la etiqueta
names(train_raw)[1] <- "label"
train_raw$label <- as.factor(train_raw$label)

# --------------------------------------------------------------------------
# FIX PARA ERROR "VALID R VARIABLE NAME" (Script 4)
# Caret da error si las clases son números ("0", "1"). 
# Las cambiamos a "L0", "L1", etc.
# --------------------------------------------------------------------------
levels(train_raw$label) <- paste0("L", levels(train_raw$label))
# --------------------------------------------------------------------------

# 3. Separar y Escalar
pixels <- train_raw %>% select(starts_with("pixel"))
labels <- train_raw$label

pixels_scaled <- pixels / 255

# Combinar de nuevo
train_proc <- bind_cols(label = labels, pixels_scaled)

# ==============================================================================
# PASO DE SEGURIDAD RAM: Reducir el dataset (Solo para pruebas)
# ==============================================================================
cat(">>> REDUCIENDO DATASET a 5000 filas para evitar colapso de memoria...\n")
set.seed(123) 
filas_seguras <- sample(nrow(train_proc), 5000)
train_proc <- train_proc[filas_seguras, ] 
# ==============================================================================

# 4. Particionar: train (80%) / validation (20%)
idx <- createDataPartition(train_proc$label, p = 0.8, list = FALSE)
train_set <- train_proc[idx, ]
valid_set <- train_proc[-idx, ]

cat("Dimensiones train:", dim(train_set), "\n")
cat("Dimensiones valid:", dim(valid_set), "\n")

# 5. Guardar
saveRDS(train_set, here::here("data", "processed", "train_processed.rds"))
saveRDS(valid_set, here::here("data", "processed", "valid_processed.rds"))

cat(">>> Datos procesados (REDUCIDOS y RENOMBRADOS) guardados.\n")