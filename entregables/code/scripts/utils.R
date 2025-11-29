# utils.R
# Funciones utilitarias: carga, muestreo reproducible, mostrar dígitos, guardado


library(tidyverse)
library(caret)


# Establecer semilla reproducible
set_seed <- function(s=123){
  set.seed(s)
}


# Cargar CSV (espera formatos tipo Kaggle: train.csv con label y 784 pixeles)
load_csv <- function(path){
  df <- readr::read_csv(path, col_types = cols())
  return(df)
}


# Mostrar una imagen MNIST a partir de una fila (vector de 784 pixeles)
plot_mnist_row <- function(row_vec, title = NULL){
  # row_vec: numeric vector length 784 (o data frame row)
  if(is.data.frame(row_vec)) row_vec <- as.numeric(row_vec)
  mat <- matrix(row_vec, nrow = 28, byrow = TRUE)
  # rotar para que se vea correctamente
  mat <- t(apply(mat, 2, rev))
  op <- par(mar=c(0,0,2,0))
  image(mat, axes = FALSE, main = title)
  par(op)
}


# Guardar modelo y metadatos
save_model <- function(model, filename="final_model.rds"){
  saveRDS(model, filename)
}


# Cargar modelo
load_model <- function(filename){
  readRDS(filename)
}


# Convertir dataframe de pixeles a scaled numeric matrix [0,1]
pixels_to_matrix <- function(df_pixels){
  # df_pixels: data.frame con columnas pixel0..pixel783
  mat <- as.matrix(df_pixels)
  mat <- mat / 255
  return(mat)
}