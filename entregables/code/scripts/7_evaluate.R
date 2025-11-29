# 7_evaluate.R
library(caret)
library(nnet)
library(tidyverse)
source(here::here("scripts", "utils.R"))

# 1. Cargar validación y todos los modelos
cat(">>> Cargando modelos...\n")
valid <- readRDS(here::here("data", "processed", "valid_pca.rds"))
model_rf <- readRDS(here::here("models", "model_rf_tuned.rds"))
model_svm <- readRDS(here::here("models", "model_svm_tuned.rds"))
model_rpart <- readRDS(here::here("models", "model_rpart_tuned.rds"))
model_nnet <- readRDS(here::here("models", "model_nnet_tuned.rds"))
meta_model <- readRDS(here::here("models", "meta_model.rds"))

# 2. Generar Meta-features (necesario para evaluar el Ensemble)
cat(">>> Generando predicciones base para el Ensemble...\n")
pred_rf    <- predict(model_rf, valid, type = "prob")
pred_svm   <- predict(model_svm, valid, type = "prob")
pred_rpart <- predict(model_rpart, valid, type = "prob")
pred_nnet  <- predict(model_nnet, valid, type = "prob")

meta_valid <- bind_cols(pred_rf, pred_svm, pred_rpart, pred_nnet)
colnames(meta_valid) <- make.names(colnames(meta_valid), unique = TRUE)

# IMPORTANTE: Añadir la etiqueta real a meta_valid para que measure_perf funcione
meta_valid$label <- valid$label

# 3. Evaluación de Tiempos y Precisión (Corregido para evitar error de tipos)
cat("\n>>> Evaluando tiempos y precisión...\n")

# Función auxiliar para medir tiempo
measure_perf <- function(model, data, name){
  start <- Sys.time()
  
  # FIX: Detectar si es modelo Caret ("train") o modelo nativo (como "multinom")
  # Caret usa type="raw" para devolver clases. Multinom usa type="class".
  if (inherits(model, "train")) {
    p <- predict(model, data, type = "raw")
  } else {
    p <- predict(model, data, type = "class")
  }
  
  end <- Sys.time()
  time_taken <- as.numeric(difftime(end, start, units = "secs"))
  
  # Calcular accuracy asegurando comparación segura (character vs character)
  acc <- mean(as.character(p) == as.character(data$label))
  
  return(c(Model = name, Accuracy = round(acc, 4), Time_Secs = round(time_taken, 4)))
}

# Ejecutar comparativa
results <- rbind(
  measure_perf(model_rf, valid, "RandomForest"),
  measure_perf(model_svm, valid, "SVM"),
  measure_perf(model_rpart, valid, "Arbol"),
  measure_perf(model_nnet, valid, "NNet"),
  measure_perf(meta_model, meta_valid, "Ensemble") 
)

# Convertir a dataframe para mejor visualización
results_df <- as.data.frame(results)
print(results_df)

# Guardar tabla comparativa
write.csv(results_df, here::here("results", "performance_comparison.csv"))

# 4. Detalle Final (Matriz de Confusión Completa del Ensemble)
# Recalculamos la predicción final para mostrar el detalle de caret
final_probs <- predict(meta_model, meta_valid, type = "class")
cm <- confusionMatrix(factor(final_probs), valid$label)

cat("\n>>> ACCURACY FINAL DEL ENSEMBLE (Check):", cm$overall["Accuracy"], "\n")

# Guardar salida detallada
sink(here::here("results", "evaluation.txt"))
cat("--- COMPARATIVA DE TIEMPOS Y ACCURACY ---\n")
print(results_df)
cat("\n\n--- MATRIZ DE CONFUSIÓN DETALLADA (ENSEMBLE) ---\n")
print(cm)
sink()

cat(">>> Evaluación completada. Resultados en carpeta 'results/'.\n")