###############################################################
# run_all.R
# Script maestro para ejecutar TODA la práctica Digit Recognition.
#
# Ejecuta en orden:
#   1_data_prep.R
#   2_eda.R (opcional)
#   3_feature_engineering.R (opcional)
#   4_models.R
#   5_tuning.R
#   6_ensemble.R
#   7_evaluate.R
#   8_save_model.R
#
# Autor: (Tu nombre)
###############################################################

# Definir las librerías necesarias para todo el proyecto
#pkgs <- c(
#  "tidyverse",    # Para manipulación de datos (la que te ha fallado)
#  "caret",        # Para Machine Learning general
#  "randomForest", # Probable que la uses en script 4 o 6
#  "e1071",        # Para SVM o funciones de ayuda
#  "xgboost",      # Si usas boosting
#  "data.table",   # Lectura rápida
#  "here"          # Ya la tienes, pero por si acaso
#)

# Instalar las que falten
#install.packages(pkgs)

# -------------------------------------------------------------
# 0. Cargar librería here (para rutas fiables y portables)
# -------------------------------------------------------------
#if (!require(here)) {
#  install.packages("here")
  library(here)
#}

cat("=====================================================\n")
cat(">>> run_all.R iniciado\n")
cat(">>> Directorio base detectado por {here}:\n")
print(here())
cat("=====================================================\n\n")

# -------------------------------------------------------------
# 1. Crear carpetas si no existen
# -------------------------------------------------------------
dirs <- c("data", "models", "results", "scripts")

for (d in dirs) {
  full <- here(d)
  if (!dir.exists(full)) {
    cat(">>> Carpeta", d, "no existe. Creándola...\n")
    dir.create(full)
  }
}

# -------------------------------------------------------------
# Función auxiliar para ejecutar scripts
# -------------------------------------------------------------
run_script <- function(file) {
  path <- here("scripts", file)
  
  cat("\n=====================================================\n")
  cat(">>> Ejecutando:", path, "\n")
  cat("=====================================================\n")
  
  start <- Sys.time()
  
  tryCatch(
    {
      source(path)
      end <- Sys.time()
      elapsed <- round(as.numeric(difftime(end, start, units = "secs")), 2)
      
      cat(">>> [OK] Script ejecutado correctamente:", file, "\n")
      cat(">>> Tiempo:", elapsed, "segundos\n")
    },
    error = function(e) {
      cat("\n>>> [ERROR] en el script:", file, "\n")
      print(e)
      stop("\n*** La ejecución se detuvo por un error. ***\n")
    }
  )
}

# -------------------------------------------------------------
# 2. Ejecución secuencial de TODOS los scripts
# -------------------------------------------------------------
scripts_to_run <- c(
  "1_data_prep.R",
  "2_eda.R",                 # opcional - descomentar si lo quieres ejecutar
  "3_feature_engineering.R", # opcional
  "4_models.R",
  "5_tunning.R",
  "6_ensemble.R",
  "7_evaluate.R",
  "8_save_model.R"
)

cat(">>> Scripts a ejecutar en orden:\n")
print(scripts_to_run)

for (s in scripts_to_run) {
  run_script(s)
}

# -------------------------------------------------------------
# 3. Fin del proceso
# -------------------------------------------------------------
cat("\n=====================================================\n")
cat(">>> TODOS LOS SCRIPTS SE EJECUTARON CORRECTAMENTE\n")
cat("=====================================================\n")

