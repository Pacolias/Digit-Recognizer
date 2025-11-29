# 2_eda.R
# Análisis exploratorio: distribución clases, ejemplos, sparsity

library(tidyverse)
# No cargamos xgboost aquí para evitar conflictos con 'slice'
source(here::here("scripts", "utils.R"))

# Cargar datos procesados (asegura usar readRDS y here)
train_set <- readRDS(here::here("data", "processed", "train_processed.rds"))

cat(">>> Datos cargados. Filas:", nrow(train_set), "\n")

# 1. Distribución de clases
dist <- train_set %>% count(label) %>% arrange(label)
print(dist)

# 2. Mostrar ejemplos por clase
# Configuramos el panel de gráficos
par(mfrow=c(2,5), mar=c(0.5,0.5,2,0.5))

for(d in 0:9){
  # FIX: Ahora las etiquetas son "L0", "L1"... así que buscamos paste0("L", d)
  target_label <- paste0("L", d)
  
  # FIX: Usamos head(1) en lugar de slice(1) para evitar conflicto con xgboost
  row <- train_set %>% 
    filter(label == target_label) %>% 
    head(1) %>% 
    select(-label)
  
  if(nrow(row) > 0) {
    plot_mnist_row(as.numeric(row), title = paste0("Clase: ", target_label))
  } else {
    cat("Advertencia: No se encontraron ejemplos para la clase", target_label, "\n")
  }
}

# Restaurar panel gráfico
par(mfrow=c(1,1))


# 3. Sparsity: porcentaje de píxeles no nulos medianos por clase
sparsity <- train_set %>% 
  group_by(label) %>% 
  summarize(nonzero = mean(rowSums(across(starts_with("pixel")) > 0) / 784))

print(sparsity)


# 4. Histograma del número de píxeles activos
cat("Generando histograma de pixeles activos...\n")
p <- train_set %>%
  rowwise() %>%
  mutate(active = sum(c_across(starts_with("pixel")) > 0)) %>%
  ungroup() %>%
  ggplot(aes(active)) + 
  geom_histogram(bins=50, fill="steelblue", color="white") + 
  labs(title="Píxeles activos por imagen", x="Nº Píxeles > 0", y="Frecuencia")

# Guardar el gráfico en results para no perderlo
ggsave(here::here("results", "pixel_histogram.png"), p)

cat(">>> Script 2 completado. Gráfico guardado en results/pixel_histogram.png\n")