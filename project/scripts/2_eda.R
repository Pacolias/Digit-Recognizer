# 2_eda.R
# Análisis exploratorio: distribución clases, ejemplos, sparsity


library(tidyverse)
source(here::here("scripts", "utils.R"))

train_set <- readRDS(here::here("data", "processed", "train_processed.rds"))


# Distribución de clases
dist <- train_set %>% count(label) %>% arrange(label)
print(dist)


# Mostrar ejemplos por clase
par(mfrow=c(2,5), mar=c(0.5,0.5,2,0.5))
for(d in 0:9){
  row <- train_set %>% filter(label == as.character(d)) %>% slice(1) %>% select(-label)
  plot_mnist_row(as.numeric(row), title = paste0("Clase: ", d))
}
par(mfrow=c(1,1))


# Sparsity: porcentaje de píxeles no nulos medianos por clase
sparsity <- train_set %>% group_by(label) %>% summarize(nonzero = mean(rowSums(across(starts_with("pixel")) > 0) / 784))
print(sparsity)


# Histograma del número de píxeles activos
train_set %>%
  rowwise() %>%
  mutate(active = sum(c_across(starts_with("pixel")) > 0)) %>%
  ungroup() %>%
  ggplot(aes(active)) + geom_histogram(bins=50) + labs(title="Pixels activos por imagen")