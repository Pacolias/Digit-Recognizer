###############################################################
# 8_save_model.R
# Guardar el modelo final usando SOLO los primeros 100 dígitos
# Requisito obligatorio de la práctica
###############################################################

library(caret)
library(nnet)

# Cargar modelos entrenados
load("data/processed_train.RData")
load("models/model_rpart_tuned.RData")
load("models/model_rf_tuned.RData")
load("models/model_svm_tuned.RData")
load("models/model_nnet_tuned.RData")
load("models/meta_model.RData")

###############################################################
# Crear dataset reducido con los primeros 100 dígitos
###############################################################

train_small <- train[1:100, ]

# Generar meta-features para estos 100 casos
pred_rpart <- predict(model_rpart_tuned, train_small, type = "prob")
pred_rf    <- predict(model_rf_tuned, train_small, type = "prob")
pred_svm   <- predict(model_svm_tuned, train_small, type = "prob")
pred_nnet  <- predict(model_nnet_tuned, train_small, type = "prob")

meta_small <- data.frame(
  pred_rpart,
  pred_rf,
  pred_svm,
  pred_nnet,
  label = train_small$label
)

###############################################################
# Re-entrenar el meta-modelo SOLO con estos 100 casos
###############################################################

meta_model_small <- multinom(label ~ ., data = meta_small, MaxNWts = 50000)

###############################################################
# Guardar el modelo final como pide la práctica
###############################################################

save(meta_model_small, file = "models/final_model_100.RData")
saveRDS(meta_model_small, file = "models/final_model_100.rds")

###############################################################
# Fin
###############################################################
