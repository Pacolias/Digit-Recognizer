# 🖐️ Digit Recognition Pipeline en R

![R](https://img.shields.io/badge/Language-R-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Data](https://img.shields.io/badge/Data-MNIST-lightgrey)

Este proyecto implementa un flujo de trabajo (pipeline) completo de Machine Learning para la clasificación de dígitos manuscritos (MNIST Dataset). El sistema está construido de forma modular en **R**, utilizando técnicas de reducción de dimensionalidad (**PCA**) y un **Stacking Ensemble** (Random Forest + SVM + Árbol de decisión + Perceptrón Multicapa) para maximizar la precisión.

## 📂 Estructura del Proyecto

El proyecto está organizado para garantizar la reproducibilidad y el orden. La ejecución se controla desde un script maestro.

```text
project/
├── .here                # Archivo ancla para rutas relativas (¡Importante!)
├── run_all.R            # 🚀 SCRIPT MAESTRO: Ejecuta todo el pipeline
├── data/
│   ├── raw/             # Datos originales (train.csv, test.csv)
│   └── processed/       # Datos limpios y transformados (.rds)
├── models/              # Modelos entrenados (.rds)
├── results/             # Gráficos y métricas de evaluación
└── scripts/
    ├── 1_data_prep.R            # Limpieza, normalización y sampling
    ├── 2_eda.R                  # Análisis Exploratorio de Datos
    ├── 3_feature_engineering.R  # PCA y selección de variables
    ├── 4_models.R               # Entrenamiento de modelos base
    ├── 5_tunning.R              # Ajuste de hiperparámetros
    ├── 6_ensemble.R             # Creación del Stacking Ensemble
    ├── 7_evaluate.R             # Evaluación final
    ├── 8_save_model.R           # Exportación del modelo final (requisito)
    └── utils.R                  # Funciones auxiliares
