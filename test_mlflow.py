#!/usr/bin/env python
"""
Script de ejemplo para generar datos de prueba en MLflow UI.

Crea un experimento llamado "Contador-Vehiculos-Demo" con varios runs
de regresión logística, variando hiperparámetros y registrando:
- Parámetros
- Métricas (accuracy, precision, recall, f1)
- Modelo (sklearn)
- Tags de contexto
"""

from __future__ import annotations

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def create_sample_experiment() -> None:
    """Crea un experimento de demo en MLflow con regresión logística."""
    experiment_name = "Contador-Vehiculos-Demo"
    mlflow.set_experiment(experiment_name)

    print(f"📂 Creando experimento: {experiment_name}")

    # --------------------------------------------------------------
    # Generar dataset sintético
    # --------------------------------------------------------------
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --------------------------------------------------------------
    # Ejecutar varios runs con diferentes hiperparámetros
    # --------------------------------------------------------------
    solvers = ["liblinear", "lbfgs"]
    C_values = [0.1, 1.0, 10.0]

    for solver in solvers:
        for C in C_values:
            with mlflow.start_run():
                # Registrar parámetros
                mlflow.log_param("solver", solver)
                mlflow.log_param("C", C)
                mlflow.log_param("max_iter", 100)

                # Entrenar modelo
                model = LogisticRegression(
                    solver=solver, C=C, max_iter=100, random_state=42
                )
                model.fit(X_train, y_train)

                # Predicciones
                y_pred = model.predict(X_test)

                # Calcular métricas
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="weighted")
                recall = recall_score(y_test, y_pred, average="weighted")
                f1 = f1_score(y_test, y_pred, average="weighted")

                # Registrar métricas
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)

                # Registrar modelo
                mlflow.sklearn.log_model(model, "model")

                # Tags de contexto
                mlflow.set_tag("model_type", "LogisticRegression")
                mlflow.set_tag("dataset", "synthetic")

                print(
                    f"✅ Run completado: solver={solver}, C={C}, "
                    f"accuracy={accuracy:.4f}"
                )


if __name__ == "__main__":
    create_sample_experiment()
    print("\n🎉 ¡Experimento de demo creado exitosamente!")
    print("👉 Ahora puedes verlo en MLflow UI: http://localhost:5000")
