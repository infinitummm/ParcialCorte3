#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# IMPLEMENTACIÓN COMPLETA: REGRESIÓN LINEAL EN PYTHON
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import time
import sys
from typing import Tuple, List

class LinearRegression:
    """Modelo de regresión lineal implementado en Python con NumPy"""
    
    def __init__(self, learning_rate: float = 0.01):
        """
        Inicializa el modelo de regresión lineal
        
        Args:
            learning_rate: Tasa de aprendizaje (típicamente 0.001 - 0.1)
        """
        self.w = 0.0
        self.b = 0.0
        self.learning_rate = learning_rate
        self.epochs_trained = 0
        self.mse_history = []
    
    def compute_gradient(self, x_batch: np.ndarray, y_batch: np.ndarray) -> Tuple[float, float, float]:
        """
        Calcula gradientes para un mini-batch
        
        Args:
            x_batch: Vector de características
            y_batch: Vector de etiquetas
            
        Returns:
            Tupla (dw, db, mse) con gradientes y error
        """
        m = len(x_batch)
        
        # y_pred = w * x + b
        y_pred = self.w * x_batch + self.b
        
        # error = y_pred - y
        error = y_pred - y_batch
        
        # Gradientes usando NumPy
        dw = (2/m) * np.dot(error, x_batch)  # sum(error * x) * 2/m
        db = (2/m) * np.sum(error)            # sum(error) * 2/m
        
        # MSE
        mse = np.mean(error ** 2)
        
        return dw, db, mse
    
    def update_weights(self, dw: float, db: float) -> None:
        """
        Actualiza los parámetros usando gradiente descendente
        
        Args:
            dw: Gradiente respecto a w
            db: Gradiente respecto a b
        """
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db
    
    def train_epoch(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Entrena el modelo una época
        
        Args:
            x: Vector de características
            y: Vector de etiquetas
            
        Returns:
            Error cuadrático medio (MSE) de esta época
        """
        dw, db, mse = self.compute_gradient(x, y)
        self.update_weights(dw, db)
        return mse
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int) -> None:
        """
        Entrena el modelo por N épocas
        
        Args:
            x: Vector de características
            y: Vector de etiquetas
            epochs: Número de épocas de entrenamiento
        """
        for epoch in range(epochs):
            mse = self.train_epoch(x, y)
            self.mse_history.append(mse)
            self.epochs_trained = epoch + 1
            
            # Mostrar progreso cada 200 épocas
            if (epoch + 1) % 200 == 0:
                print(f"Epoch {epoch+1}, MSE: {mse:.4f}, w: {self.w:.4f}, b: {self.b:.4f}")
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza predicción para una entrada
        
        Args:
            x: Valor(es) de entrada
            
        Returns:
            Valor(es) predicho(s): y = w*x + b
        """
        return self.w * x + self.b
    
    def get_params(self) -> Tuple[float, float]:
        """Retorna los parámetros del modelo"""
        return self.w, self.b
    
    def set_params(self, w: float, b: float) -> None:
        """Establece los parámetros del modelo"""
        self.w = w
        self.b = b
    
    def get_mse_history(self) -> List[float]:
        """Retorna el histórico de MSE"""
        return self.mse_history


def benchmark_small_dataset():
    """Benchmark 1: Dataset pequeño (5 muestras, 1000 épocas)"""
    print("\n" + "="*70)
    print("BENCHMARK 1: Dataset Pequeño (5 muestras, 1000 épocas)")
    print("="*70 + "\n")
    
    # Datos de ejemplo (idénticos a Rust)
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([2, 4, 6, 8, 10], dtype=float)
    
    learning_rate = 0.01
    epochs = 1000
    
    model = LinearRegression(learning_rate)
    
    start_time = time.perf_counter()
    model.train(x, y, epochs)
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    w, b = model.get_params()
    
    print("\nModelo entrenado:")
    print(f"  w ≈ {w:.4f}")
    print(f"  b ≈ {b:.4f}")
    
    print("\nTiempos:")
    print(f"  Total: {duration:.6f} segundos")
    print(f"  Por época: {duration * 1000 / epochs:.6f} ms")
    print(f"  Throughput: {epochs / duration:.0f} epochs/sec")
    
    x_nuevo = 7.0
    y_pred = model.predict(x_nuevo)
    print(f"\nPredicción:")
    print(f"  Para x = {x_nuevo}, y_pred ≈ {y_pred:.4f}")
    
    mse_history = model.get_mse_history()
    print(f"\nErrores:")
    print(f"  MSE inicial: {mse_history[0]:.6f}")
    print(f"  MSE final: {mse_history[-1]:.6f}")
    print(f"  Reducción: {(1 - mse_history[-1] / mse_history[0]) * 100:.2f}%")


def benchmark_medium_dataset():
    """Benchmark 2: Dataset mediano (1000 muestras, 100 épocas)"""
    print("\n" + "="*70)
    print("BENCHMARK 2: Dataset Mediano (1000 muestras, 100 épocas)")
    print("="*70 + "\n")
    
    # Generar dataset: y = 2*x + 0.5 + ruido
    x = np.arange(1, 1001, dtype=float)
    noise = np.array([((i * 7919) % 100) / 100.0 - 0.5 for i in range(1, 1001)])
    y = 2.0 * x + 0.5 + noise
    
    learning_rate = 0.00001
    epochs = 100
    
    model = LinearRegression(learning_rate)
    
    start_time = time.perf_counter()
    model.train(x, y, epochs)
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    w, b = model.get_params()
    
    print(f"\nDataset: {len(x)} muestras")
    print(f"\nModelo entrenado:")
    print(f"  w ≈ {w:.4f} (esperado: 2.0)")
    print(f"  b ≈ {b:.4f} (esperado: 0.5)")
    
    print(f"\nTiempos:")
    print(f"  Total: {duration:.6f} segundos")
    print(f"  Por época: {duration * 1000 / epochs:.6f} ms")
    print(f"  Throughput: {len(x) * epochs / (duration * 1000):.0f} samples/ms")
    
    mse_history = model.get_mse_history()
    print(f"\nErrores:")
    print(f"  MSE inicial: {mse_history[0]:.6f}")
    print(f"  MSE final: {mse_history[-1]:.6f}")


def benchmark_large_dataset():
    """Benchmark 3: Dataset grande (10000 muestras, 100 épocas)"""
    print("\n" + "="*70)
    print("BENCHMARK 3: Dataset Grande (10000 muestras, 100 épocas)")
    print("="*70 + "\n")
    
    # Generar dataset
    x = np.arange(1, 10001, dtype=float)
    noise = np.array([((i * 7919) % 100) / 100.0 - 0.5 for i in range(1, 10001)])
    y = 2.0 * x + 0.5 + noise
    
    learning_rate = 0.00001
    epochs = 100
    
    model = LinearRegression(learning_rate)
    
    start_time = time.perf_counter()
    model.train(x, y, epochs)
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    w, b = model.get_params()
    
    print(f"\nDataset: {len(x)} muestras")
    print(f"\nModelo entrenado:")
    print(f"  w ≈ {w:.4f}")
    print(f"  b ≈ {b:.4f}")
    
    print(f"\nTiempos:")
    print(f"  Total: {duration:.6f} segundos")
    print(f"  Por época: {duration * 1000 / epochs:.6f} ms")
    print(f"  Throughput: {len(x) * epochs / (duration * 1000):.0f} samples/ms")
    
    mse_history = model.get_mse_history()
    print(f"\nErrores:")
    print(f"  MSE inicial: {mse_history[0]:.6f}")
    print(f"  MSE final: {mse_history[-1]:.6f}")


def benchmark_scalability():
    """Benchmark 4: Análisis de escalabilidad"""
    print("\n" + "="*70)
    print("BENCHMARK 4: Análisis de Escalabilidad")
    print("="*70 + "\n")
    
    datasets = [
        (100, 100),
        (1_000, 100),
        (10_000, 100),
        (100_000, 10),
    ]
    
    print(f"{'Tamaño':<12} {'Épocas':<12} {'Tiempo (ms)':<12} {'Throughput':<15}")
    print("-" * 55)
    
    for size, epochs in datasets:
        x = np.arange(1, size + 1, dtype=float)
        noise = np.array([((i * 7919) % 100) / 100.0 - 0.5 for i in range(1, size + 1)])
        y = 2.0 * x + 0.5 + noise
        
        model = LinearRegression(0.00001)
        
        start_time = time.perf_counter()
        model.train(x, y, epochs)
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        total_ops = (size * epochs) / 1000.0
        time_ms = duration * 1000.0
        throughput = total_ops / time_ms
        
        print(f"{size:<12} {epochs:<12} {time_ms:<12.4f} {throughput:<15.0f}")


def benchmark_microbenchmarks():
    """Benchmark 5: Micro-benchmarks de operaciones individuales"""
    print("\n" + "="*70)
    print("BENCHMARK 5: Micro-benchmarks")
    print("="*70 + "\n")
    
    # Test 1: Multiplicación escalar (NumPy)
    print("Test 1: Multiplicación escalar (1M operaciones con NumPy)")
    start = time.perf_counter()
    result = 1.5
    for _ in range(1_000_000):
        result = result * 2.5
    duration = time.perf_counter() - start
    print(f"  Tiempo: {duration * 1_000_000:.6f} µs")
    print(f"  Tiempo/op: {duration * 1_000_000_000 / 1_000_000:.4f} ns\n")
    
    # Test 2: Dot product con NumPy
    print("Test 2: Dot product NumPy (1000 elementos, 100 veces)")
    a = np.arange(1000, dtype=float)
    b = a * 2.0
    
    start = time.perf_counter()
    for _ in range(100):
        _ = np.dot(a, b)
    duration = time.perf_counter() - start
    print(f"  Tiempo: {duration * 1000:.6f} ms")
    print(f"  Tiempo/op: {duration * 1_000_000 / 100:.4f} µs\n")
    
    # Test 3: Suma de array con NumPy
    print("Test 3: Suma de array NumPy (1000 elementos, 1000 veces)")
    c = np.arange(1000, dtype=float)
    
    start = time.perf_counter()
    for _ in range(1000):
        _ = np.sum(c)
    duration = time.perf_counter() - start
    print(f"  Tiempo: {duration * 1000:.6f} ms")
    print(f"  Tiempo/op: {duration * 1_000_000 / 1000:.4f} µs\n")


def main():
    print("\n╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "REGRESIÓN LINEAL - IMPLEMENTACIÓN EN PYTHON".center(68) + "║")
    print("║" + "Benchmarks y Análisis de Desempeño".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")
    
    # Ejecutar benchmarks
    benchmark_small_dataset()
    benchmark_medium_dataset()
    benchmark_large_dataset()
    benchmark_scalability()
    benchmark_microbenchmarks()
    
    print("\n" + "="*70)
    print("FIN DE BENCHMARKS")
    print("="*70)


if __name__ == "__main__":
    main()
