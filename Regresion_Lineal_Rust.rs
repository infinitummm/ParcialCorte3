// ═══════════════════════════════════════════════════════════════════════════════
// IMPLEMENTACIÓN COMPLETA: REGRESIÓN LINEAL EN RUST
// ═══════════════════════════════════════════════════════════════════════════════

use std::time::Instant;
use std::f64;

/// Estructura del modelo de regresión lineal
#[derive(Debug, Clone)]
struct LinearRegression {
    w: f64,                      // Peso/pendiente
    b: f64,                      // Bias/intercepto
    learning_rate: f64,          // Tasa de aprendizaje
    epochs_trained: u32,         // Contador de épocas entrenadas
    mse_history: Vec<f64>,       // Histórico de errores
}

impl LinearRegression {
    /// Crea nueva instancia del modelo
    /// 
    /// # Argumentos
    /// * `learning_rate` - Tasa de aprendizaje (típicamente 0.001 - 0.1)
    fn new(learning_rate: f64) -> Self {
        LinearRegression {
            w: 0.0,
            b: 0.0,
            learning_rate,
            epochs_trained: 0,
            mse_history: Vec::new(),
        }
    }

    /// Calcula gradientes para un mini-batch
    /// 
    /// # Argumentos
    /// * `x_batch` - Vector de características
    /// * `y_batch` - Vector de etiquetas
    /// 
    /// # Retorna
    /// Tupla (dw, db, mse)
    /// * `dw` - Gradiente respecto a w
    /// * `db` - Gradiente respecto a b
    /// * `mse` - Error cuadrático medio
    fn compute_gradient(&self, x_batch: &[f64], y_batch: &[f64]) -> (f64, f64, f64) {
        debug_assert_eq!(x_batch.len(), y_batch.len(), "Dimensiones deben coincidir");
        
        let m = x_batch.len() as f64;
        let mut error_sum = 0.0;
        let mut dw = 0.0;
        let mut mse_sum = 0.0;

        // Iterar sobre todas las muestras
        for (x, y) in x_batch.iter().zip(y_batch.iter()) {
            let y_pred = self.w * x + self.b;
            let error = y_pred - y;
            
            error_sum += error;
            dw += error * x;
            mse_sum += error * error;
        }

        // Fórmulas del descenso de gradiente
        // dw = (2/m) * sum(error * x)
        // db = (2/m) * sum(error)
        // mse = sum(error²) / m
        dw = (2.0 / m) * dw;
        let db = (2.0 / m) * error_sum;
        let mse = mse_sum / m;

        (dw, db, mse)
    }

    /// Actualiza los parámetros usando gradiente descendente
    /// 
    /// # Argumentos
    /// * `dw` - Gradiente respecto a w
    /// * `db` - Gradiente respecto a b
    fn update_weights(&mut self, dw: f64, db: f64) {
        self.w -= self.learning_rate * dw;
        self.b -= self.learning_rate * db;
    }

    /// Entrena el modelo una época
    /// 
    /// # Argumentos
    /// * `x` - Vector completo de características
    /// * `y` - Vector completo de etiquetas
    /// 
    /// # Retorna
    /// Error cuadrático medio (MSE) de esta época
    fn train_epoch(&mut self, x: &[f64], y: &[f64]) -> f64 {
        let (dw, db, mse) = self.compute_gradient(x, y);
        self.update_weights(dw, db);
        mse
    }

    /// Entrena el modelo por N épocas
    /// 
    /// # Argumentos
    /// * `x` - Vector de características
    /// * `y` - Vector de etiquetas
    /// * `epochs` - Número de épocas de entrenamiento
    fn train(&mut self, x: &[f64], y: &[f64], epochs: u32) {
        for epoch in 0..epochs {
            let mse = self.train_epoch(x, y);
            self.mse_history.push(mse);
            self.epochs_trained = epoch + 1;

            // Mostrar progreso cada 200 épocas
            if (epoch + 1) % 200 == 0 {
                println!("Epoch {}, MSE: {:.4}, w: {:.4}, b: {:.4}", 
                         epoch + 1, mse, self.w, self.b);
            }
        }
    }

    /// Realiza predicción para una entrada
    /// 
    /// # Argumentos
    /// * `x` - Valor de entrada
    /// 
    /// # Retorna
    /// Valor predicho: y = w*x + b
    fn predict(&self, x: f64) -> f64 {
        self.w * x + self.b
    }

    /// Obtiene el histórico de MSE
    fn get_mse_history(&self) -> &[f64] {
        &self.mse_history
    }

    /// Obtiene los parámetros del modelo
    fn get_params(&self) -> (f64, f64) {
        (self.w, self.b)
    }

    /// Establece los parámetros del modelo
    fn set_params(&mut self, w: f64, b: f64) {
        self.w = w;
        self.b = b;
    }
}

/// Benchmark 1: Dataset pequeño (5 muestras, 1000 épocas)
fn benchmark_small_dataset() {
    println!("\n{}", "═".repeat(70));
    println!("BENCHMARK 1: Dataset Pequeño (5 muestras, 1000 épocas)");
    println!("{}\n", "═".repeat(70));

    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0, 10.0];

    let learning_rate = 0.01;
    let epochs = 1000;

    let mut model = LinearRegression::new(learning_rate);
    
    let start = Instant::now();
    model.train(&x, &y, epochs);
    let duration = start.elapsed();

    let (w, b) = model.get_params();
    println!("\nModelo entrenado:");
    println!("  w ≈ {:.4}", w);
    println!("  b ≈ {:.4}", b);
    
    println!("\nTiempos:");
    println!("  Total: {:.6} segundos", duration.as_secs_f64());
    println!("  Por época: {:.6} ms", duration.as_secs_f64() * 1000.0 / epochs as f64);
    println!("  Throughput: {:.0} epochs/sec", epochs as f64 / duration.as_secs_f64());

    let x_nuevo = 7.0;
    let y_pred = model.predict(x_nuevo);
    println!("\nPredicción:");
    println!("  Para x = {}, y_pred ≈ {:.4}", x_nuevo, y_pred);

    let mse_history = model.get_mse_history();
    println!("\nErrores:");
    println!("  MSE inicial: {:.6}", mse_history.first().unwrap_or(&0.0));
    println!("  MSE final: {:.6}", mse_history.last().unwrap_or(&0.0));
    println!("  Reducción: {:.2}%", 
             (1.0 - mse_history.last().unwrap_or(&0.0) / mse_history.first().unwrap_or(&1.0)) * 100.0);
}

/// Benchmark 2: Dataset mediano (1000 muestras, 100 épocas)
fn benchmark_medium_dataset() {
    println!("\n{}", "═".repeat(70));
    println!("BENCHMARK 2: Dataset Mediano (1000 muestras, 100 épocas)");
    println!("{}\n", "═".repeat(70));

    let mut x = Vec::new();
    let mut y = Vec::new();
    
    // Generar dataset: y = 2*x + 0.5 + ruido
    for i in 1..=1000 {
        let xi = i as f64;
        x.push(xi);
        // Agregar pequeño ruido para realismo
        let noise = ((i * 7919) % 100) as f64 / 100.0 - 0.5;
        y.push(2.0 * xi + 0.5 + noise);
    }

    let learning_rate = 0.00001;
    let epochs = 100;

    let mut model = LinearRegression::new(learning_rate);
    
    let start = Instant::now();
    model.train(&x, &y, epochs);
    let duration = start.elapsed();

    let (w, b) = model.get_params();
    println!("\nDataset: {} muestras", x.len());
    println!("\nModelo entrenado:");
    println!("  w ≈ {:.4} (esperado: 2.0)", w);
    println!("  b ≈ {:.4} (esperado: 0.5)", b);
    
    println!("\nTiempos:");
    println!("  Total: {:.6} segundos", duration.as_secs_f64());
    println!("  Por época: {:.6} ms", duration.as_secs_f64() * 1000.0 / epochs as f64);
    println!("  Throughput: {:.0} samples/ms", 
             (x.len() as f64 * epochs as f64) / (duration.as_millis() as f64));

    let mse_history = model.get_mse_history();
    println!("\nErrores:");
    println!("  MSE inicial: {:.6}", mse_history.first().unwrap_or(&0.0));
    println!("  MSE final: {:.6}", mse_history.last().unwrap_or(&0.0));
}

/// Benchmark 3: Dataset grande (10000 muestras, 100 épocas)
fn benchmark_large_dataset() {
    println!("\n{}", "═".repeat(70));
    println!("BENCHMARK 3: Dataset Grande (10000 muestras, 100 épocas)");
    println!("{}\n", "═".repeat(70));

    let mut x = Vec::new();
    let mut y = Vec::new();
    
    for i in 1..=10000 {
        let xi = i as f64;
        x.push(xi);
        let noise = ((i * 7919) % 100) as f64 / 100.0 - 0.5;
        y.push(2.0 * xi + 0.5 + noise);
    }

    let learning_rate = 0.00001;
    let epochs = 100;

    let mut model = LinearRegression::new(learning_rate);
    
    let start = Instant::now();
    model.train(&x, &y, epochs);
    let duration = start.elapsed();

    let (w, b) = model.get_params();
    println!("\nDataset: {} muestras", x.len());
    println!("\nModelo entrenado:");
    println!("  w ≈ {:.4}", w);
    println!("  b ≈ {:.4}", b);
    
    println!("\nTiempos:");
    println!("  Total: {:.6} segundos", duration.as_secs_f64());
    println!("  Por época: {:.6} ms", duration.as_secs_f64() * 1000.0 / epochs as f64);
    println!("  Throughput: {:.0} samples/ms", 
             (x.len() as f64 * epochs as f64) / (duration.as_millis() as f64));

    let mse_history = model.get_mse_history();
    println!("\nErrores:");
    println!("  MSE inicial: {:.6}", mse_history.first().unwrap_or(&0.0));
    println!("  MSE final: {:.6}", mse_history.last().unwrap_or(&0.0));
}

/// Benchmark 4: Comparación de escalabilidad
fn benchmark_scalability() {
    println!("\n{}", "═".repeat(70));
    println!("BENCHMARK 4: Análisis de Escalabilidad");
    println!("{}\n", "═".repeat(70));

    let datasets = vec![
        (100, 100),
        (1_000, 100),
        (10_000, 100),
        (100_000, 10),
    ];

    println!("{:<12} {:<12} {:<12} {:<15}", "Tamaño", "Épocas", "Tiempo (ms)", "Throughput");
    println!("{}", "-".repeat(55));

    for (size, epochs) in datasets {
        let mut x = Vec::new();
        let mut y = Vec::new();
        
        for i in 1..=size {
            let xi = i as f64;
            x.push(xi);
            let noise = ((i * 7919) % 100) as f64 / 100.0 - 0.5;
            y.push(2.0 * xi + 0.5 + noise);
        }

        let mut model = LinearRegression::new(0.00001);
        
        let start = Instant::now();
        model.train(&x, &y, epochs);
        let duration = start.elapsed();

        let total_ops = (size as f64 * epochs as f64) / 1000.0;
        let time_ms = duration.as_millis() as f64;
        let throughput = total_ops / time_ms;

        println!("{:<12} {:<12} {:<12.4} {:<15.0}", 
                 size, epochs, time_ms, throughput);
    }
}

/// Benchmark 5: Micro-benchmarks de operaciones individuales
fn benchmark_microbenchmarks() {
    println!("\n{}", "═".repeat(70));
    println!("BENCHMARK 5: Micro-benchmarks");
    println!("{}\n", "═".repeat(70));

    // Test 1: Multiplicación escalar
    println!("Test 1: Multiplicación escalar (1M operaciones)");
    let start = Instant::now();
    let mut result = 1.5;
    for _ in 0..1_000_000 {
        result = result * 2.5;
    }
    let duration = start.elapsed();
    println!("  Tiempo: {:.6} µs", duration.as_secs_f64() * 1_000_000.0);
    println!("  Tiempo/op: {:.4} ns\n", duration.as_secs_f64() * 1_000_000_000.0 / 1_000_000.0);

    // Test 2: Dot product
    println!("Test 2: Dot product (1000 elementos, 100 veces)");
    let a: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let b: Vec<f64> = (0..1000).map(|i| (i as f64) * 2.0).collect();
    
    let start = Instant::now();
    for _ in 0..100 {
        let _: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    }
    let duration = start.elapsed();
    println!("  Tiempo: {:.6} ms", duration.as_secs_f64() * 1000.0);
    println!("  Tiempo/op: {:.4} µs\n", duration.as_secs_f64() * 1_000_000.0 / 100.0);

    // Test 3: Suma de array
    println!("Test 3: Suma de array (1000 elementos, 1000 veces)");
    let c: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    
    let start = Instant::now();
    for _ in 0..1000 {
        let _: f64 = c.iter().sum();
    }
    let duration = start.elapsed();
    println!("  Tiempo: {:.6} ms", duration.as_secs_f64() * 1000.0);
    println!("  Tiempo/op: {:.4} µs\n", duration.as_secs_f64() * 1_000_000.0 / 1000.0);
}

fn main() {
    println!("\n╔{}╗", "═".repeat(68));
    println!("║{}║", format!("{:^68}", "REGRESIÓN LINEAL - IMPLEMENTACIÓN EN RUST"));
    println!("║{}║", format!("{:^68}", "Benchmarks y Análisis de Desempeño"));
    println!("╚{}╝", "═".repeat(68));

    // Ejecutar benchmarks
    benchmark_small_dataset();
    benchmark_medium_dataset();
    benchmark_large_dataset();
    benchmark_scalability();
    benchmark_microbenchmarks();

    println!("\n{}", "═".repeat(70));
    println!("FIN DE BENCHMARKS");
    println!("{}", "═".repeat(70));
}
