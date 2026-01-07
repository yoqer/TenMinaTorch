# Manual de Uso - TenMinaTorc

**Framework Completo para Deep Learning, Reinforcement Learning y Modelos Relacionales**

---

## Tabla de Contenidos

1. [Introducción](#introducción)
2. [Instalación](#instalación)
3. [Arquitectura del Framework](#arquitectura-del-framework)
4. [Guía de Inicio Rápido](#guía-de-inicio-rápido)
5. [Transformer RLM](#transformer-rlm)
6. [Reinforcement Learning](#reinforcement-learning)
7. [Relational Networks](#relational-networks)
8. [vLLM (Inferencia Rápida)](#vllm-inferencia-rápida)
9. [Control de Entrenamiento](#control-de-entrenamiento)
10. [Entrenamiento End-to-End](#entrenamiento-end-to-end)
11. [Exportación de Modelos](#exportación-de-modelos)
12. [Troubleshooting](#troubleshooting)

---

## Introducción

TenMinaTorc Framework es un framework completo construido sobre **TenMINATorch Lite** que proporciona:

- **Transformer RLM**: Arquitectura Transformer optimizada para Reinforcement Learning con hasta 8192 neuronas
- **Reinforcement Learning**: Algoritmos Q-Learning (DQN) y PPO
- **Relational Networks**: Redes para aprendizaje relacional
- **vLLM**: Inferencia rápida con KV-Cache y Continuous Batching
- **Control de Entrenamiento**: Early Stopping, checkpoints, límite de iteraciones

---

## Instalación

### 1. Instalar TenMINATorch Lite

```bash
# Desde el ZIP
unzip TenMINATorch.zip
cd TenMINATorch
pip install -e .

# O añadir al path de Python
import sys
sys.path.insert(0, '/ruta/a/TenMINATorch')
```

### 2. Instalar el Framework

```bash
# Extraer el framework
unzip TenMinaTorc.zip
cd TenMinaTorc

# Añadir al path
import sys
sys.path.insert(0, '/ruta/a/TenMinaTorc')
```

### 3. Dependencias Opcionales

```bash
# Para aceleración
pip install numba jax jaxlib

# Para integración con otros frameworks
pip install tensorflow torch
```

---

## Arquitectura del Framework

```
minitorch_framework/
├── models/
│   └── transformer_rlm.py      # Transformer RLM
├── rl/
│   └── reinforcement_learning.py  # Q-Learning, PPO
├── vllm/
│   └── relational_vllm.py      # Relational Networks, vLLM
├── examples/
│   └── train_complete_model.py # Script de entrenamiento
└── docs/
    └── MANUAL_DE_USO.md        # Este manual
```

---

## Guía de Inicio Rápido

### Ejemplo Mínimo

```python
import sys
sys.path.insert(0, '/home/ubuntu')
sys.path.insert(0, '/home/ubuntu/TenMinaTorc')

from minitorch_lite import Tensor, Adam
from models.transformer_rlm import TransformerRLM

# Crear modelo
model = TransformerRLM(
    vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048
)

# Crear optimizador
optimizer = Adam(model.parameters(), lr=0.0001)

# Forward pass
inputs = np.random.randint(0, 10000, (2, 128))  # (batch_size, seq_len)
logits = model.forward(inputs)

print(f"Salida: {logits.shape}")  # (2, 128, 10000)
```

---

## Transformer RLM

### Características

- **Multi-Head Self-Attention**: Hasta 64 cabezas
- **Feed-Forward Networks**: Hasta 8192 neuronas ocultas
- **Positional Encoding**: Secuencias de hasta 8192 tokens
- **Layer Normalization**: Estabilización del entrenamiento

### Uso Básico

```python
from models.transformer_rlm import TransformerRLM

# Configuración de gran escala
model = TransformerRLM(
    vocab_size=50000,      # Vocabulario grande
    d_model=2048,          # Dimensión del modelo
    num_heads=32,          # 32 cabezas de atención
    num_layers=24,         # 24 capas Transformer
    d_ff=8192,             # 8192 neuronas en FFN
    max_len=8192,          # Secuencias largas
    dropout=0.1
)

print(f"Parámetros totales: ~{model._count_parameters():,}")
```

### Generación de Texto

```python
# Generar texto autoregressivamente
start_tokens = np.array([[1, 2, 3]])  # Tokens iniciales
generated = model.generate(
    start_tokens,
    max_len=100,
    temperature=0.8
)

print(f"Tokens generados: {generated}")
```

### Arquitectura Detallada

| Componente | Descripción |
|------------|-------------|
| `MultiHeadAttention` | Atención multi-cabeza con escalado |
| `FeedForward` | Red feed-forward con ReLU |
| `LayerNorm` | Normalización por capa |
| `TransformerBlock` | Bloque completo con residual connections |
| `PositionalEncoding` | Codificación de posición sinusoidal |

---

## Reinforcement Learning

### Q-Learning (DQN)

```python
from rl.reinforcement_learning import DQNAgent

# Crear agente
agent = DQNAgent(
    state_dim=84,          # Dimensión del estado
    action_dim=4,          # Número de acciones
    learning_rate=0.001,
    gamma=0.99,
    buffer_capacity=100000,
    batch_size=64
)

# Bucle de entrenamiento
for episode in range(1000):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Seleccionar acción
        action = agent.select_action(state)
        
        # Ejecutar acción
        next_state, reward, done, _ = env.step(action)
        
        # Almacenar transición
        agent.store_transition(state, action, reward, next_state, done)
        
        # Entrenar
        loss = agent.train_step()
        
        state = next_state
        total_reward += reward
    
    print(f"Episodio {episode}: Recompensa = {total_reward}")
```

### PPO (Proximal Policy Optimization)

```python
from rl.reinforcement_learning import PPOAgent

# Crear agente PPO
agent = PPOAgent(
    state_dim=84,
    action_dim=4,
    learning_rate=0.0003,
    gamma=0.99,
    epsilon_clip=0.2,
    epochs=10
)

# Bucle de entrenamiento
for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        # Seleccionar acción
        action = agent.select_action(state)
        
        # Ejecutar acción
        next_state, reward, done, _ = env.step(action)
        
        # Almacenar recompensa
        agent.store_reward(reward, done)
        
        state = next_state
    
    # Entrenar al final del episodio
    loss = agent.train_step()
    print(f"Episodio {episode}: Pérdida = {loss:.4f}")

# Guardar modelo
agent.save('ppo_model.pkl')
```

---

## Relational Networks

### Uso Básico

```python
from vllm.relational_vllm import RelationalNetwork

# Crear red relacional
relnet = RelationalNetwork(
    object_dim=256,       # Dimensión de cada objeto
    relation_dim=256,     # Dimensión de la relación
    output_dim=10         # Dimensión de salida
)

# Forward pass
objects = Tensor(np.random.randn(2, 5, 256), requires_grad=True)  # (batch, num_objects, dim)
output = relnet(objects)

print(f"Salida: {output.shape}")  # (2, 10)
```

### Aplicaciones

- **Razonamiento Visual**: Relaciones entre objetos en imágenes
- **Razonamiento Lógico**: Relaciones entre entidades
- **Grafos**: Relaciones entre nodos

---

## vLLM (Inferencia Rápida)

### Características

- **KV-Cache**: Almacena claves y valores de atención
- **Continuous Batching**: Procesa múltiples requests de diferentes longitudes
- **Top-K y Top-P Sampling**: Control de generación

### Uso Básico

```python
from vllm.relational_vllm import vLLM

# Crear vLLM
vllm_engine = vLLM(
    model=transformer_model,
    max_batch_size=8,
    max_seq_len=2048
)

# Generar texto
prompt_tokens = np.array([[1, 2, 3, 4, 5]])
generated = vllm_engine.generate(
    prompt_tokens,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9
)

print(f"Tokens generados: {generated}")
```

### Generación en Batch

```python
# Múltiples prompts
prompts = [
    np.array([1, 2, 3]),
    np.array([4, 5, 6, 7]),
    np.array([8, 9])
]

# Generar en batch
results = vllm_engine.batch_generate(
    prompts,
    max_new_tokens=50,
    temperature=0.7
)

for i, result in enumerate(results):
    print(f"Prompt {i}: {result}")
```

---

## Control de Entrenamiento

### Opciones de Control

| Opción | Descripción | Valor por Defecto |
|--------|-------------|-------------------|
| `max_iterations` | Máximo de iteraciones | 69 |
| `early_stop_patience` | [NEW] Iteraciones sin mejora antes de parar | 12 |
| `max_tokens` | Límite de tokens | None |
| `token_limit_iterations` | Máximo de iteraciones con límite de tokens | 12 |

### Uso Básico

```python
from minitorch_lite import create_training_controller

# Crear controlador
controller = create_training_controller(
    max_iterations=69,
    early_stop=True,
    early_stop_patience=12,
    max_tokens=10,
    checkpoint_dir='./checkpoints'
)

# Bucle de entrenamiento
while controller.should_continue():
    loss = train_step(data)
    controller.update(loss, tokens_processed=batch_size)

# Guardar checkpoint
controller.save_checkpoint(model.state_dict())

# Resumen
print(controller.get_summary())
```

### Checkpoints

```python
# Guardar checkpoint
checkpoint_path = controller.save_checkpoint(
    model_state=model.state_dict(),
    optimizer_state=optimizer.state_dict()
)

# Cargar checkpoint
checkpoint = controller.load_checkpoint(checkpoint_path)
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optimizer_state'])
```

---

## Entrenamiento End-to-End

### Script Completo

```bash
# Entrenar modelo completo
python examples/train_complete_model.py --config config.json --epochs 10 --generate
```

### Configuración (config.json)

```json
{
  "model": {
    "vocab_size": 50000,
    "d_model": 512,
    "num_heads": 8,
    "num_layers": 6,
    "d_ff": 2048,
    "max_len": 2048
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 0.0001,
    "max_iterations": 69,
    "early_stop_patience": 12,
    "checkpoint_dir": "./checkpoints"
  }
}
```

### Pipeline Personalizado

```python
from examples.train_complete_model import CompleteTrainingPipeline

# Crear pipeline
pipeline = CompleteTrainingPipeline(config)

# Entrenar
pipeline.train(num_epochs=10)

# Evaluar
test_loss = pipeline.evaluate(test_inputs, test_targets)

# Generar texto
generated_text = pipeline.generate_text("Hello world", max_length=100)

# Exportar modelo
pipeline.export_model(format='numpy')
```

---

## Exportación de Modelos

### Formatos Soportados

- **NumPy** (.npz): Formato por defecto
- **Keras** (.npz): Compatible con TensorFlow/Keras
- **PyTorch** (.pt): Compatible con PyTorch
- **Pickle** (.pkl): Formato nativo de Python

### Exportar

```python
from TenMINATorch import ModelExporter

exporter = ModelExporter()

# Exportar a NumPy
exporter.export(model.state_dict(), 'model.npz', format='numpy')

# Exportar a PyTorch
exporter.export(model.state_dict(), 'model.pt', format='pytorch')

# Exportar a Keras
exporter.export(model.state_dict(), 'model_keras.npz', format='keras')
```

### Importar

```python
# Importar desde PyTorch
weights = exporter.import_weights('model.pt', format='pytorch')
model.load_state_dict(weights)

# Importar desde Keras
weights = exporter.import_weights('model_keras.npz', format='keras')
model.load_state_dict(weights)
```

---

## Troubleshooting

### Problema: RecursionError en NumPy

**Solución**: Usar backend alternativo

```python
from TenMINATorch import set_backend

# Cambiar a Numba
set_backend('numba')

# O a JAX
set_backend('jax')
```

### Problema: Out of Memory

**Solución**: Reducir tamaño del modelo o batch

```python
# Reducir batch_size
config['training']['batch_size'] = 16

# Reducir dimensiones del modelo
config['model']['d_model'] = 256
config['model']['d_ff'] = 1024
```

### Problema: Entrenamiento muy lento

**Solución**: Usar vLLM y backends acelerados

```python
# Instalar Numba
pip install numba

# Cambiar backend
set_backend('numba')

# Usar vLLM para inferencia
vllm_engine = vLLM(model, max_batch_size=8)
```

### Problema: Early Stopping demasiado temprano

**Solución**: Aumentar paciencia

```python
controller = create_training_controller(
    early_stop_patience=24,  # Aumentar de 12 a 24
    max_iterations=100
)
```

---

## Ejemplos Adicionales

### Ejemplo 1: Entrenar en Datos Personalizados

```python
# Cargar datos
train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')

# Crear pipeline
pipeline = CompleteTrainingPipeline(config)

# Entrenar
for epoch in range(10):
    for batch_inputs, batch_labels in dataloader:
        loss = pipeline.train_step(batch_inputs, batch_labels)
        pipeline.controller.update(loss)
```

### Ejemplo 2: Fine-tuning desde Checkpoint

```python
# Cargar checkpoint
pipeline.load_checkpoint('checkpoints/model_final.pkl')

# Continuar entrenamiento
pipeline.train(num_epochs=5)
```

### Ejemplo 3: Generación Interactiva

```python
while True:
    prompt = input("Prompt: ")
    if prompt == 'exit':
        break
    
    generated = pipeline.generate_text(prompt, max_length=100)
    print(f"Generado: {generated}")
```

---

## Conclusión

Este framework proporciona todas las herramientas necesarias para entrenar modelos de deep learning de gran escala con control avanzado de entrenamiento y opciones de inferencia rápida.

Para más información, consulta:
- README.md de MiniTorch Lite
- Código fuente en `/minitorch_framework/`
- Ejemplos en `/minitorch_framework/examples/`

---

**Autor**: MiniTorch Framework Team  
**Versión**: 1.0.0  
**Fecha**: 2026
