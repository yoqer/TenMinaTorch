# TenMinaTorch Lite

**Lightweight Deep Learning Library with Multi-Backend Support**

Una librería de deep learning ligera y flexible con soporte para múltiples backends de cálculo numérico y control avanzado de entrenamiento.

## Características Principales

### 🚀 Múltiples Backends
- **NumPy** (por defecto): Compatibilidad universal
- **Numba**: Aceleración JIT hasta 1000x en operaciones intensivas
- **JAX**: Autograd avanzado y compilación XLA
- **CuPy**: Aceleración GPU con CUDA

### 🎯 Control de Entrenamiento
- **[NEW] Early Stop**: Salto automático tras 12 iteraciones sin mejora
- **Límite de Tokens**: Corte automático a N tokens con máximo 12 iteraciones
- **Por defecto**: Máximo 69 repeticiones
- **Checkpoints**: Guardado y reanudación de entrenamiento

### 🔗 Integración con Frameworks Externos
- **Keras/TensorFlow**: Exportar/importar pesos
- **PyTorch**: Conversión bidireccional de tensores
- **JAX**: Compilación JIT y autograd avanzado

## Instalación

### Instalación básica (solo NumPy)
```bash
pip install TenMINATorch
```

### Instalación con backends adicionales
```bash
# Con Numba (aceleración JIT)
pip install TenMINATorch[numba]

# Con JAX (autograd avanzado)
pip install TenMINATorch[jax]

# Con todas las dependencias opcionales
pip install TenMINATorch[all]
```

### Instalación desde código fuente
```bash
git clone https://github.com/TenMinaTorch/TenMINATorch.git
cd TenMINATorch
pip install -e .
```

### Uso sin PyPI (instalación local)
```python
import sys
sys.path.insert(0, '/ruta/a/TenMINATorch')
import TenMINATorch as mtl
```

## Uso Básico

### Crear Tensores y Operaciones
```python
import TenMINATorch as mtl

# Crear tensores
x = mtl.Tensor([[1, 2], [3, 4]], requires_grad=True)
y = mtl.Tensor([[5, 6], [7, 8]], requires_grad=True)

# Operaciones
z = x @ y  # Multiplicación de matrices
loss = z.sum()
loss.backward()

print(x.grad)  # Gradientes
```

### Cambiar Backend
```python
import TenMINATorch as mtl

# Ver backends disponibles
mtl.print_backend_info()

# Cambiar a Numba (si está instalado)
mtl.set_backend('numba')
```

### Control de Entrenamiento
```python
import TenMINATorch as mtl

# Crear controlador con Early Stopping [NEW]
controller = mtl.create_training_controller(
    max_iterations=69,           # Máximo 69 repeticiones
    early_stop=True,             # [NEW] Activar Early Stop
    early_stop_patience=12,      # [NEW] Salto tras 12 iteraciones sin mejora
    max_tokens=10,               # Límite de 10 tokens
    token_limit_iterations=12    # Máximo 12 iteraciones con límite
)

# Bucle de entrenamiento
while controller.should_continue():
    loss = train_step(data)
    controller.update(loss, tokens_processed=batch_tokens)

# Guardar checkpoint
controller.save_checkpoint(model.state_dict())
```

### Red Neuronal Simple
```python
import TenMINATorch as mtl

# Definir modelo
model = mtl.Sequential(
    mtl.Linear(784, 128),
    mtl.ReLU(),
    mtl.Dropout(0.5),
    mtl.Linear(128, 10),
    mtl.Softmax()
)

# Optimizador
optimizer = mtl.Adam(model.parameters(), lr=0.001)

# Función de pérdida
criterion = mtl.CrossEntropyLoss()

# Entrenamiento
for epoch in range(10):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

### Exportar/Importar Pesos
```python
import TenMINATorch as mtl

exporter = mtl.ModelExporter()

# Exportar a formato Keras
exporter.export(model.state_dict(), 'model_weights.npz', format='keras')

# Importar desde PyTorch
weights = exporter.import_weights('pytorch_model.pt', format='pytorch')
model.load_state_dict(weights)
```

## Opciones de Control de Entrenamiento

| Opción | Descripción | Valor por Defecto |
|--------|-------------|-------------------|
| `max_iterations` | Máximo de iteraciones totales | 69 |
| `early_stop_patience` | [NEW] Iteraciones sin mejora antes de parar | 12 |
| `max_tokens` | Límite de tokens por entrenamiento | None (sin límite) |
| `token_limit_iterations` | Máximo de iteraciones con límite de tokens | 12 |
| `save_checkpoint_on_stop` | Guardar checkpoint al detenerse | True |

## Backends Disponibles

| Backend | Descripción | Instalación |
|---------|-------------|-------------|
| `numpy` | Backend por defecto, compatible universal | Incluido |
| `numba` | Aceleración JIT para bucles intensivos | `pip install numba` |
| `jax` | Autograd avanzado y compilación XLA | `pip install jax jaxlib` |
| `cupy` | Aceleración GPU con CUDA | `pip install cupy` |

## Integraciones Externas

### Keras/TensorFlow
```python
from TenMINATorch import KerasIntegration

# Exportar pesos para Keras
KerasIntegration.save_for_keras(weights, 'model.npz')

# Crear modelo Keras desde configuración
keras_model = KerasIntegration.create_keras_model_from_config(config)
```

### PyTorch
```python
from TenMINATorch import PyTorchIntegration

# Convertir tensor
torch_tensor = PyTorchIntegration.to_pytorch_tensor(numpy_array)

# Exportar pesos
PyTorchIntegration.save_for_pytorch(weights, 'model.pt')
```

### JAX
```python
from TenMINATorch import JAXIntegration

# Convertir a array JAX
jax_array = JAXIntegration.to_jax_array(numpy_array)

# Compilar función con JIT
fast_func = JAXIntegration.jit_compile(my_function)
```

## Estructura del Proyecto

```
TenMINATorch/
├── __init__.py       # Exportaciones principales
├── tensor.py         # Clase Tensor
├── autograd.py       # Motor de diferenciación automática
├── backends.py       # Soporte multi-backend
├── nn.py             # Capas de red neuronal
├── optim.py          # Optimizadores
├── training.py       # Control de entrenamiento
├── integrations.py   # Integraciones externas
├── setup.py          # Configuración PyPI
└── README.md         # Documentación
```

## Puede actualizarla y perfeccionarla o especializarla.

Los cambios seran valorados, es una librería de usos generales, Open Source sin restricciones, si desea compartir las mejoras o especializaciones, pueden ser referenciadas como ramas, como las de AlIAmAlIA; o ser añadidas colaborando mediante un Pull Request.



## Para Subida a PyPI

1. **Crear cuenta en PyPI**: https://pypi.org/account/register/

2. **Instalar herramientas**:
   ```bash
   pip install twine build
   ```

3. **Construir distribución**:
   ```bash
   python -m build
   ```

4. **Subir a PyPI**:
   ```bash
   twine upload dist/*
   ```

## Licencia

MIT License

## Contribuir

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request en GitHub.
