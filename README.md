# SDTeoRAY_Practica2

## Cómo correr

1. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate
```

En Windows:

```bash
venv\Scripts\activate
```

2. Instalar dependencias

```bash
pip install ray[tune] torch torchvision matplotlib pandas
```

3. Probar que los workers funcionan

```bash
python tests/test_workers.py
```

4. Ejecutar entrenamiento distribuido (2 workers por defecto)

```bash
python src/train_distributed.py
```

5. Para cambiar número de workers, editar `NUM_WORKERS` en `src/config.py`
