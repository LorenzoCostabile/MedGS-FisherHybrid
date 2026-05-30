# Fisher-Hybrid

`Fisher-Hybrid` es una extensión de `MedGS` para control de densidad guiado por incertidumbre de Fisher en reconstrucción médica con Gaussian Splatting.

La idea principal es mantener el pipeline de entrenamiento y renderizado de MedGS, pero modificar la fase de densificación y poda:

- se definen particiones de frames para `train`, `fisher-val`, `test` y `gap`
- se calcula un proxy de incertidumbre tipo Fisher a partir de pérdidas de imagen y segmentación
- esa señal se usa para filtrar candidatos de densificación
- y para introducir una poda conservadora guiada por incertidumbre

Este repositorio corresponde al código de entrega para reproducir los experimentos principales del trabajo.

## Qué Implementa Este Repositorio

Los archivos del método que realmente contienen la modificación sobre MedGS son:

- `train.py`
- `utils/fisher_utils.py`
- `scene/gaussian_model.py`
- `arguments/__init__.py`
- `scene/__init__.py`
- `models/scenes/dataset_readers.py`
- `gaussian_renderer/__init__.py`

Los modos de densidad relevantes son:

- `--density_mode heuristic`
- `--density_mode fisher_hybrid`

Los argumentos principales de partición son:

- `--holdout_stride`
- `--holdout_offset`
- `--second_holdout_offset`
- `--test_split`
- `--train_pool_stride`
- `--gap_start_frac`
- `--gap_end_frac`

## Flujo Recomendado

La forma recomendada de usar el repositorio es mediante `Docker Compose`.

El `docker-compose.yml` monta el repositorio dentro del contenedor:

```yaml
volumes:
  - .:/workspace
```

Eso significa:

- la imagen aporta el entorno CUDA/PyTorch y las extensiones compiladas
- el código se toma del árbol actual del repositorio mediante `/workspace`
- los cambios hechos en el host se ven inmediatamente dentro del contenedor
- solo hace falta reconstruir la imagen si cambias dependencias, submódulos compilados o el `Dockerfile`

## Requisitos

- GPU NVIDIA
- driver NVIDIA + NVIDIA Container Toolkit
- Docker + Docker Compose

## Clonado y Submódulos

Si vas a reconstruir la imagen localmente, clona el repositorio con submódulos:

```bash
git clone --recurse-submodules <repo_url>
cd Fisher-Hybrid
```

Si ya lo habías clonado sin submódulos:

```bash
git submodule update --init --recursive
```

La reconstrucción depende de:

- `submodules/diff-gaussian-rasterization`
- `submodules/simple-knn`

## Uso con Docker

### Opción 1: usar una imagen preconstruida

Si ya existe una imagen publicada en Docker Hub:

```bash
docker pull storiano/fisher-hybrid:cuda12.4
docker compose run --rm --no-build medgs bash
```

En este caso:

- la imagen aporta el entorno
- el repositorio montado por volumen aporta el código

### Opción 2: construir localmente

```bash
docker compose build medgs
docker compose run --rm medgs bash
```

## Paquete de Datos

La entrega asume que los datos se proporcionan en un ZIP con los frames ya preparados en formato MedGS.

Ejemplo:

```bash
unzip fisher_hybrid_data.zip
```

Después de extraerlo, deberían existir raíces como estas:

```text
data/
├── real_014_P3_1_right_img/
│   ├── original/
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   └── ...
│   └── mirror/
└── real_014_P3_1_right_seg/
    ├── original/
    │   ├── 0000.png
    │   ├── 0001.png
    │   └── ...
    └── mirror/
```

En entrenamiento conjunto:

- `-s` apunta a la raíz de imagen
- `--seg_source_path` apunta a la raíz de segmentación

Ambas raíces deben tener:

- el mismo número de frames
- el mismo indexado
- carpetas `original/` y `mirror/` compatibles

## Protocolos de Split

### 1. Same-budget

Este protocolo se usa para comparar:

- `baseline same-budget`
- `fisher same-budget`

Interpretación:

- `train`: frames usados por la pérdida principal de reconstrucción
- `fisher-val`: frames reservados para calcular el proxy Fisher y guiar densificación/poda
- `untouched test`: frames reservados exclusivamente para evaluación final

Configuración de referencia:

```text
--holdout_stride 8
--holdout_offset 0
--second_holdout_offset 4
--test_split primary
```

Eso significa:

- los frames con `idx % 8 == 0` pasan a `fisher-val`
- los frames con `idx % 8 == 4` pasan al test final intocable
- el resto se usa en la pérdida principal

### 2. Full-budget

Este protocolo se usa para el baseline heurístico más fuerte.

Interpretación:

- todos los frames no test se usan para entrenar
- no existe un subconjunto separado `fisher-val`

Configuración de referencia:

```text
--holdout_stride 8
--holdout_offset 4
--second_holdout_offset -1
--test_split primary
```

Eso significa:

- los frames con `idx % 8 == 4` quedan reservados para test final
- todos los demás se usan para entrenamiento

## Entrenamientos de Referencia

Estos son los tres entrenamientos principales usados en el trabajo.

### 1. Baseline full-budget

Es el baseline heurístico más fuerte.

```bash
docker compose run --rm --no-build medgs \
  python -u train.py \
  -s data/real_014_P3_1_right_img \
  -m output/expH_014_P3_1_right_baseline_full \
  --pipeline joint \
  --seg_source_path data/real_014_P3_1_right_seg \
  --iterations 30000 \
  --holdout_stride 8 \
  --holdout_offset 4 \
  --second_holdout_offset -1 \
  --test_split primary \
  --density_mode heuristic \
  --save_iterations 5000 10000 15000 20000 25000 30000 \
  --checkpoint_iterations 5000 10000 15000 20000 25000 30000 \
  --test_iterations 5000 10000 15000 20000 25000 30000
```

### 2. Baseline same-budget

Usa el mismo subconjunto principal de reconstrucción que Fisher, pero no usa `fisher-val`.

```bash
docker compose run --rm --no-build medgs \
  python -u train.py \
  -s data/real_014_P3_1_right_img \
  -m output/expA_014_P3_1_right_baseline \
  --pipeline joint \
  --seg_source_path data/real_014_P3_1_right_seg \
  --iterations 30000 \
  --holdout_stride 8 \
  --holdout_offset 0 \
  --second_holdout_offset 4 \
  --test_split primary \
  --density_mode heuristic \
  --save_iterations 5000 10000 15000 20000 25000 30000 \
  --checkpoint_iterations 5000 10000 15000 20000 25000 30000 \
  --test_iterations 5000 10000 15000 20000 25000 30000
```

### 3. Fisher same-budget

Es la variante propuesta con control de densidad guiado por Fisher.

```bash
docker compose run --rm --no-build medgs \
  python -u train.py \
  -s data/real_014_P3_1_right_img \
  -m output/expA_014_P3_1_right_fisher \
  --pipeline joint \
  --seg_source_path data/real_014_P3_1_right_seg \
  --iterations 30000 \
  --holdout_stride 8 \
  --holdout_offset 0 \
  --second_holdout_offset 4 \
  --test_split primary \
  --density_mode fisher_hybrid \
  --fisher_views_per_update 4 \
  --fisher_ema_decay 0.8 \
  --fisher_weight_xyz 0.5 \
  --fisher_weight_deform 0.5 \
  --fisher_keep_quantile 0.5 \
  --fisher_prune_quantile 0.1 \
  --fisher_prune_opacity 0.05 \
  --fisher_prune_patience 3 \
  --save_iterations 5000 10000 15000 20000 25000 30000 \
  --checkpoint_iterations 5000 10000 15000 20000 25000 30000 \
  --test_iterations 5000 10000 15000 20000 25000 30000
```

## Experimento Sparse

El experimento `sparse` reduce el subconjunto principal de entrenamiento manteniendo fijo el mismo `fisher-val` y el mismo `test`.

Regímenes:

- `dense`: `train_pool_stride=1`
- `x2`: `train_pool_stride=2`
- `x4`: `train_pool_stride=4`

He dejado un script para lanzar las variantes principales:

- `scripts/run_experiment_sparse.sh`

Este script se ejecuta **desde el host** y lanza internamente los comandos `docker compose run --rm --no-build medgs ...`.

Uso recomendado para el profesor:

```bash
docker compose pull
bash scripts/run_experiment_sparse.sh
```

Opcionalmente puedes sobreescribir variables:

```bash
docker compose pull
CASE_NAME=014_P3_1_right ITERATIONS=30000 bash scripts/run_experiment_sparse.sh
```

## Experimento de Bloque Continuo Oculto

El experimento `contiguous gap` elimina un bloque continuo del barrido y evalúa la reconstrucción dentro de esa región no observada.

Parámetros de referencia:

- `gap_start_frac=0.4`
- `gap_end_frac=0.6`

He dejado un script para lanzar este experimento:

- `scripts/run_experiment_gap.sh`

Este script se ejecuta **desde el host** y lanza internamente los comandos `docker compose run --rm --no-build medgs ...`.

Uso recomendado para el profesor:

```bash
docker compose pull
bash scripts/run_experiment_gap.sh
```

También puedes ajustar el tamaño del hueco:

```bash
docker compose pull
CASE_NAME=014_P3_1_right GAP_START_FRAC=0.4 GAP_END_FRAC=0.6 bash scripts/run_experiment_gap.sh
```

## Visualización de Resultados

He añadido un script para renderizar un modelo entrenado y generar vídeos de imagen y máscara:

- `scripts/visualize_result.sh`

Este script se ejecuta **desde el host** y llama internamente a `docker compose run --rm --no-build medgs ...`.

Ejemplo sobre el `baseline full-budget`:

```bash
docker compose pull
bash scripts/visualize_result.sh \
  output/expH_014_P3_1_right_baseline_full \
  data/real_014_P3_1_right_img \
  30000 \
  primary \
  8 4 -1
```

Ejemplo sobre `fisher same-budget` evaluado en el test intocable:

```bash
docker compose pull
bash scripts/visualize_result.sh \
  output/expA_014_P3_1_right_fisher \
  data/real_014_P3_1_right_img \
  30000 \
  secondary \
  8 0 4
```

Ejemplo sobre el experimento de `gap`:

```bash
docker compose pull
GAP_START_FRAC=0.4 GAP_END_FRAC=0.6 \
bash scripts/visualize_result.sh \
  output/expC_014_P3_1_right_gap20_fisher \
  data/real_014_P3_1_right_img \
  30000 \
  gap \
  8 0 -1
```

El script genera:

- `render_img/`
- `render_mask/`
- `render_img.mp4`
- `render_mask.mp4`

## Entrenamiento General

El script principal sigue soportando los pipelines de MedGS:

- `img`
- `seg`
- `joint`

Entrenamiento conjunto típico:

```bash
python -u train.py \
  -s <img_dataset_dir> \
  -m <output_dir> \
  --pipeline joint \
  --seg_source_path <seg_dataset_dir>
```

## Renderizado

Renderizado manual:

```bash
python render.py --model_path <model_dir> --pipeline both
```

Opciones útiles:

- `--iteration <int>`
- `--pipeline {img,seg,both}`
- `--interp <int>`

## Notas de Reproducibilidad

- Si usas imagen preconstruida + volumen montado, la imagen aporta el entorno y `/workspace` aporta el código.
- Si reconstruyes localmente, asegúrate de inicializar submódulos antes.
- El protocolo `same-budget` no debe interpretarse como un baseline de información completa: la variante Fisher aprovecha `fisher-val` para guiar el control de densidad, mientras que el baseline heurístico ignora ese subconjunto.

## Licencia y Base del Proyecto

Este repositorio está basado en MedGS y conserva la estructura y licencia del código base donde corresponde. `Fisher-Hybrid` añade sobre esa base el control de densidad guiado por Fisher.
