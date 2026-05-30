Fisher-Hybrid
==============

Este paquete contiene una version de MedGS extendida con control de densidad guiado por incertidumbre de Fisher.

Repositorio:
https://github.com/LorenzoCostabile/MedGS-FisherHybrid

Documentacion principal:
README.md

Enlace directo a la seccion del experimento sparse:
https://github.com/LorenzoCostabile/MedGS-FisherHybrid#experimento-sparse


COMO EMPEZAR
============

1. Colocar este repositorio en una carpeta local.

2. Descargar el paquete de datos y extraerlo en la raiz del repositorio.
   Si los datos vienen ya preparados en formato MedGS, deben aparecer carpetas dentro de:

   data/

   Si los datos vienen con la estructura:

   <caso>/
     images/
     masks/

   se pueden adaptar con:

   bash scripts/prepare_case.sh <ruta_al_caso>

   Ejemplo:

   bash scripts/prepare_case.sh 014_P3_1_right


USO RECOMENDADO CON DOCKER
==========================

Opcion recomendada: usar la imagen ya publicada.

1. Descargar la imagen:

   docker pull storiano/fisher-hybrid:cuda12.4

2. Entrar en el repositorio:

   cd Fisher-Hybrid

3. Abrir una shell dentro del contenedor:

   docker compose run --rm medgs bash

Nota:
- La imagen aporta el entorno CUDA/PyTorch.
- El repositorio se monta por volumen en /workspace.
- Los cambios locales del codigo se ven directamente dentro del contenedor.


ENTRENAMIENTOS PRINCIPALES
==========================

Los tres entrenamientos principales descritos en la memoria son:

1. Baseline full-budget
2. Baseline same-budget
3. Fisher same-budget

Los comandos exactos estan documentados en:

README.md

Secciones recomendadas:
- "Entrenamientos de Referencia"
- "Protocolos de Split"


EXPERIMENTOS ADICIONALES
========================

Experimento sparse:

   bash scripts/run_experiment_sparse.sh

Experimento de bloque continuo oculto:

   bash scripts/run_experiment_gap.sh

Visualizacion de renders y videos:

   bash scripts/visualize_result.sh <model_path> <source_path> <iteration> <test_split> <holdout_stride> <holdout_offset> [second_holdout_offset]


PRUEBA RAPIDA
=============

Si se quiere comprobar que el paquete funciona sin lanzar un entrenamiento largo, se recomienda editar la variable ITERATIONS al ejecutar los scripts.

Ejemplo:

   CASE_NAME=014_P3_1_right ITERATIONS=500 bash scripts/run_experiment_sparse.sh


SI SE QUIERE RECONSTRUIR LA IMAGEN DOCKER
=========================================

Solo es necesario si no se usa la imagen publicada o si se quieren recompilar dependencias.

Antes de reconstruir, inicializar submodulos:

   git submodule update --init --recursive

Despues:

   docker compose build medgs


NOTA FINAL
==========

Para una explicacion completa del metodo, los splits, los experimentos y los scripts auxiliares, consultar:

- README.md
- documento/main.pdf
- documento/results.pdf
