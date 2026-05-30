#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Uso:"
  echo "  bash scripts/prepare_case.sh <ruta_al_caso> [nombre_salida]"
  echo
  echo "Ejemplo:"
  echo "  bash scripts/prepare_case.sh medgs_data/014_P3_1_right"
  echo
  echo "Por defecto crea:"
  echo "  data/real_<caso>_img"
  echo "  data/real_<caso>_seg"
  echo
  echo "Variables opcionales:"
  echo "  MODE=symlink   (por defecto)"
  echo "  MODE=copy      (copia los PNG en vez de enlazarlos)"
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

CASE_DIR="${1%/}"
CASE_NAME="${2:-$(basename "$CASE_DIR")}"
MODE="${MODE:-symlink}"

IMAGES_DIR="${CASE_DIR}/images"
MASKS_DIR="${CASE_DIR}/masks"

if [[ ! -d "${IMAGES_DIR}" ]]; then
  echo "Error: no existe ${IMAGES_DIR}"
  exit 1
fi

if [[ ! -d "${MASKS_DIR}" ]]; then
  echo "Error: no existe ${MASKS_DIR}"
  exit 1
fi

IMG_OUT="data/real_${CASE_NAME}_img"
SEG_OUT="data/real_${CASE_NAME}_seg"
IMG_ORIG="${IMG_OUT}/original"
SEG_ORIG="${SEG_OUT}/original"
IMG_MIRROR="${IMG_OUT}/mirror"
SEG_MIRROR="${SEG_OUT}/mirror"

prepare_link_target() {
  local src="$1"
  local dst="$2"
  local dst_dir
  local rel_src

  if [[ -L "${dst}" ]]; then
    rm -f "${dst}"
  elif [[ -d "${dst}" ]]; then
    if find "${dst}" -mindepth 1 -maxdepth 1 | read -r _; then
      echo "Error: ${dst} ya existe y no esta vacio. Borrarlo o usar otra salida."
      exit 1
    fi
    rmdir "${dst}"
  elif [[ -e "${dst}" ]]; then
    echo "Error: ${dst} ya existe y no es una carpeta vacia ni un symlink."
    exit 1
  fi

  dst_dir="$(dirname "${dst}")"
  rel_src="$(realpath --relative-to="${dst_dir}" "${src}")"
  ln -s "${rel_src}" "${dst}"
}

prepare_copy_target() {
  local src="$1"
  local dst="$2"

  mkdir -p "${dst}"
  cp -a "${src}/." "${dst}/"
}

mkdir -p "${IMG_OUT}" "${SEG_OUT}" "${IMG_MIRROR}" "${SEG_MIRROR}"

case "${MODE}" in
  symlink)
    prepare_link_target "${IMAGES_DIR}" "${IMG_ORIG}"
    prepare_link_target "${MASKS_DIR}" "${SEG_ORIG}"
    ;;
  copy)
    prepare_copy_target "${IMAGES_DIR}" "${IMG_ORIG}"
    prepare_copy_target "${MASKS_DIR}" "${SEG_ORIG}"
    ;;
  *)
    echo "Error: MODE debe ser 'symlink' o 'copy'"
    exit 1
    ;;
esac

echo "Caso preparado correctamente:"
echo "  Imagenes:      ${IMG_OUT}"
echo "  Segmentacion:  ${SEG_OUT}"
echo
echo "Carpetas creadas:"
echo "  ${IMG_ORIG}"
echo "  ${IMG_MIRROR}"
echo "  ${SEG_ORIG}"
echo "  ${SEG_MIRROR}"
echo
echo "Nota: las carpetas 'mirror/' se crean vacias. El loader medico de MedGS/Fisher-Hybrid"
echo "generara automaticamente las imagenes espejadas cuando las necesite."
