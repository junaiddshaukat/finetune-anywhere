# Extractor de pendientes de notas de turno (Qwen3-1.7B ajustado)

Convierte notas de turno de mantenimiento en español en registros JSON estructurados.
Ajuste fino (LoRA) de Qwen3-1.7B, ejecutable de forma local en una laptop sin GPU dedicada.
Entrada al Qwen Local Hackathon 2026.

## What it does

Para un encargado de mantenimiento que releva turnos: convierte una nota de turno en español
en un JSON con los trabajos detectados. Un registro por par equipo-trabajo, con `equipo`,
`pendiente`, `estado` (abierto / cerrado / por confirmar), `cita` literal de la nota y `plazo`.
Las notas sin trabajos devuelven `{"registros": []}`. El punto difícil: negaciones
("no se cerró..."), correcciones sobre la marcha y notas que solo informan.

## Why we built it

En pases de turno reales los pendientes se pierden entre texto libre. Una extracción
estructurada permite tableros y alertas. El modelo base no sigue el contrato JSON ni respeta
negaciones (0/41 trabajos correctos), así que el ajuste fino era necesario.

## Dataset

88 notas de turno de mantenimiento en español, ficticias y originales de este proyecto
(paquete "Bloque 2": referencias redactadas antes de generar salidas, manifiesto SHA-256
fijado antes de la línea base): 40 entrenamiento, 12 validación, 36 evaluación held-out,
con particiones disjuntas y categorías equilibradas (claros, negación, varios equipos,
contradictorios, sin pendientes). NO provienen del starter: la receta `json-extractor` de
finetune-anywhere es una tarea distinta (60 casos de peticiones de compra). Del starter
(MIT, (c) 2026 M Junaid) se reutilizan el formato de receta, la configuración de
entrenamiento y el script de validación.

Los datos están en `data/` (train / valid / eval) con su manifiesto en `SHA256SUMS`,
bajo licencia CC-BY-4.0 (ver `LICENSE-DATOS.txt`).

## Training

unsloth 2026.11.12 sobre Qwen/Qwen3-1.7B (revisión a972f43), LoRA rango 16 alfa 32,
lr 2e-4, 3 épocas, semilla 42, max_seq_length 1024, batch 1 con acumulación 4.
Plataforma: Google Colab gratuito, Tesla T4. Entrenamiento medido: 86,5 segundos.
torch 2.14.1+cu130, transformers 4.57.6, trl 0.22.2, peft 0.21.2.

## Run it locally

- Descarga del modelo (GGUF Q4_K_M, 1,1 GB): en la sección Releases del repositorio
  https://github.com/Linder-debug/relevo-mantenimiento-qwen3, tag `v1.0`, asset
  `qwen3-1.7b-pendientes.gguf`. SHA-256 del archivo evaluado:
  90c7be886cdf270989b6dab9af559ef17e4ece3eceb533f73e48aa00745a13e0.
- Runtime: llama.cpp `llama-server -m qwen3-1.7b-pendientes.gguf -c 2048`, cliente
  `/v1/chat/completions` con `temperature 0`, `max_tokens 256` y el prompt de sistema
  incluido en `prompt-sistema.txt`. Probado en laptop Windows i9-13900H, 16 GB RAM, solo CPU.

## Before and after

Evaluación con `evaluar.py` (en el repositorio) sobre 36 notas held-out (41 trabajos esperados):

| Métrica | Base | Ajustado |
| --- | ---: | ---: |
| Trabajos identificados | 0 / 41 | 36 / 41 |
| F1 micro | 0.0000 | 0.8372 |
| Precisión / exhaustividad | 0 / 0 | 0.80 / 0.878 |
| Cierres falsos | 7 | 1 |
| Salidas ilegibles | 7 | 0 |

Ejemplo real (nota held-out fila 1, salidas verbatim en `model.yaml`): el base devuelve JSON
con tipos incorrectos (pendiente=true, plazo="null"); el ajustado devuelve el registro correcto.

Limitación conocida, declarada abiertamente: la ejecución local con el GGUF Q4_K_M en
llama.cpp b12728 rinde por debajo de la evaluación HF (F1 0,6234 vs 0,8372; 11 notas con
trabajo esperado devuelven lista vacía de inmediato). Al momento de esta publicación la causa
no está aislada (cuantización, conversión o runtime). Detalles medidos en
`nota-estado-local.md` del repositorio.

## What we learned

El ajuste con 40 ejemplos basta para el contrato JSON y las negaciones en la evaluación HF.
Pendiente: cerrar la brecha entre el modelo servido con transformers y su versión cuantizada
local, y fijar commits de las tuberías de conversión para hacerla reproducible.

## Team

- Linder Lopez — https://github.com/Linder-debug

Créditos: starter finetune-anywhere (MIT, (c) 2026 M Junaid) por el formato de receta, la
configuración de entrenamiento y el script de validación; unsloth, llama.cpp y Qwen3-1.7B
como herramientas y modelo base.
