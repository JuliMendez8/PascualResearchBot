<div align="center">
  <img src="https://pascualbravo.edu.co/wp-content/uploads/2019/12/cropped-Institucion_Pascual_Bravo_Logo.png" width="350px">
</div>

# Pascual Research: Sistema Inteligente de Generación y Análisis de Información Académica Basado en Modelos de Lenguaje

Este proyecto pretende construir un ChatBot con la LLM DeepSeek-R1-Distill-Qwen-R1 que integre y procese información proveniente de trabajos y fuentes académicas (tales como el repositorio institucional) para proporcionar apoyo y orientación a proyectos emergentes dentro de la Institución Universitaria Pascual Bravo.

# Objetivo

Este proyecto tiene como propósito:

- Optimizar el acceso a la información científica y académica en la IUPB.  
- Reentrenar un LLM con datos institucionales para mejorar la precisión de las respuestas.  
- Desarrollar un bot accesible desde Telegram para consulta directa.  
- Implementar un pipeline reproducible que incluye:
  - Generación de datos sintéticos desde PDFs institucionales  
  - Evaluación automática de la calidad de las respuestas  
  - Construcción de un dataset  
  - Entrenamiento del modelo usando LoRA  
  - Evaluación mediante tareas GLUE  
  - Implementación y despliegue del chatbot

---

# Instalación

El proyecto fue ejecutado en **Ubuntu 22.04**, pero puede correr en cualquier entorno Linux con Python 3.10+.

---

## 1. Requisitos del sistema

- Python **3.10+**  
- Git  
- GPU NVIDIA con **CUDA 11.8+**  
- **Ollama** para generación y evaluación de datos sintéticos  

Instalar Ollama: https://ollama.com/download

```
curl -fsSL https://ollama.com/install.sh | sh

ollama pull mistral
ollama run mistral

ollama pull llama3
ollama run llama3
```

## 2. Instalar dependencias

El archivo requirements.txt incluye todas las librerías necesarias:

```
pip install -r requirements.txt
```

# Ejecución

## 1. Pipeline de Datos Sintéticos

Este proyecto genera datasets QA a partir de documentos PDF institucionales, los evalúa con modelos locales y los unifica para su uso en el entrenamiento del LLM.

---

### 1.1 Generación de datos QA desde PDFs institucionales
Se requiere que todos los archivos PDF sobre los cuales se van a generar los datos sintéticos estén en la carpeta .../data/synthetic/PDFs. Luego, se ejecuta lo siguiente

```
python3 source/synthetic_generation/qa_pdf_local.py
```

Este script:
- Lee PDFs desde data/synthetic/PDFs
- Divide el contenido en fragmentos
- Usa Mistral (Ollama) para generar pares pregunta-respuesta (Un archivo JOSN de QA por cada PDF)

Guarda los resultados en: 
```
data/synthetic/Outputs/
```

### 1.2 Evaluación de QA generados (Calidad y pertinencia)

```
python3 source/synthetic_generation/qa_judge.py
```

Usa LLaMA 3 (Ollama) para evaluar:
- claridad
- relevancia
- fidelidad
- puntaje (1–5)

Resultados generados en: 
```
data/synthetic/Evaluated_Outputs/
```

### 1.3 Unificación del dataset final

```
data/synthetic/Evaluated_Outputs/
```

Genera el dataset final para entrenamiento (Lo puedes renombrar como quieras):

```
data/training/IUPB_Dataset.jsonl
```

Formato en el que se unifica y transforma el dataset final (ChatML):

```
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

## 2. Entrenamiento del Modelo
Para entrenar el modelo DeepSeek con QLoRA:

```
python3 /home/estudiante1/workspaces/ac_llm_DeepSeek_V0.1/source/training/Training_Model.py
```

El entrenamiento:
- Usa cuantización 4-bit para reducir memoria
- Aplica LoRA en capas clave (Q, K, V, O)
- Utiliza 90% del dataset para entrenamiento y 10% para evaluación
- Guarda el proceso de entrenamiento en source/logs/salida.log

Guarda el/los checkpoints del modelo final en:

```
model/DeepSeek-R1/
```

## 3. Evaluación GLUE

El proyecto utiliza datasets adaptados a tareas del benchmark GLUE:
- SST2
- QQP
- QNLI

Ejecutar:

```
python3 /home/estudiante1/workspaces/ac_llm_DeepSeek_V0.1/source/evaluation/evaluate_GLUE.py
```

Métricas reportadas por tarea:
- Accuracy
- F1-score
- Manual F1
- MSE
- Cantidad de muestras evaluadas

Los logs generados se guardan en source/logs/salidaGLUE.log


## 4. Bot de Telegram

El bot utiliza el modelo entrenado para responder preguntas en español sobre documentos institucionales.

### 4.1 Configura tu token de Telegram
```
export TELEGRAM_TOKEN="TU_TOKEN_AQUI"
```

### 4.2 Ejecuta el Bot
```
python3 /home/estudiante1/workspaces/ac_llm_DeepSeek_V0.1/source/evaluation/runBot.py
```

Características:
- Limpia tokens especiales (<think>, <im_start>, etc.)
- Responde siempre en español
- Limita respuestas a 4000 caracteres
- Usa el modelo LoRA fusionado

Los logs generados se guardan en source/logs/salidaBot.log


# Resultados

<div style="display: flex; justify-content: space-between; align-items: center; width: 100%;">

  <img src="https://drive.google.com/uc?export=view&id=1IbzbLoCIL9nkx7JTdNm-nGGWwlWw2Evc" style="width: 24%;">

  <img src="https://drive.google.com/uc?export=view&id=1iaXNTRm32gnRJoTGJA2xz3uS60VnBYe4" style="width: 24%;">

  <img src="https://drive.google.com/uc?export=view&id=1MAx7yi5W69-mAK4tY6xsjwrciteUVRIe" style="width: 24%;">

  <img src="https://drive.google.com/uc?export=view&id=1ug-Ob0fgN3xKAZkW_rDdWdp_uGsYJx3I" style="width: 24%;">

</div>

El modelo responde con un contexto claro y con coherencia

# Resultados GLUE

Estos son los resultados de las tareas GLUE con las cuales se evaluó el modelo:

| Tarea                     | SST2 | QQP | QNLI |
|----------------------------|----------|----------|----------|
| **Cantidad de muestras evaluadas** | 30       | 30       | 30       |
| **Accuracy**               | 0.83     | 0.50     | 0.70     |
| **F1**                     | 0.85     | 0.61     | 0.75     |
| **MSE**                    | 0.16     | 0.50     | 0.30     |


# Contacto

Autora: [Juliana Jaramillo](https://github.com/JuliMendez8) \
Asesor: [Rubén Fonnegra](https://github.com/rubenfonnegra) 
