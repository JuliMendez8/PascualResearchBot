import json
import os
import subprocess
from pypdf import PdfReader

# RUTAS
# PDF_FOLDER: Carpeta con el archivo PDF sobre el cuales se van a generar datos sinteticos en formatos QNLI, QPP y SST2 del Benchamark GLUE
# OUTPUT_FOLDER: Carpeta donde se guardarán los archivos .json generados para las tareas GLUE

PDF_FOLDER = "/home/estudiante1/workspaces/ac_llm_DeepSeek_V0.1/data/synthetic/Test_PDFs"
OUTPUT_FOLDER = "/home/estudiante1/workspaces/ac_llm_DeepSeek_V0.1/data/glue"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Función de extracción de texto 
def extract_pdfs_text(pdf_path, start_page=12):
    reader = PdfReader(pdf_path)
    text = ""
    total_pages = len(reader.pages)
    start_page = min(start_page - 1, total_pages - 1) 
    for i in range(start_page, total_pages):
        page_text = reader.pages[i].extract_text()
        if page_text:
            text += page_text.strip() + "\n"
    return text

# Función de división del texto extraido
def split_text(text, max_chars=3000):
    text = " ".join(text.split()) 
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

# Llamar al modelo local seleccionado (mistral en este caso) con Ollama.
def run_ollama(prompt, model="mistral"):
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    output = result.stdout.decode("utf-8").strip()
    return output

def safe_json_parse(output):
    # Se intenta parsear la salida de Ollama como JSON.
    try:
        return json.loads(output)
    except:
        print("No se pudo parsear como JSON. Texto devuelto:")
        print(output)
        return []

# Generación de dataset QQP
def generate_QQP_data(chunks, total_entries=30, model="mistral"):
    prompt_template = """
    Genera {n} pares de preguntas en español basadas en el siguiente texto.
    Cada par debe estar en formato JSON con los siguientes campos:
    - question1: primera pregunta
    - question2: segunda pregunta
    - label: 1 si las preguntas tienen la misma intención, 0 si no
    - idx: identificador único incremental
    Devuelve solo un JSON válido en formato lista.

    Texto:
    {chunk}
    """

    dataset = []
    idx_counter = 0
    for chunk in chunks:
        remaining = total_entries - len(dataset)
        if remaining <= 0:
            break
        output = run_ollama(prompt_template.format(n=min(5, remaining), chunk=chunk), model=model)
        data = safe_json_parse(output)
        for item in data:
            item["idx"] = idx_counter
            idx_counter += 1
        dataset.extend(data)
    return dataset[:total_entries]

# Generación de dataset SST2
def generate_SST2_data(chunks, total_entries=30, model="mistral"):
    prompt_template = """
    Genera {n} oraciones en español basadas en el siguiente texto.
    A cada oración asígnale una etiqueta de sentimiento:
    - 1 si expresa una opinión o tono positivo
    - 0 si expresa una opinión o tono negativo
    Devuelve solo un JSON válido en formato lista, con:
    - sentence
    - label
    - idx

    Texto:
    {chunk}
    """

    dataset = []
    idx_counter = 0
    for chunk in chunks:
        remaining = total_entries - len(dataset)
        if remaining <= 0:
            break
        output = run_ollama(prompt_template.format(n=min(10, remaining), chunk=chunk), model=model)
        data = safe_json_parse(output)
        for item in data:
            item["idx"] = idx_counter
            idx_counter += 1
        dataset.extend(data)
    return dataset[:total_entries]

# Generación de dataset QNLI
def generate_QNLI_data(chunks, total_entries=30, model="mistral"):
    prompt_template = """
    Genera {n} pares de pregunta y oración en español basadas en el siguiente texto.
    Cada par debe tener uno de los siguientes valores para 'label':
    - 1 si la oración responde o está relacionada con la pregunta (entailment)
    - 0 si la oración NO responde, contradice o es irrelevante a la pregunta (not_entailment)
    
    Asegúrate de que haya variedad entre pares con label 1 y label 0, genera al menos la mitad de las preguntas con label 0.
    Devuelve solo un JSON válido en formato lista con estos campos:
    - question
    - sentence
    - label
    - idx

    Texto:
    {chunk}
    """

    dataset = []
    idx_counter = 0
    for chunk in chunks:
        remaining = total_entries - len(dataset)
        if remaining <= 0:
            break
        output = run_ollama(prompt_template.format(n=min(10, remaining), chunk=chunk), model=model)
        data = safe_json_parse(output)
        for item in data:
            item["idx"] = idx_counter
            idx_counter += 1
        dataset.extend(data)
    return dataset[:total_entries]

# Este bloque se ejecutará unicamente si corres el script directamente.
if __name__ == "__main__":
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("No se encontraron PDFs en la carpeta de test.")
        exit()

    all_text = ""
    for pdf_file in pdf_files:
        print(f"Leyendo {pdf_file} (desde la página 12)...")
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        all_text += extract_pdfs_text(pdf_path, start_page=12) + "\n"

    chunks = split_text(all_text)

    print("\n=== Generando datasets de test GLUE ===\n")

    qqp_data = generate_QQP_data(chunks, 30)
    sst2_data = generate_SST2_data(chunks, 30)
    qnli_data = generate_QNLI_data(chunks, 30)

    with open(os.path.join(OUTPUT_FOLDER, "QQP_test.json"), "w", encoding="utf-8") as f:
        json.dump(qqp_data, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUTPUT_FOLDER, "SST2_test.json"), "w", encoding="utf-8") as f:
        json.dump(sst2_data, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUTPUT_FOLDER, "QNLI_test.json"), "w", encoding="utf-8") as f:
        json.dump(qnli_data, f, indent=2, ensure_ascii=False)

    print("Datasets generados exitosamente en:")
    print(f" - {OUTPUT_FOLDER}/QQP_test.json")
    print(f" - {OUTPUT_FOLDER}/SST2_test.json")
    print(f" - {OUTPUT_FOLDER}/QNLI_test.json")
