import json
from pypdf import PdfReader
import subprocess
import os

# RUTAS
# PDF_FOLDER: Carpeta con todos los archivos PDF sobre los cuales se van a generar datos sinteticos en formato QA
# OUTPUT_FOLDER: Carpeta donde se guardarán los archivos .json generados con formato QA

PDF_FOLDER = "/home/estudiante1/workspaces/ac_llm_DeepSeek_V0.1/data/synthetic/PDFs"
OUTPUT_FOLDER = "/home/estudiante1/workspaces/ac_llm_DeepSeek_V0.1/data/synthetic/Outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 1. Leer PDF
def extract_pdfs_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for pagina in reader.pages:
        text += pagina.extract_text() + "\n"
    return text

# 2. Dividir texto en chunks
def split_text(text, max_chars=2000):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

# 3. Llamar al modelo local seleccionado (mistral en este caso) con Ollama. Se le da un rol de "Generador"
def generate_qa(chunk, n_questions=5, model="mistral"):
    prompt = f"""
    Genera {n_questions} preguntas y respuestas en español
    basadas en el siguiente texto. Devuelve solo un JSON válido:

    [
      {{"question": "...", "answer": "..."}}
    ]

    Texto:
    {chunk}
    """
    
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    
    output = result.stdout.decode("utf-8").strip()
    
    try:
        return json.loads(output)
    except:
        print("No se pudo parsear como JSON. Texto devuelto:")
        print(output)
        return []

# 4. Función principal para llamar demás funciones para generar las preguntas QA
def process_pdf(pdf_path, output_json, questions_per_chunk=5):
    text = extract_pdfs_text(pdf_path)
    chunks = split_text(text)

    qa_total = []
    for i, chunk in enumerate(chunks, 1):
        print(f"Procesando chunk {i}/{len(chunks)}...")
        qa = generate_qa(chunk, questions_per_chunk)
        qa_total.extend(qa)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(qa_total, f, indent=2, ensure_ascii=False)

    print(f"Q&A guardados")


# 5. Procesar todos los PDFs de la carpeta PDFs. Este bloque se ejecutará unicamente si corres el script directamente.
if __name__ == "__main__":
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("No se encontraron PDFs en la carpeta 'PDFs'.")
    else:
        for pdf_file in pdf_files:
            pdf_path = os.path.join(PDF_FOLDER, pdf_file)
            output_name = os.path.splitext(pdf_file)[0] + "_qa.json"
            output_path = os.path.join(OUTPUT_FOLDER, output_name)

            if os.path.exists(output_path): #Si hay algún PDF que ya se haya procesado con anterioridad, se ignora. Si se quiere volver a procesar se debe eliminar de la carpeta de Outputs
                print(f"PDF {pdf_file} ya procesado.")
                continue

            try:
                process_pdf(pdf_path, output_path, questions_per_chunk=3)
                print(f"Procesado: {pdf_file}")
            except Exception as e:
                print(f"Error {pdf_file}: {e}")
