import os
import json

# RUTAS
# JSONspath: Carpeta donde están los archivos .json generados con QA
# DATASETPATH: Carpeta donde se guardará el dataset final
# OUTPUT_FILE: Ruta completa del archivo final (incluido el nombre)

JSONspath = "/home/estudiante1/workspaces/ac_llm_DeepSeek_V0.1/data/synthetic/Outputs"
DATASETPATH = "/home/estudiante1/workspaces/ac_llm_DeepSeek_V0.1/data/training"
OUTPUT_FILE = os.path.join(DATASETPATH, "IUPB_Dataset.jsonl")

def unify_json_files(folder_path, output_file):
    with open(output_file, "w", encoding="utf-8") as fout:
        # Se recorren todos los archivos del directorio
        for filename in os.listdir(folder_path): 
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                question = item.get("question", "").strip()
                                answer = item.get("answer", "")

                                # Si la respuesta que se encuentra es una lista, se convierte en un string
                                if isinstance(answer, list): 
                                    answer = " ".join(answer)

                                # Estructura final del Dataset (si bien no es la estructura requerida para el modelo DeepSeek-R1-Distill-Qwen-7B, 
                                # es una estructura de fácil manejo sobre la cual se pueden hacer reestructuraciones a necesidad de forma sencilla)
                                formatted = {
                                    "messages": [
                                        {"role": "user", "content": question},
                                        {"role": "assistant", "content": answer}
                                    ]
                                }

                                fout.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                        else:
                            print(f"El archivo {filename} no contiene una lista.")
                    except json.JSONDecodeError as e:
                        print(f"Error leyendo {filename}: {e}")

    print(f"Dataset JSONL creado en: {output_file}")

# Este bloque se ejecutará unicamente si corres el script directamente.
if __name__ == "__main__":
    unify_json_files(JSONspath, OUTPUT_FILE)