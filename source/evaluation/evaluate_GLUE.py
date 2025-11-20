import torch
import os
import json
from datasets import load_dataset
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from evaluate import load
import numpy as np
from tqdm import tqdm
import random

# RUTAS
# DATASET_OUTPUT_FOLDER: Ruta del dataset de evaluación

DATASET_OUTPUT_FOLDER = "/home/estudiante1/workspaces/ac_llm_DeepSeek_V0.1/data/glue"

LOCAL_FILES = {
    "sst2": os.path.join(DATASET_OUTPUT_FOLDER, "SST2_test_V2.json"),
    "qqp": os.path.join(DATASET_OUTPUT_FOLDER, "QQP_test_V2.json"),
    "qnli": os.path.join(DATASET_OUTPUT_FOLDER, "QNLI_test_V2.json"),
}

# Función para generar prompt con formato para DeepSeek-R1
def format_deepseek_prompt_classification(user_input, system_prompt):
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    return prompt

# Calcular MSE 
def calculate_mse(predictions, references):
    return np.mean((references - predictions) ** 2)

# Evaluar con datasets locales
def evaluate_glue(model_pipeline, tasks=None, samples_per_task=None, seed=42):
    #Evalúa un modelo de lenguaje fine-tuneado en las tareas de GLUE. 
    #Prioriza la carga de datasets locales y omite cualquier tarea no definida localmente.
    random.seed(seed)
    
    if tasks is None:
        tasks = ["sst2", "qqp", "qnli"] #tareas a evaluar
    
    all_results = {}
    
    System_Prompt = "Eres un clasificador EXTREMADAMENTE estricto. Tu unica tarea es analizar el texto y **SOLAMENTE** responder con la etiqueta numérica correspondiente (0, 1, o 2). **Tu respuesta debe ser un solo dígito sin explicación, espacios o texto adicional. Si por alguna razón imprimes texto extra, DEBE ser en ESPAÑOL.**"
    
    f1_metric = load("f1")

    for task in tasks:
        print(f"\nEvaluando {task.upper()} con {samples_per_task} muestras")
        
        if task in LOCAL_FILES:
            file_path = LOCAL_FILES[task]
            if not os.path.exists(file_path):
                 print(f"Archivo local no encontrado en {file_path}. Saltando tarea.")
                 continue

            dataset = load_dataset("json", data_files={"validation": file_path}, split="validation")
            dataset = {"validation": dataset}
            
            try:
                glue_metric = load("glue", task)
            except Exception as e:
                print(f"Error al cargar la métrica 'glue/{task}': {e}. Saltando tarea.")
                continue

            splits = ["validation"]
            
            print(f"Cargando dataset LOCAL para {task.upper()} desde: {file_path}")
        else:
            print(f"Advertencia: La tarea '{task}' no está definida en LOCAL_FILES. Omitiendos")
            continue

        task_results = {}
        
        for split in splits:
            results = []
            labels = []
            
            val_data = dataset[split] if isinstance(dataset, dict) else dataset
            
            total_examples = len(val_data)
            
            samples_to_use = min(samples_per_task if samples_per_task is not None else total_examples, total_examples)
            sample_indices = random.sample(range(total_examples), samples_to_use)
            
            for idx in tqdm(sample_indices):
                example = val_data[idx]
                
                #Prompts por tarea
                if task == "sst2":
                    task_prompt = f"Analisis de sentimiento: {example['sentence']}\nResponde 0 para negativo, 1 para positivo."
                elif task == "qqp":
                    task_prompt = f"¿Son estas preguntas equivalentes?\nQ1: {example['question1']}\nQ2: {example['question2']}\nResponde 0 o 1."
                elif task == "qnli":
                    task_prompt = f"¿Esta oracion responde la pregunta?\nPregunta: {example['question']}\nOracion: {example['sentence']}\nResponde 0 para no, 1 para si."
                else:
                    tqdm.write(f"Advertencia: Tarea '{task}' no implementada en la lógica de prompt.")
                    continue
                
                full_prompt = format_deepseek_prompt_classification(task_prompt, System_Prompt)
                
                try:
                    output = model_pipeline(
                        full_prompt,
                        max_new_tokens=3, 
                        return_full_text=False,
                        do_sample=False,
                        temperature=0.0,
                    )
                    
                    response = output[0]["generated_text"].strip()
                    
                    # Se extrae el primer dígito numérico dado que se esperan respuestas de 0 o 1
                    digits = ''.join(filter(str.isdigit, response))
                    
                    if digits:
                        pred = int(digits[0]) 
                    else:
                        pred = -1 
                        
                    if pred != -1:
                        results.append(pred)
                        labels.append(example["label"])
                    else:
                        tqdm.write(f"No se pudo extraer la predicción numérica para el ejemplo {idx}. Respuesta: '{response[:50]}...'")
                        continue
                        
                except Exception as e:
                    tqdm.write(f"Error procesando ejemplo {idx}: {str(e)}")
                    continue
            
            if results:
                predictions_np = np.array(results)
                references_np = np.array(labels)

                # Cálculo de métricas
                metrics = glue_metric.compute(predictions=predictions_np, references=references_np)
                
                try:
                    f1_value = f1_metric.compute(predictions=predictions_np, references=references_np)["f1"]
                    metrics["f1_manual"] = float(f1_value)
                except Exception as e:
                    metrics["f1_manual"] = f"No se pudo calcular F1: {e}"

                mse_value = calculate_mse(predictions_np, references_np)
                metrics["mse"] = float(mse_value)

                task_results[split] = {
                    "metrics": metrics,
                    "num_samples": len(results)
                }
            else:
                task_results[split] = {
                    "metrics": "Fallo en procesar cualquier ejemplo",
                    "num_samples": 0
                }
                
        all_results[task] = task_results["validation"]
    
    return all_results

def print_results(results):
    print("\nGLUE Benchmark Results:")
    print("-" * 50)
        
    for task, metrics in results.items():
        print(f"\n{task.upper()}:")
        if isinstance(metrics, dict) and "num_samples" in metrics:
            print(f"  Samples evaluated: {metrics['num_samples']}")
            print(f"  Metrics: {metrics['metrics']}")
        else:
             print(f"  Metrics: {metrics}")

if __name__ == "__main__":
    # Modelo del proyecto
    base_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" 
    lora_path = "/home/estudiante1/workspaces/ac_llm_DeepSeek_V0.1/model/DeepSeek-R1" #Directorio dondeesta guardado nuestro modelo ya entrenado

    print(f"Cargando Tokenizer desde: {lora_path}")
    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    
    print(f"Cargando Modelo Base desde: {base_model_id}")
    device_map = "auto"
    if not torch.cuda.is_available():
        device_map = "cpu"
        print("Advertencia: CUDA no disponible, usando device_map='cpu'. El proceso será lento.")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.merge_and_unload() 

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    print("\nIniciando Evaluación con Datasets Locales")
    results = evaluate_glue(pipe, tasks=["sst2", "qqp", "qnli"], samples_per_task=30)
    print_results(results)
