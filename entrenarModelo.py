import pandas as pd
import torch
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# Cargar los datos
with open('es_train.txt', 'r', encoding='utf-8') as f:
    es_lines = [line.strip() for line in f.readlines()[:50000]]

with open('ast_train.txt', 'r', encoding='utf-8') as f:
    ast_lines = [line.strip() for line in f.readlines()[:50000]]

# Crear DataFrame
df = pd.DataFrame({'Spanish': es_lines, 'Asturian': ast_lines})

# Convertir en dataset de Hugging Face
dataset = Dataset.from_pandas(df)
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Definir el modelo y tokenizador
model_name = 't5-small'  # Se puede cambiar por 't5-base' o más grande
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Preprocesamiento
def preprocess_function(examples):
    inputs = ["translate Spanish to Asturian: " + ex for ex in examples['Spanish']]
    targets = [ex for ex in examples['Asturian']]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Configurar entrenamiento
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy='epoch',
    save_total_limit=2,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Iniciar entrenamiento
trainer.train()

# Guardar el modelo entrenado
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
