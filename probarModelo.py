import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Cargar el modelo entrenado
model_path = "./trained_model"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

def translate(text):
    model.eval()
    inputs = tokenizer("translate Spanish to Asturian: " + text, return_tensors="pt", max_length=1000, truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=1000,  # longitud máxima de la secuencia de salida
            num_beams=100,     # Usar beam search para mejorar la calidad de la traducción
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Ejemplo de prueba
if __name__ == "__main__":
    sample_text = "Yo estoy bien, ¿cómo estás?, ¿tienes tiempo libre? Lo mejor es viajar a Asturias."
    translated_text = translate(sample_text)
    print("Traducción:", translated_text)
