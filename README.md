# 📖 Traductor Español - Asturiano con Machine Learning

Este proyecto es un modelo de Machine Learning entrenado con 50,000 datos para traducir texto de Español a Asturiano. Utiliza `PyTorch`, `Transformers`, `Pandas` y `Datasets`. Puedes entrenar el modelo con más datos y probar su funcionamiento con los scripts proporcionados.

---

## 🚀 Instalación

Para comenzar, clona el repositorio e instala las librerías necesarias ejecutando:

```bash
git clone https://github.com/davidCabrero/ML_Traductor.git
```

```bash
pip install torch transformers pandas datasets
```

---

## 📂 Descarga de Archivos Necesarios

Antes de probar el modelo, descarga y descomprime los siguientes archivos desde Google Drive:

1️⃣ **`archivos_trained.zip`** → Descomprimir en `trained_model/`

2️⃣ **`archivos_results1.zip`** → Descomprimir en `results/checkpoint-11250/`

3️⃣ **`archivos_results2.zip`** → Descomprimir en `results/checkpoint-16875/`

🔗 *Enlace de descarga:* [Google Drive](https://drive.google.com/drive/folders/1ohGESoRzvavyx01w50rMbntjwvG6fqoC?usp=sharing)

---

## 🎯 Entrenar el Modelo

Para entrenar el modelo con más datos diferentes, ejecuta:

```bash
python entrenarModelo.py
```

---

## 📝 Probar el Modelo

Para probar la traducción de español a asturiano, ejecuta:

```bash
python probarModelo.py
```

---

## 📌 Estructura del Proyecto

```
📂 proyecto-traductor
 ├── 📂 trained_model
 ├── 📂 results
 │   ├── 📂 checkpoint-11250
 │   ├── 📂 checkpoint-16875
 ├── 📜 entrenarModelo.py
 ├── 📜 probarModelo.py
 ├── 📜 README.md
```
