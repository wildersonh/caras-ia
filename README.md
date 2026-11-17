# 🧠 Proyecto de Machine Learning y Reconocimiento Facial

Sistema interactivo en **Python** con interfaz gráfica en **Tkinter** para la detección y análisis facial:
- 👨 / 👩 **Clasificación de Género**: Identifica si la persona es Hombre o Mujer con porcentaje de confianza.
- 🎨 **Detección de Color de Piel**: Categoriza el tono (Muy Claro, Claro, Intermedio/Trigueño, Moreno, Oscuro) con cálculo cromático CIE-Lab / ITA y muestra el color exacto.
- 👁️ **Detección de Color de Ojos**: Analiza el iris (Marrón, Azul, Verde, Negro, Miel/Ámbar, Gris) y muestra el color de iris obtenido.
- 📍 **Puntos Clave (Facial Landmarks)**: Dibuja caja delimitadora y puntos clave en ojos, nariz, boca y contorno mandibular.
- 📝 **Módulo de Evaluación Interactiva ("Yo valúo la respuesta")**: Permite al usuario aprobar o corregir las predicciones de la IA y guardarlas en un historial (`evaluaciones_usuario.csv`).
- 🧠 **Entrenamiento Supervisado**: Botón para reentrenar y calibrar el clasificador de Machine Learning supervisado (`train_supervised_model.py`) con los datos evaluados.

---

## 🚀 Cómo Ejecutar la Aplicación

Abre la terminal o PowerShell en la carpeta del proyecto y ejecuta:

```bash
python app.py
```

---

## 📁 Estructura del Proyecto

```text
proyecto IA/
│
├── dataset/                   # ¡Tu dataset con las 30 imágenes de hombres y mujeres!
│   ├── hombre 1.jpg ... hombre 12.jpg
│   └── mujer 1.jpg ... mujer 18.jpg
│
├── app.py                     # Interfaz Gráfica interactiva en Tkinter con visor del dataset
├── face_engine.py             # Motor de IA: MediaPipe FaceLandmarks + ONNX Gender Classification
├── train_supervised_model.py  # Módulo de Machine Learning Supervisado (Scikit-Learn)
├── evaluaciones_usuario.csv   # Registro de evaluaciones del usuario (historial)
│
├── models/
│   ├── face_landmarker.task   # Modelo de 478 puntos faciales de MediaPipe
│   ├── genderage.onnx         # Red neuronal para clasificación de género y edad
│   └── modelo_supervisado.pkl # Modelo ML supervisado entrenado
│
└── requirements.txt           # Dependencias del proyecto
```

---

## 💡 Cómo usar la Aplicación

1. **Navegar por tu Dataset**:
   - La aplicación inicia automáticamente cargando la primera imagen de tu carpeta `dataset/`.
   - Utiliza los botones **"◀ Anterior"** y **"Siguiente ▶"** para pasar una por una por todas tus fotos (`hombre 1` a `hombre 12`, `mujer 1` a `mujer 18`).
   - Usa la **lista desplegable (Combobox)** para saltar directamente a cualquier imagen de tu dataset.
   - También cuentas con filtros rápidos: **"👨 Hombres (1-12)"**, **"👩 Mujeres (1-18)"** y **"🎲 Aleatorio"**.
   - O presiona **"📂 Cargar Imagen Propia..."** para subir cualquier foto externa.
2. **Visualizar la Respuesta de la IA**:
   - En la imagen verás la **caja delimitadora** y los **puntos clave**:
     - 🔵 Azul / Cyan: Ojos e iris
     - 🟢 Verde: Nariz (punta y puente)
     - 🔴 Rojo: Boca y labios
     - 🟡 Amarillo: Mentón y contorno
   - En el panel derecho verás:
     - Género predicho (Hombre / Mujer) con % de confianza.
     - Color de ojos con muestra visual de color y código HEX.
     - Color de piel con muestra visual de color y código HEX.
     - Edad aproximada.
3. **Evaluar la Respuesta ("Tú valúas")**:
   - Si la IA acertó: presiona **"✅ Todo Correcto"**.
   - Si deseas corregir algún valor: selecciona los valores reales en el menú desplegable y presiona **"💾 Guardar mi Evaluación"**.
   - Las respuestas se registran en `evaluaciones_usuario.csv` y actualizan la métrica de precisión en tiempo real.
4. **Entrenar el Modelo Supervisado**:
   - Presiona **"🧠 Entrenar Clasificador ML"** para ejecutar el entrenamiento supervisado con las evaluaciones registradas.
