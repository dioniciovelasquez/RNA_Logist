import streamlit as st
import numpy as np

st.title("🧠 Calculadora de Neurona Artificial")

st.markdown("""Esta herramienta permite calcular la salida de una neurona con función de activación **sigmoide logística** (logsig), incluyendo el sesgo.""")

# Número de entradas (sin contar el sesgo)
num_inputs = st.slider("Selecciona el número de entradas (sin contar el sesgo):", 1, 10, 4)

# Entradas y pesos
inputs = []
weights = []

st.subheader("🔢 Ingresar valores de entradas y pesos")

for i in range(num_inputs):
    x = st.number_input(f"Entrada x{i+1}", value=0.0, key=f"x{i}")
    w = st.number_input(f"Peso w{i+1}", value=0.0, key=f"w{i}")
    inputs.append(x)
    weights.append(w)

# Sesgo
st.subheader("⚙️ Sesgo")
bias_weight = st.number_input("Peso del sesgo (bias)", value=0.0, key="bias")

# Agregar sesgo como entrada constante 1
inputs.append(1.0)
weights.append(bias_weight)

# Cálculo de la salida
def logsig(x):
    return 1 / (1 + np.exp(-x))

net_input = np.dot(inputs, weights)
output = logsig(net_input)

# Mostrar resultados
st.subheader("📤 Resultado")
st.write(f"**Entrada neta (net_i):** {net_input:.4f}")
st.write(f"**Salida activada (y_i):** {output:.4f}")
