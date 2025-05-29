# 📦 Kerakli kutubxonalar
import streamlit as st
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image

st.set_page_config(page_title="MNIST MLP", page_icon="🧠")
st.title("🔢 Raqamlarni MLP bilan Klassifikatsiya qilish")

# 📥 MNIST ma’lumotlarini yuklaymiz
@st.cache_resource
def load_data_and_model():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X / 255.0
    y = y.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=20, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

with st.spinner("Model o‘qitilmoqda..."):
    model, acc = load_data_and_model()

st.success(f"✅ Model tayyor (Aniqlik: {acc:.2%})")

# 📷 Foydalanuvchi rasm yuklashi
uploaded_file = st.file_uploader("Raqam rasmi yuklang (28x28, qora-oq)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L").resize((28, 28))
    st.image(image, caption="Yuklangan rasm", width=100)

    image_array = 255 - np.array(image)
    X_input = image_array.flatten().reshape(1, -1) / 255.0

    prediction = model.predict(X_input)[0]
    st.success(f"🧠 Bashorat: **{prediction}**")

    if st.checkbox("📊 Piksel qiymatlarini ko‘rsatish"):
        st.write(image_array)
