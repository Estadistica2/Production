import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

# ------------------------ Load model and objects ------------------------ #
def load_model(filename):
    model = tf.keras.models.load_model(filename, compile=False)
    return model

def load_object(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

model = load_model('model.h5')
scaler = load_object('scaler.pkl')
ohe = load_object('ohe.pkl')

# ------------------------ Streamlit App ------------------------ #
st.title('Production Cost Prediction')

# ------------------------ Inputs ------------------------ #
ancho = st.number_input('Ancho (cm)', min_value = 0.0, value = 0.0, step = 1.0)
largo = st.number_input('Largo (m)', min_value = 0.0, value = 0.0, step = 1.0)
peso = st.number_input('Peso (G mt2)', min_value = 0.0, value = 0.0, step = 1.0)
cubre_cierre = st.selectbox('Cubre Cierre', [0, 1])
talla_prenda = st.selectbox('Talla Prenda', [1, 2, 3])
proveedor = st.selectbox('Proveedor', ohe.categories_[0])

# ------------------------ Prediction Function ------------------------ #
def predict(input_data):
    observation = pd.DataFrame(input_data, index = [0])

    provider_encoded = ohe.transform(observation[['proveedor']])
    provider_df = pd.DataFrame(provider_encoded, columns = ohe.get_feature_names_out(['proveedor']))

    observation = pd.concat([observation.drop(columns=['proveedor']), provider_df], axis = 1)
    observation = scaler.transform(observation)

    return model.predict(observation)

# ------------------------ Predict Button ------------------------ #
if st.button('Predict'):
    input_data = {
        'ancho_cm': ancho,
        'largo_m': largo,
        'peso_g_m2': peso,
        'cubre_cierre': cubre_cierre,
        'talla_prenda': talla_prenda,
        'proveedor': proveedor
    }

    prediction = predict(input_data)
    st.success(f'La cantidad de producción estimada es de: {prediction[0][0]:.0f}')
