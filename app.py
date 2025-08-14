import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

# ------------------------ Load model and objects ------------------------ #
def load_model(filename):
    return tf.keras.models.load_model(filename, compile=False)

def load_object(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

model = load_model('model.h5') 
inverse_model = load_model('model_inverse.h5')

scaler_x = load_object('scaler_x.pkl')
scaler_y = load_object('scaler_y.pkl')
inverse_ohe = load_object('ohe_inverse.pkl')

scaler = load_object('scaler.pkl')
ohe = load_object('ohe.pkl')

# ------------------------ Streamlit App ------------------------ #
st.title('Estimación de Producción y Telas')

# ------------------------ Producción ------------------------ #
st.header('Estimación de Prendas')

ancho = st.number_input('Ancho (cm)', min_value=0.0, value=0.0, step=1.0)
largo = st.number_input('Largo (m)', min_value=0.0, value=0.0, step=1.0)
peso = st.number_input('Peso (G/m2)', min_value=0.0, value=0.0, step=1.0)
cubre_cierre = st.selectbox('Cubre Cierre', [0, 1])
talla_prenda = st.selectbox('Talla Prenda', [1, 2, 3])
proveedor = st.selectbox('Proveedor', ohe.categories_[0])

def predict(input_data):
    obs = pd.DataFrame(input_data, index=[0])
    provider_encoded = ohe.transform(obs[['proveedor']])
    provider_df = pd.DataFrame(provider_encoded, columns=ohe.get_feature_names_out(['proveedor']))
    obs = pd.concat([obs.drop(columns=['proveedor']), provider_df], axis=1)
    obs = scaler.transform(obs)
    return model.predict(obs)

if st.button('Predecir Prendas'):
    input_data = {
        'ancho_cm': ancho,
        'largo_m': largo,
        'peso_g_m2': peso,
        'cubre_cierre': cubre_cierre,
        'talla_prenda': talla_prenda,
        'proveedor': proveedor
    }
    prediction = predict(input_data)
    st.success(f'Cantidad de producción estimada: {prediction[0][0]:.0f}')


# ------------------------ Telas (inversa) ------------------------ #
st.header('Estimación de Telas')

prendas_totales = st.number_input('Prendas Totales', min_value=0, value=0, step=1)
talla_prenda_inverse = st.selectbox('Talla Prenda (Inverse)', [1, 2, 3])
cubre_cierre_inverse = st.selectbox('Cubre Cierre (Inverse)', [0, 1])
proveedor_inverse = st.selectbox('Proveedor (Inverse)', inverse_ohe.categories_[0])

def inverse_predict(input_data):
    obs = pd.DataFrame(input_data, index=[0])
    cat_encoded = inverse_ohe.transform(obs[['proveedor', 'talla_prenda']])
    cat_df = pd.DataFrame(cat_encoded, columns=inverse_ohe.get_feature_names_out(['proveedor', 'talla_prenda']))
    obs = pd.concat([obs.drop(columns=['proveedor', 'talla_prenda']), cat_df], axis=1)
    obs_scaled = scaler_x.transform(obs)
    pred_scaled = inverse_model.predict(obs_scaled)
    pred = scaler_y.inverse_transform(pred_scaled)
    return [float(v) for v in pred[0]]

if st.button('Predecir Telas'):
    input_data = {
        'prendas_totales': prendas_totales,
        'cubre_cierre': cubre_cierre_inverse,
        'talla_prenda': talla_prenda_inverse,
        'proveedor': proveedor_inverse
    }
    prediction = inverse_predict(input_data)
    st.success(f'Peso G/m2: {prediction[0]:.2f}, Largo m: {prediction[1]:.2f}, Ancho cm: {prediction[2]:.2f}')

# ------------------------ Footer ------------------------ #
st.write('-------------------------------------')
st.write('Desarrollado por [Esteban Flores] ')
st.write('-------------------------------------')

# ------------------------ End Streamlit App ------------------------ #