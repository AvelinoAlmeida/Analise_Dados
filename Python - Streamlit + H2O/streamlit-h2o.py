import streamlit as st
import h2o
import pandas as pd
import numpy as np

h2o.init()

model_path = "DRF_carros" 
model = h2o.load_model(model_path)

def make_prediction():
    input_data = {
        "mpg": st.session_state.mpg,
        "displacement": st.session_state.displacement,
        "horsepower": st.session_state.horsepower,
        "weight": st.session_state.weight,
        "acceleration": st.session_state.acceleration,
    }

    input_df = pd.DataFrame([input_data])
    input_h2o = h2o.H2OFrame(input_df)

    prediction = model.predict(input_h2o).as_data_frame()
    st.session_state["prediction_label"] = prediction.iloc[0, 0]
    st.session_state["prob_american"] = np.round(prediction.iloc[0, 1],2)
    st.session_state["prob_european"] = np.round(prediction.iloc[0, 2],2)
    st.session_state["prob_japanese"] = np.round(prediction.iloc[0, 3],2)    
    st.session_state["probabilities"] = prediction.iloc[0, 1:].to_dict()

st.title('Car prediction DSS :sunglasses:')

st.image("cars.jpg", caption="Just some cars...")


# TODO: ir buscar limites dos sliders ao csv

st.slider("MPG", 0.0, 50.0, 25.0, key="mpg", on_change=make_prediction)
st.slider("Displacement", 0.0, 500.0, 250.0, key="displacement", on_change=make_prediction)
st.slider("Horsepower", 0.0, 500.0, 150.0, key="horsepower", on_change=make_prediction)
st.slider("Weight", 0.0, 6000.0, 3000.0, key="weight", on_change=make_prediction)
st.slider("Acceleration", 0.0, 20.0, 10.0, key="acceleration", on_change=make_prediction)

st.write(f"Predicted origin: :red[**{st.session_state.get('prediction_label')}**]")
st.write('**Probability (american):** '+str(st.session_state.get('prob_american'))+'%')
st.write('**Probability (european):** '+str(st.session_state.get('prob_european'))+'%')
st.write('**Probability (japanese):** '+str(st.session_state.get('prob_japanese'))+'%')

if "probabilities" in st.session_state:
    prob_df = pd.DataFrame(
        list(st.session_state["probabilities"].items()),
        columns=["Label", "Probability"],
    ).set_index("Label")

    st.bar_chart(prob_df)
