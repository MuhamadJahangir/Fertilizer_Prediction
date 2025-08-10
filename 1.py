import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Final_Fertilizer_Dataset.csv")

data = load_data()

# Initialize LabelEncoders
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
fertilizer_encoder = LabelEncoder()

# Fit encoders
Soil = soil_encoder.fit_transform(data['Soil'])
Crop = crop_encoder.fit_transform(data['Crop'])
Fertilizer = fertilizer_encoder.fit_transform(data['Fertilizer'])

# Prepare input and output
encoded_df_in = pd.DataFrame({'Soil': Soil, 'Crop': Crop})
encoded_df_out = pd.DataFrame({'Fertilizer': Fertilizer})
Input = pd.concat([data[['Temperature', 'Moisture', 'Nitrogen', 'Potassium', 
                        'Phosphorous', 'Carbon', 'PH']], encoded_df_in], axis=1)
Output = encoded_df_out

# Train model
@st.cache_resource
def train_model():
    Input_train, Input_test, Output_train, Output_test = train_test_split(Input, Output, test_size=0.2)
    Output_train = Output_train.values.ravel()
    Output_test = Output_test.values.ravel()
    
    model = RandomForestClassifier(n_estimators=500, max_depth=8)
    model.fit(Input_train, Output_train)
    
    # Calculate accuracy for display
    y_pred = model.predict(Input_test)
    accuracy = accuracy_score(Output_test, y_pred)
    
    return model, accuracy

model, accuracy = train_model()

# Streamlit app
st.title("Fertilizer Recommendation System")

st.write(f"Model Accuracy: {accuracy:.2%}")

st.header("Input Parameters")

# Create input widgets
col1, col2 = st.columns(2)

with col1:
    temperature = st.slider("Temperature (°C)", min_value=0, max_value=50, value=25)
    moisture = st.slider("Moisture (%)", min_value=0, max_value=100, value=50)
    nitrogen = st.slider("Nitrogen (ppm)", min_value=0, max_value=200, value=50)
    potassium = st.slider("Potassium (ppm)", min_value=0, max_value=200, value=50)

with col2:
    phosphorous = st.slider("Phosphorous (ppm)", min_value=0, max_value=200, value=50)
    carbon = st.slider("Carbon (%)", min_value=0, max_value=10, value=2)
    ph = st.slider("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    soil_type = st.selectbox("Soil Type", soil_encoder.classes_)
    crop_type = st.selectbox("Crop Type", crop_encoder.classes_)

# Predict button
if st.button("Recommend Fertilizer"):
    # Encode soil and crop
    soil_encoded = soil_encoder.transform([soil_type])[0]
    crop_encoded = crop_encoder.transform([crop_type])[0]
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'Temperature': [temperature],
        'Moisture': [moisture],
        'Nitrogen': [nitrogen],
        'Potassium': [potassium],
        'Phosphorous': [phosphorous],
        'Carbon': [carbon],
        'PH': [ph],
        'Soil': [soil_encoded],
        'Crop': [crop_encoded]
    })
    
    # Make prediction
    prediction_encoded = model.predict(input_data)[0]
    recommended_fertilizer = fertilizer_encoder.inverse_transform([prediction_encoded])[0]
    
    st.success(f"Recommended Fertilizer: **{recommended_fertilizer}**")
    
    # Show feature importance (optional)
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': Input.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    st.bar_chart(feature_importance.set_index('Feature'))