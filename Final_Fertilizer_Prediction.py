import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("Final_Fertilizer_Dataset.csv")

# Initialize LabelEncoders for each categorical column
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
fertilizer_encoder = LabelEncoder()

# Fit and transform the categorical columns
Soil = soil_encoder.fit_transform(data['Soil'])
Crop = crop_encoder.fit_transform(data['Crop'])
Fertilizer = fertilizer_encoder.fit_transform(data['Fertilizer'])

encoded_df_in = pd.DataFrame({
    'Soil': Soil,
    'Crop': Crop,
})
encoded_df_out = pd.DataFrame({
    'Fertilizer': Fertilizer,
})
Input = pd.concat([data[['Temperature', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous', 'Carbon', 'PH']], encoded_df_in], axis=1)
Output = encoded_df_out

Input_train, Input_test, Output_train, Output_test = train_test_split(Input, Output, test_size=0.2)

Output_train = Output_train.values.ravel()
Output_test = Output_test.values.ravel()

model = RandomForestClassifier(n_estimators=500, max_depth=8)
model.fit(Input_train, Output_train)

y_pred = model.predict(Input_test)
print(f"Accuracy: {accuracy_score(Output_test, y_pred):.4f}")


def suggest_fertilizer(temperature, moisture, nitrogen, potassium, phosphorous, carbon, ph, soil_type, crop_type):
    try:
        # Transform soil and crop types using the already fitted encoders
        soil_encoded = soil_encoder.transform([soil_type])[0]
        crop_encoded = crop_encoder.transform([crop_type])[0]

        # Create a dataframe with the input values
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

        # Get the predicted class and its probability
        prediction_encoded = model.predict(input_data)[0]
        prediction = fertilizer_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get confidence (probability) of the predicted class
        proba = model.predict_proba(input_data)
        confidence = proba[0][prediction_encoded] * 100  # Convert to percentage
        
        return print(f"Recommended Fertilizer: {prediction})")

    except Exception as e:
        print(f"Error: {e}")

# Example usage
suggest_fertilizer(50.17984508, 0.725892978, 66.70187235, 96.42906521, 76.96356027, 0.496299708, 6.227357805, 'Loamy Soil', 'rice')