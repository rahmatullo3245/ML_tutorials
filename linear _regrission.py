import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("📈 Linear Regression with Two Features")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data Preview")
    st.write(df.head())

    if df.shape[1] < 3:
        st.error("The dataset must contain at least two input features and one target column.")
    else:
        all_columns = df.columns.tolist()

        x1_column = st.selectbox("Select feature X1:", all_columns, index=0)
        x2_column = st.selectbox("Select feature X2:", all_columns, index=1)
        y_column = st.selectbox("Select target Y:", all_columns, index=2)

        if x1_column != x2_column and x1_column != y_column and x2_column != y_column:
            X = df[[x1_column, x2_column]]
            y = df[y_column]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)

            st.subheader("🔮 Make a New Prediction")
            new_x1 = st.number_input(f"Enter value for {x1_column} (X1):", value=0.0)
            new_x2 = st.number_input(f"Enter value for {x2_column} (X2):", value=0.0)

            if st.button("Predict"):
                new_input = [[new_x1, new_x2]]
                prediction = model.predict(new_input)[0]

                # Make predictions and calculate metrics
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                st.markdown("### 📌 Model Evaluation")
                st.markdown(f"- **Mean Absolute Error (MAE):** {mae:.2f}")
                st.markdown(f"- **Mean Squared Error (MSE):** {mse:.2f}")
                st.markdown(f"- **R² Score:** {r2:.2f}")

                st.markdown("### 🎯 Predicted Value")
                st.markdown(f"The predicted value for **{y_column}** is: **`{prediction:.2f}`**")
        else:
            st.error("Please select three different columns: X1, X2, and Y.")
