import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Food Nutrition Explorer", layout="wide")

st.title("Food Nutrition Explorer & Predictor")
st.caption(
    "Upload a food nutrition CSV file to explore nutritional values "
    "and predict a selected nutrition value using other numeric features."
)

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.drop_duplicates().copy()

    st.subheader("Data Preview")
    st.write(f"Number of rows: {len(df)}")
    st.dataframe(df.head(10), hide_index=True, width="stretch")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        st.error("This file has no numeric columns to explore or use for prediction.")
        st.stop()

    st.divider()

    st.subheader("Explore the Data")

    selected_col = st.selectbox(
        "Choose a numeric column",
        numeric_cols
    )

    chart_type = st.selectbox(
        "Choose a chart type",
        ["Line", "Scatter", "Histogram"]
    )

    chart_df = df[[selected_col]].dropna().copy()
    chart_df["Record"] = range(1, len(chart_df) + 1)

    if chart_type == "Line":
        fig = px.line(
            chart_df,
            x="Record",
            y=selected_col,
            title=f"{selected_col} by Record"
        )
    elif chart_type == "Scatter":
        fig = px.scatter(
            chart_df,
            x="Record",
            y=selected_col,
            title=f"{selected_col} by Record"
        )
    else:
        fig = px.histogram(
            chart_df,
            x=selected_col,
            title=f"Distribution of {selected_col}"
        )

    st.plotly_chart(fig, width="stretch")

    st.divider()

    st.subheader("Predict a Nutrition Value")

    preferred_targets = [
        "Calories (kcal)",
        "Protein (g)",
        "Sugars (g)"
    ]

    target_options = [
        col for col in preferred_targets
        if col in numeric_cols
    ]

    if not target_options:
        st.error(
            "The CSV needs at least one of these numeric columns: "
            "Calories (kcal), Protein (g), or Sugars (g)."
        )
        st.stop()

    target_col = st.selectbox(
        "Target: What do you want to predict?",
        target_options
    )

    input_options = [
        col for col in numeric_cols
        if col != target_col
    ]

    default_inputs = [
        col for col in ["Protein (g)", "Fat (g)", "Carbohydrates (g)"]
        if col in input_options
    ][:2]

    selected_inputs = st.multiselect(
        "Inputs: Choose at least two numeric columns",
        options=input_options,
        default=default_inputs
    )

    st.info(
        "The model estimates the selected nutrition value from your chosen inputs. "
        "Predictions are estimates and may be inaccurate, so check them against "
        "reliable nutrition information."
    )

    if len(selected_inputs) < 2:
        st.warning("Please select at least two input columns.")
    else:
        model_df = df[selected_inputs + [target_col]].dropna()

        if len(model_df) < 5:
            st.error(
                "There are not enough complete rows to train and evaluate the model."
            )
        else:
            X = model_df[selected_inputs]
            y = model_df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            col1, col2 = st.columns(2)

            col1.metric("Test R²", f"{r2:.3f}")
            col2.metric("Test MAE", f"{mae:.2f}")

            st.caption(
                "R² indicates how well the model explains variation in the test data. "
                "MAE is the average absolute prediction error, in the target column's units."
            )

            st.write("Enter values for your selected inputs:")

            input_values = {}

            for col in selected_inputs:
                min_val = float(model_df[col].min())
                max_val = float(model_df[col].max())
                default_val = float(model_df[col].median())

                input_values[col] = st.number_input(
                    col,
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    key=f"input_{col}"
                )

            if st.button("Predict"):
                input_df = pd.DataFrame(
                    [input_values],
                    columns=selected_inputs
                )

                prediction = model.predict(input_df)[0]

                st.success(
                    f"Predicted {target_col}: {prediction:.2f}"
                )

else:
    st.info("Upload your food nutrition CSV file to get started.")
