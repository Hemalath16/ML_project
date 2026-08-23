import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve
)

from ml_pipeline import train_models


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Water Potability ML",
    page_icon="💧",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
]

TARGET = "Potability"

REQUIRED_COLUMNS = FEATURES + [TARGET]


# ============================================================
# LOAD DEFAULT DATASET
# ============================================================

@st.cache_data
def load_default_data():
    return pd.read_csv(
        "data/water_potability.csv"
    )


# ============================================================
# TRAIN ML MODELS
# ============================================================

@st.cache_data
def run_ml_pipeline(data):
    return train_models(data)


# ============================================================
# LOAD DATASET
# ============================================================

df = load_default_data()

valid_dataset = True


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("💧 Water Potability ML")

st.sidebar.subheader("Dataset")

dataset_option = st.sidebar.radio(
    "Choose Dataset",
    [
        "Default Dataset",
        "Upload New Dataset"
    ]
)


# ============================================================
# UPLOAD DATASET
# ============================================================

if dataset_option == "Upload New Dataset":

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV Dataset",
        type=["csv"]
    )

    if uploaded_file is None:

        valid_dataset = False

        st.sidebar.info(
            "Please upload a CSV file."
        )

    else:

        try:

            uploaded_df = pd.read_csv(
                uploaded_file
            )

            missing_columns = [
                column
                for column in REQUIRED_COLUMNS
                if column not in uploaded_df.columns
            ]

            if missing_columns:

                valid_dataset = False

                st.sidebar.error(
                    "Invalid dataset."
                )

                st.sidebar.write(
                    "Missing columns:"
                )

                for column in missing_columns:
                    st.sidebar.write(
                        f"- {column}"
                    )

            else:

                df = uploaded_df.copy()

                st.sidebar.success(
                    "Dataset accepted!"
                )

        except Exception as error:

            valid_dataset = False

            st.sidebar.error(
                f"Could not read dataset: {error}"
            )


# ============================================================
# SIDEBAR DATASET INFORMATION
# ============================================================

st.sidebar.divider()

st.sidebar.subheader(
    "Current Dataset"
)

if valid_dataset:

    st.sidebar.write(
        f"Rows: **{df.shape[0]}**"
    )

    st.sidebar.write(
        f"Columns: **{df.shape[1]}**"
    )

    st.sidebar.write(
        f"Features: **{len(FEATURES)}**"
    )

    st.sidebar.write(
        f"Missing Values: "
        f"**{int(df.isnull().sum().sum())}**"
    )


# ============================================================
# NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset Analysis",
        "Visualization",
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "KNN",
        "Model Comparison",
        "Water Prediction"
    ]
)


# ============================================================
# TRAIN MODELS
# ============================================================

results = None

if valid_dataset:

    try:

        with st.spinner(
            "Preparing data and training ML models..."
        ):

            results = run_ml_pipeline(df)

    except Exception as error:

        valid_dataset = False

        st.error(
            f"Model training failed: {error}"
        )


# ============================================================
# MODEL RESULT FUNCTION
# ============================================================

def show_model_results(
    model_name,
    model_results,
    show_feature_importance=False
):

    result = model_results[model_name]

    model = result["model"]

    y_test = result["y_test"]

    y_pred = result["y_pred"]

    y_probability = result["y_probability"]

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    st.subheader(
        "Model Performance"
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Accuracy",
        f"{result['accuracy'] * 100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{result['precision'] * 100:.2f}%"
    )

    col3.metric(
        "Recall",
        f"{result['recall'] * 100:.2f}%"
    )

    col4.metric(
        "F1 Score",
        f"{result['f1'] * 100:.2f}%"
    )

    col5.metric(
        "ROC-AUC",
        f"{result['roc_auc'] * 100:.2f}%"
    )

    st.divider()

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader(
        "Confusion Matrix"
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Not Potable",
            "Actual Potable"
        ],
        columns=[
            "Predicted Not Potable",
            "Predicted Potable"
        ]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    st.subheader(
        "Classification Report"
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "Not Potable",
            "Potable"
        ],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    # --------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------

    st.subheader(
        "ROC Curve"
    )

    fpr, tpr, _ = roc_curve(
        y_test,
        y_probability
    )

    roc_df = pd.DataFrame({
        "False Positive Rate": fpr,
        "True Positive Rate": tpr
    })

    roc_fig = px.line(
        roc_df,
        x="False Positive Rate",
        y="True Positive Rate",
        title=f"{model_name} ROC Curve"
    )

    roc_fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(
            dash="dash"
        )
    )

    st.plotly_chart(
        roc_fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    if show_feature_importance:

        st.subheader(
            "Feature Importance"
        )

        importance_df = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": model.feature_importances_
        }).sort_values(
            "Importance",
            ascending=False
        )

        importance_fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"{model_name} Feature Importance"
        )

        st.plotly_chart(
            importance_fig,
            use_container_width=True
        )


# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.title(
        "💧 Water Potability Prediction"
    )

    st.subheader(
        "Machine Learning Based Water Quality Analysis"
    )

    st.write(
        "Analyze water-quality parameters and predict "
        "whether water is potable using machine learning."
    )

    st.divider()

    if valid_dataset:

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Dataset Rows",
            df.shape[0]
        )

        col2.metric(
            "Features",
            len(FEATURES)
        )

        col3.metric(
            "Target",
            "Potability"
        )

        col4.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

        st.divider()

        st.subheader(
            "Available ML Models"
        )

        model_table = pd.DataFrame({
            "Model": [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "KNN"
            ],
            "Status": [
                "Ready",
                "Ready",
                "Ready",
                "Ready"
            ]
        })

        st.dataframe(
            model_table,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "Use the sidebar to explore the dataset, "
            "visualizations, model results, comparison, "
            "and live water prediction."
        )

    else:

        st.warning(
            "Please upload a valid water potability dataset."
        )


# ============================================================
# DATASET ANALYSIS
# ============================================================

elif page == "Dataset Analysis":

    st.title(
        "📊 Dataset Analysis"
    )

    if not valid_dataset:

        st.warning(
            "Please upload a valid dataset."
        )

    else:

        st.subheader(
            "Dataset Preview"
        )

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        st.subheader(
            "Dataset Information"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Rows",
            df.shape[0]
        )

        col2.metric(
            "Columns",
            df.shape[1]
        )

        col3.metric(
            "Features",
            len(FEATURES)
        )

        col4.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

        st.subheader(
            "Column Data Types"
        )

        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str)
        })

        st.dataframe(
            dtype_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "Missing Values"
        )

        missing_df = pd.DataFrame({
            "Column": df.columns,
            "Missing Values": df.isnull().sum().values
        })

        st.dataframe(
            missing_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "Duplicate Rows"
        )

        duplicate_count = int(
            df.duplicated().sum()
        )

        if duplicate_count == 0:

            st.success(
                "No duplicate rows found."
            )

        else:

            st.warning(
                f"{duplicate_count} duplicate rows found."
            )

        st.subheader(
            "Statistical Summary"
        )

        st.dataframe(
            df.describe(),
            use_container_width=True
        )

        st.subheader(
            "Potability Distribution"
        )

        potability_counts = (
            df[TARGET]
            .value_counts()
            .sort_index()
        )

        target_df = pd.DataFrame({
            "Status": [
                "Not Potable",
                "Potable"
            ],
            "Count": [
                potability_counts.get(0, 0),
                potability_counts.get(1, 0)
            ]
        })

        st.dataframe(
            target_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# VISUALIZATION
# ============================================================

elif page == "Visualization":

    st.title(
        "📈 Exploratory Data Analysis"
    )

    if not valid_dataset:

        st.warning(
            "Please upload a valid dataset."
        )

    else:

        st.write(
            "Interactive graphs generated from the selected dataset."
        )

        st.subheader(
            "1. Water Potability Distribution"
        )

        potability_counts = (
            df[TARGET]
            .value_counts()
            .sort_index()
        )

        target_df = pd.DataFrame({
            "Status": [
                "Not Potable",
                "Potable"
            ],
            "Count": [
                potability_counts.get(0, 0),
                potability_counts.get(1, 0)
            ]
        })

        fig1 = px.bar(
            target_df,
            x="Status",
            y="Count",
            title="Potable vs Not Potable Water"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

        st.subheader(
            "2. Feature Distribution"
        )

        selected_feature = st.selectbox(
            "Select Feature",
            FEATURES
        )

        fig2 = px.histogram(
            df,
            x=selected_feature,
            nbins=30,
            title=f"Distribution of {selected_feature}"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        st.subheader(
            "3. Feature vs Potability"
        )

        comparison_feature = st.selectbox(
            "Select Feature",
            FEATURES,
            key="comparison_feature"
        )

        comparison_df = df.copy()

        comparison_df[
            "Water Status"
        ] = comparison_df[TARGET].map({
            0: "Not Potable",
            1: "Potable"
        })

        fig3 = px.box(
            comparison_df,
            x="Water Status",
            y=comparison_feature,
            color="Water Status",
            title=(
                f"{comparison_feature} "
                "vs Water Potability"
            )
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

        st.subheader(
            "4. Correlation Heatmap"
        )

        correlation = df.corr(
            numeric_only=True
        )

        fig4 = ff.create_annotated_heatmap(
            z=correlation.values,
            x=list(correlation.columns),
            y=list(correlation.columns),
            annotation_text=correlation.round(2).values,
            colorscale="Blues",
            showscale=True
        )

        fig4.update_layout(
            title="Feature Correlation Matrix",
            height=700
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )

        st.subheader(
            "5. Missing Values by Feature"
        )

        missing_plot_df = pd.DataFrame({
            "Feature": df.columns,
            "Missing Values": df.isnull().sum().values
        })

        missing_plot_df = missing_plot_df[
            missing_plot_df["Missing Values"] > 0
        ]

        if not missing_plot_df.empty:

            fig5 = px.bar(
                missing_plot_df,
                x="Feature",
                y="Missing Values",
                title="Missing Values"
            )

            st.plotly_chart(
                fig5,
                use_container_width=True
            )

        else:

            st.success(
                "No missing values found."
            )


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

elif page == "Logistic Regression":

    st.title(
        "🤖 Logistic Regression"
    )

    if results is not None:

        show_model_results(
            "Logistic Regression",
            results
        )

    else:

        st.warning(
            "Model results are not available."
        )


# ============================================================
# DECISION TREE
# ============================================================

elif page == "Decision Tree":

    st.title(
        "🌳 Decision Tree"
    )

    if results is not None:

        show_model_results(
            "Decision Tree",
            results,
            show_feature_importance=True
        )

    else:

        st.warning(
            "Model results are not available."
        )


# ============================================================
# RANDOM FOREST
# ============================================================

elif page == "Random Forest":

    st.title(
        "🌲 Random Forest"
    )

    if results is not None:

        show_model_results(
            "Random Forest",
            results,
            show_feature_importance=True
        )

    else:

        st.warning(
            "Model results are not available."
        )


# ============================================================
# KNN
# ============================================================

elif page == "KNN":

    st.title(
        "📍 K-Nearest Neighbors"
    )

    if results is not None:

        show_model_results(
            "KNN",
            results
        )

    else:

        st.warning(
            "Model results are not available."
        )


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    st.title(
        "🏆 ML Model Comparison"
    )

    if results is None:

        st.warning(
            "Model results are not available."
        )

    else:

        model_names = [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "KNN"
        ]

        comparison_data = []

        for model_name in model_names:

            result = results[model_name]

            comparison_data.append({
                "Model": model_name,
                "Accuracy": result["accuracy"] * 100,
                "Precision": result["precision"] * 100,
                "Recall": result["recall"] * 100,
                "F1 Score": result["f1"] * 100,
                "ROC-AUC": result["roc_auc"] * 100
            })

        comparison_df = pd.DataFrame(
            comparison_data
        )

        st.subheader(
            "Performance Comparison"
        )

        st.dataframe(
            comparison_df.round(2),
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "Accuracy Comparison"
        )

        accuracy_fig = px.bar(
            comparison_df,
            x="Model",
            y="Accuracy",
            text="Accuracy",
            title="Model Accuracy"
        )

        accuracy_fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        st.plotly_chart(
            accuracy_fig,
            use_container_width=True
        )

        st.subheader(
            "All Metrics Comparison"
        )

        metrics_long = comparison_df.melt(
            id_vars="Model",
            value_vars=[
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC"
            ],
            var_name="Metric",
            value_name="Score"
        )

        all_metrics_fig = px.bar(
            metrics_long,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            title="Comparison of All Metrics"
        )

        all_metrics_fig.update_layout(
            yaxis_title="Score (%)",
            yaxis_range=[0, 100]
        )

        st.plotly_chart(
            all_metrics_fig,
            use_container_width=True
        )

        st.subheader(
            "Model Ranking"
        )

        ranking_df = comparison_df.copy()

        ranking_df["Average Score"] = (
            ranking_df[
                [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "ROC-AUC"
                ]
            ].mean(axis=1)
        )

        ranking_df = ranking_df.sort_values(
            "Average Score",
            ascending=False
        )

        ranking_df.insert(
            0,
            "Rank",
            range(1, len(ranking_df) + 1)
        )

        st.dataframe(
            ranking_df.round(2),
            use_container_width=True,
            hide_index=True
        )

        best_model = ranking_df.iloc[0]

        st.success(
            f"🏆 Best overall model: "
            f"**{best_model['Model']}** "
            f"with an average score of "
            f"**{best_model['Average Score']:.2f}%**."
        )


# ============================================================
# WATER PREDICTION
# ============================================================

elif page == "Water Prediction":

    st.title(
        "🧪 Water Potability Prediction"
    )

    st.write(
        "Enter the nine water-quality parameters "
        "to test a new water sample."
    )

    if results is None:

        st.warning(
            "ML models are not available."
        )

    else:

        st.divider()

        st.subheader(
            "Enter Water Quality Values"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            ph = st.number_input(
                "pH",
                min_value=0.0,
                max_value=14.0,
                value=7.0,
                step=0.1
            )

            hardness = st.number_input(
                "Hardness",
                min_value=0.0,
                value=200.0,
                step=1.0
            )

            solids = st.number_input(
                "Solids",
                min_value=0.0,
                value=20000.0,
                step=100.0
            )

        with col2:

            chloramines = st.number_input(
                "Chloramines",
                min_value=0.0,
                value=7.0,
                step=0.1
            )

            sulfate = st.number_input(
                "Sulfate",
                min_value=0.0,
                value=300.0,
                step=1.0
            )

            conductivity = st.number_input(
                "Conductivity",
                min_value=0.0,
                value=400.0,
                step=1.0
            )

        with col3:

            organic_carbon = st.number_input(
                "Organic Carbon",
                min_value=0.0,
                value=10.0,
                step=0.1
            )

            trihalomethanes = st.number_input(
                "Trihalomethanes",
                min_value=0.0,
                value=60.0,
                step=1.0
            )

            turbidity = st.number_input(
                "Turbidity",
                min_value=0.0,
                value=4.0,
                step=0.1
            )

        st.divider()

        st.subheader(
            "Select Machine Learning Model"
        )

        selected_model = st.selectbox(
            "Choose Model",
            [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "KNN",
                "All Models"
            ]
        )

        predict_button = st.button(
            "🔍 Predict Water Potability",
            type="primary",
            use_container_width=True
        )

        if predict_button:

            user_data = pd.DataFrame(
                [[
                    ph,
                    hardness,
                    solids,
                    chloramines,
                    sulfate,
                    conductivity,
                    organic_carbon,
                    trihalomethanes,
                    turbidity
                ]],
                columns=FEATURES
            )

            # =================================================
            # SINGLE MODEL PREDICTION
            # =================================================

            if selected_model != "All Models":

                result = results[selected_model]

                model = result["model"]
                imputer = result["imputer"]
                scaler = result["scaler"]

                user_imputed = imputer.transform(
                    user_data
                )

                user_scaled = scaler.transform(
                    user_imputed
                )

                prediction = model.predict(
                    user_scaled
                )[0]

                probabilities = model.predict_proba(
                    user_scaled
                )[0]

                potable_probability = probabilities[1]

                st.divider()

                st.subheader(
                    "Prediction Result"
                )

                if prediction == 1:

                    st.success(
                        "💧 WATER IS POTABLE"
                    )

                else:

                    st.error(
                        "⚠️ WATER IS NOT POTABLE"
                    )

                result_col1, result_col2 = st.columns(2)

                result_col1.metric(
                    "Prediction",
                    "Potable"
                    if prediction == 1
                    else "Not Potable"
                )

                result_col2.metric(
                    "Potability Probability",
                    f"{potable_probability * 100:.2f}%"
                )

                st.progress(
                    float(potable_probability)
                )

            # =================================================
            # ALL MODELS
            # =================================================

            else:

                prediction_results = []

                for model_name in [
                    "Logistic Regression",
                    "Decision Tree",
                    "Random Forest",
                    "KNN"
                ]:

                    result = results[model_name]

                    model = result["model"]
                    imputer = result["imputer"]
                    scaler = result["scaler"]

                    user_imputed = imputer.transform(
                        user_data
                    )

                    user_scaled = scaler.transform(
                        user_imputed
                    )

                    prediction = model.predict(
                        user_scaled
                    )[0]

                    probabilities = model.predict_proba(
                        user_scaled
                    )[0]

                    potable_probability = probabilities[1]

                    prediction_results.append({
                        "Model": model_name,
                        "Prediction": (
                            "Potable"
                            if prediction == 1
                            else "Not Potable"
                        ),
                        "Potability Probability (%)":
                            round(
                                potable_probability * 100,
                                2
                            )
                    })

                prediction_df = pd.DataFrame(
                    prediction_results
                )

                st.divider()

                st.subheader(
                    "Prediction Results from All Models"
                )

                st.dataframe(
                    prediction_df,
                    use_container_width=True,
                    hide_index=True
                )

                # ---------------------------------------------
                # VOTE
                # ---------------------------------------------

                potable_count = (
                    prediction_df[
                        "Prediction"
                    ] == "Potable"
                ).sum()

                total_models = len(
                    prediction_df
                )

                not_potable_count = (
                    total_models - potable_count
                )

                st.divider()

                st.subheader(
                    "Overall Prediction"
                )

                vote_col1, vote_col2 = st.columns(2)

                vote_col1.metric(
                    "Potable Votes",
                    f"{potable_count}/{total_models}"
                )

                vote_col2.metric(
                    "Not Potable Votes",
                    f"{not_potable_count}/{total_models}"
                )

                if potable_count > total_models / 2:

                    st.success(
                        f"💧 Majority Prediction: "
                        f"POTABLE "
                        f"({potable_count}/{total_models} models)"
                    )

                elif not_potable_count > total_models / 2:

                    st.error(
                        f"⚠️ Majority Prediction: "
                        f"NOT POTABLE "
                        f"({not_potable_count}/{total_models} models)"
                    )

                else:

                    st.warning(
                        "⚖️ No majority prediction."
                    )