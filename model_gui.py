import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, davies_bouldin_score
)

def model_page():
    st.set_page_config(page_title="Machine Learning", layout="wide")
    st.title("Machine Learning")

def prepare_data(df):
    problem_type = st.radio("Select problem type", ["Classification", "Regression", "Clustering"])

    if problem_type in ["Classification", "Regression"]:
        
        target_col = st.selectbox("Select target variable", df.columns)
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        if problem_type == "Classification":
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        
        test_size = st.slider("Test size ratio", 0.1, 0.5, 0.2, 0.05)
        random_state = st.slider("Random state", 0, 100, 42)
        
        try:
            if problem_type == "Classification":
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, 
                    test_size=test_size, 
                    random_state=random_state,
                    stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=test_size, 
                    random_state=random_state
                )
        except ValueError as e:
            st.error(f"Error in train-test split: {str(e)}")
            st.stop()

    if problem_type == "Classification":
        st.subheader("Classification Models")
        model_name = st.selectbox("Select model", 
                                ["SVM", "Random Forest", "Naive Bayes", 
                                "Logistic Regression", "KNN", "Decision Tree"])
        
        model_params = {}
        if model_name == "SVM":
            model_params['C'] = st.slider("Regularization parameter (C)", 0.01, 10.0, 1.0)
            model_params['kernel'] = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
            model = SVC(**model_params, probability=True)
        
        elif model_name == "Random Forest":
            model_params['n_estimators'] = st.slider("Number of trees", 10, 200, 100)
            model_params['max_depth'] = st.slider("Max depth", 1, 20, 5)
            model = RandomForestClassifier(**model_params, random_state=42)
        
        elif model_name == "Naive Bayes":
            model = GaussianNB()
        
        elif model_name == "Logistic Regression":
            model_params['C'] = st.slider("Inverse of regularization strength (C)", 
                                        0.01, 10.0, 1.0)
            model_params['penalty'] = st.selectbox("Penalty", ["l2", "none"])
            model = LogisticRegression(**model_params, random_state=42, max_iter=1000)
        
        elif model_name == "KNN":
            model_params['n_neighbors'] = st.slider("Number of neighbors", 1, 20, 5)
            model_params['weights'] = st.selectbox("Weights", ["uniform", "distance"])
            model = KNeighborsClassifier(**model_params)
        
        elif model_name == "Decision Tree":
            model_params['max_depth'] = st.slider("Max depth", 1, 20, 5)
            model_params['criterion'] = st.selectbox("Criterion", ["gini", "entropy"])
            model = DecisionTreeClassifier(**model_params, random_state=42)

    elif problem_type == "Regression":
        st.subheader("Regression Models")
        model_name = st.selectbox("Select model", 
                                ["Linear Regression", "SVR", "Random Forest", 
                                "KNN", "Decision Tree"])
        
        model_params = {}
        if model_name == "Linear Regression":
            model = LinearRegression()
        
        elif model_name == "SVR":
            model_params['C'] = st.slider("Regularization parameter (C)", 0.01, 10.0, 1.0)
            model_params['kernel'] = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
            model = SVR(**model_params)
        
        elif model_name == "Random Forest":
            model_params['n_estimators'] = st.slider("Number of trees", 10, 200, 100)
            model_params['max_depth'] = st.slider("Max depth", 1, 20, 5)
            model = RandomForestRegressor(**model_params, random_state=42)
        
        elif model_name == "KNN":
            model_params['n_neighbors'] = st.slider("Number of neighbors", 1, 20, 5)
            model_params['weights'] = st.selectbox("Weights", ["uniform", "distance"])
            model = KNeighborsRegressor(**model_params)
        
        elif model_name == "Decision Tree":
            model_params['max_depth'] = st.slider("Max depth", 1, 20, 5)
            model_params['criterion'] = st.selectbox("Criterion", ["squared_error", "friedman_mse"])
            model = DecisionTreeRegressor(**model_params, random_state=42)

    elif problem_type == "Clustering":
        st.subheader("Clustering Models")
        model_name = st.selectbox("Select model", 
                                ["K-Means", "DBSCAN", "Agglomerative"])
        
        model_params = {}
        if model_name == "K-Means":
            model_params['n_clusters'] = st.slider("Number of clusters", 2, 10, 3)
            model = KMeans(**model_params, random_state=42)
        
        elif model_name == "DBSCAN":
            model_params['eps'] = st.slider("Epsilon (eps)", 0.1, 2.0, 0.5, 0.1)
            model_params['min_samples'] = st.slider("Minimum samples", 1, 20, 5)
            model = DBSCAN(**model_params)
        
        elif model_name == "Agglomerative":
            model_params['n_clusters'] = st.slider("Number of clusters", 2, 10, 3)
            model_params['linkage'] = st.selectbox("Linkage", ["ward", "complete", "average", "single"])
            model = AgglomerativeClustering(**model_params)

    if problem_type in ["Classification", "Regression"]:
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                model_pipeline.fit(X_train, y_train)
                
                if problem_type == "Classification":
                    y_pred = model_pipeline.predict(X_test)
                    y_prob = model_pipeline.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                    y_test_original = le.inverse_transform(y_test)
                    y_pred_original = le.inverse_transform(y_pred)
                    
                    st.subheader("Classification Metrics")
                    metrics_df = pd.DataFrame({
                        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                        "Value": [
                            accuracy_score(y_test_original, y_pred_original),
                            precision_score(y_test_original, y_pred_original, average='weighted', zero_division=0),
                            recall_score(y_test_original, y_pred_original, average='weighted', zero_division=0),
                            f1_score(y_test_original, y_pred_original, average='weighted', zero_division=0)
                        ]
                    })
                    st.dataframe(metrics_df.style.format({"Value": "{:.4f}"}))
                    
                    if y_prob is not None and len(np.unique(y_test)) > 1:
                        try:
                            roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
                            st.write(f"ROC AUC: {roc_auc:.4f}")
                        except Exception as e:
                            st.warning(f"Could not calculate ROC AUC: {str(e)}")

                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test_original, y_pred_original)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=le.classes_,
                            yticklabels=le.classes_)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)
                    
                    st.subheader("Classification Report")
                    report = classification_report(y_test_original, y_pred_original, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format("{:.4f}"))
                
                elif problem_type == "Regression":
                    y_pred = model_pipeline.predict(X_test)
                    
                    st.subheader("Regression Metrics")
                    metrics_df = pd.DataFrame({
                        "Metric": ["MSE", "RMSE", "MAE", "R²"],
                        "Value": [
                            mean_squared_error(y_test, y_pred),
                            np.sqrt(mean_squared_error(y_test, y_pred)),
                            mean_absolute_error(y_test, y_pred),
                            r2_score(y_test, y_pred)
                        ]
                    })
                    st.dataframe(metrics_df.style.format({"Value": "{:.4f}"}))
                    
                    st.subheader("Regression Plot")
                    fig, ax = plt.subplots()
                    ax.scatter(y_test, y_pred, alpha=0.5)
                    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
                    ax.set_xlabel('Actual')
                    ax.set_ylabel('Predicted')
                    st.pyplot(fig)

    elif problem_type == "Clustering":
        if st.button("Apply Clustering"):
            with st.spinner("Clustering data..."):
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), numeric_features),
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                    ])
                
                X_processed = preprocessor.fit_transform(X)
                
                clusters = model.fit_predict(X_processed)
                
                df_clustered = df.copy()
                df_clustered['Cluster'] = clusters
                
                st.subheader("Clustering Results")
                
                if model_name != "DBSCAN" or len(np.unique(clusters)) > 1:
                    try:
                        silhouette = silhouette_score(X_processed, clusters)
                        davies_bouldin = davies_bouldin_score(X_processed, clusters)
                        
                        metrics_df = pd.DataFrame({
                            "Metric": ["Silhouette Score", "Davies-Bouldin Index"],
                            "Value": [silhouette, davies_bouldin]
                        })
                        st.dataframe(metrics_df.style.format({"Value": "{:.4f}"}))
                    except:
                        st.warning("Could not calculate clustering metrics for this configuration")
                
                st.subheader("Cluster Distribution")
                cluster_counts = pd.Series(clusters).value_counts().sort_index()
                st.bar_chart(cluster_counts)
                
                st.subheader("Clustered Data Preview")
                st.dataframe(df_clustered.head())
                
                if st.button("Download Clustered Data"):
                    csv = df_clustered.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="clustered_data.csv",
                        mime="text/csv"
                    )