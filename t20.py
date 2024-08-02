import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Streamlit app
st.title('T20 Win Prediction Model Tuning')

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Drop 'Date' and other features related to 'Date'
    df = df.drop(columns=['Date'])
    
    # Display data and column names
    st.write("Dataset Overview:")
    st.write(df.head())
    st.write("Columns:", df.columns.tolist())

    # Encode categorical features
    encoder = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), ['Venue', 'Bat First', 'Bat Second'])
        ],
        remainder='passthrough'
    )

    X = encoder.fit_transform(df.drop(columns=['Winner']))
    y = df['Winner']

    # Encode target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define pipelines for each model with scaling
    pipelines = {
        'SVM': Pipeline([
            ('scaler', StandardScaler(with_mean=False)),  # with_mean=False for sparse matrices
            ('model', SVC(gamma='auto'))
        ]),
        'RandomForest': Pipeline([
            ('scaler', StandardScaler(with_mean=False)),  # with_mean=False for sparse matrices
            ('model', RandomForestClassifier(random_state=42))
        ]),
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler(with_mean=False)),  # with_mean=False for sparse matrices
            ('model', LogisticRegression(solver='liblinear', multi_class='auto'))
        ]),
        'DecisionTree': Pipeline([
            ('scaler', StandardScaler(with_mean=False)),  # with_mean=False for sparse matrices
            ('model', DecisionTreeClassifier())
        ])
    }

    # Define parameter grids for GridSearchCV
    param_grids = {
        'SVM': {
            'model__C': [1, 10, 20],
            'model__kernel': ['linear', 'rbf']
        },
        'RandomForest': {
            'model__n_estimators': [1, 5, 10]
        },
        'LogisticRegression': {
            'model__C': [1, 5, 10]
        },
        'DecisionTree': {
            'model__criterion': ['gini', 'entropy'],
            'model__splitter': ['best', 'random'],
            'model__max_depth': [None, 10, 20, 30, 40]
        }
    }

    # Define StratifiedKFold
    cv = StratifiedKFold(n_splits=5)

    # GridSearchCV using pipelines and StratifiedKFold
    scores = []
    for model_name, pipe in pipelines.items():
        param_grid = param_grids[model_name]
        clf = GridSearchCV(pipe, param_grid, cv=cv, return_train_score=False)
        clf.fit(X_train, y_train)
        scores.append({
            'model': model_name,
            'best_score': clf.best_score_,
            'best_params': clf.best_params_
        })

    df_results = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
    st.write("GridSearchCV Results:")
    st.write(df_results)

    # Display the best model's accuracy on the training and test sets
    best_model_name = df_results.loc[df_results['best_score'].idxmax(), 'model']
    best_model = pipelines[best_model_name]
    best_model.set_params(**df_results.loc[df_results['best_score'].idxmax(), 'best_params'])
    
    # Fit the best model on the training data
    best_model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    # Accuracy scores
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    st.write(f"Best Model: {best_model_name}")
    st.write(f"Training Accuracy: {train_accuracy:.2f}")
    st.write(f"Test Accuracy: {test_accuracy:.2f}")
else:
    st.write("Please upload a CSV file to get started.")
