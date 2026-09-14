import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

def train_decision_tree(input_file="cleaned_students_performance.csv", plots_dir="plots"):
    print("=" * 60)
    print("STEP 3: MACHINE LEARNING - DECISION TREE CLASSIFICATION")
    print("=" * 60)
    
    if not os.path.exists(input_file):
        print(f"File '{input_file}' not found. Running data cleaning first...")
        from data_cleaning import clean_data
        df = clean_data(output_file=input_file)
    else:
        df = pd.read_csv(input_file)
        
    os.makedirs(plots_dir, exist_ok=True)
    
    # Feature Selection & Encoding
    feature_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
    X_raw = df[feature_cols]
    
    # Target 1: Performance Level ('High', 'Medium', 'Low')
    y_perf = df['performance_level']
    
    # Target 2: Overall Status ('Pass', 'Fail')
    y_status = df['overall_status']
    
    # One-Hot Encoding for categorical features
    X = pd.get_dummies(X_raw, drop_first=True)
    print(f"Features encoded successfully. Design Matrix Shape: {X.shape}")
    
    # ----------------------------------------------------
    # Model 1: Multi-Class Decision Tree (Performance Level)
    # ----------------------------------------------------
    print("\n--- Training Decision Tree Classifier (Predicting Performance Level: High / Medium / Low) ---")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_perf, test_size=0.20, random_state=42, stratify=y_perf
    )
    
    print(f"Train Set Size: {len(X_train)} samples | Test Set Size: {len(X_test)} samples")
    
    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 4, 5, 6, 8],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [2, 4, 6]
    }
    
    dt_base = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(dt_base, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_dt = grid_search.best_estimator_
    print(f"Best Hyperparameters found: {grid_search.best_params_}")
    
    # Predictions & Evaluation
    y_pred = best_dt.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print("\n--- Decision Tree Performance Metrics ---")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # ----------------------------------------------------
    # Model 2: Benchmark Comparison with Random Forest
    # ----------------------------------------------------
    print("--- Benchmark: Random Forest Classifier ---")
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, average='weighted', zero_division=0)
    print(f"Random Forest Accuracy: {acc_rf * 100:.2f}% | F1-Score: {f1_rf * 100:.2f}%")
    
    # ----------------------------------------------------
    # Model 3: Binary Classification (Overall Pass / Fail)
    # ----------------------------------------------------
    print("\n--- Training Decision Tree Classifier (Predicting Overall Pass / Fail) ---")
    X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
        X, y_status, test_size=0.20, random_state=42, stratify=y_status
    )
    dt_status = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_status.fit(X_tr_s, y_tr_s)
    y_pred_status = dt_status.predict(X_te_s)
    
    acc_status = accuracy_score(y_te_s, y_pred_status)
    print(f"Pass/Fail Classification Accuracy: {acc_status * 100:.2f}%")
    
    # ----------------------------------------------------
    # Visualizations
    # ----------------------------------------------------
    print(f"\nGenerating model visualization plots in '{plots_dir}/'...")
    
    # Plot 7: Decision Tree Structure
    plt.figure(figsize=(20, 10))
    plot_tree(
        best_dt, 
        feature_names=X.columns.tolist(), 
        class_names=best_dt.classes_.astype(str).tolist(), 
        filled=True, 
        rounded=True, 
        fontsize=10
    )
    plt.title("Tuned Decision Tree Model Structure", fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "7_decision_tree_structure.png"), dpi=300)
    plt.close()
    
    # Plot 8: Confusion Matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_test, y_pred, labels=best_dt.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_dt.classes_)
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.title("Confusion Matrix - Performance Level Decision Tree")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "8_confusion_matrix.png"), dpi=300)
    plt.close()
    
    # Plot 9: Feature Importance Bar Plot
    importances = pd.Series(best_dt.feature_importances_, index=X.columns).sort_values(ascending=True)
    plt.figure(figsize=(10, 6))
    importances.plot(kind='barh', color='#2980b9')
    plt.title("Feature Importance in Decision Tree Model", fontsize=14)
    plt.xlabel("Gini Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "9_feature_importances.png"), dpi=300)
    plt.close()
    
    print("=" * 60)
    print("DECISION TREE MODELING COMPLETED SUCCESSFULLY")
    print("=" * 60 + "\n")
    
    return best_dt, X_test, y_test

if __name__ == "__main__":
    train_decision_tree()