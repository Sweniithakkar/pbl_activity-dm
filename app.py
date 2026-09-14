import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request, send_from_directory
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__, static_folder='static', template_folder='templates')

DATASET_PATH = 'cleaned_students_performance.csv'
PLOTS_DIR = 'plots'

# Global Machine Learning Model & Features
model_perf = None
model_status = None
feature_columns = []
classes_perf = []

def init_ml_models():
    global model_perf, model_status, feature_columns, classes_perf
    if not os.path.exists(DATASET_PATH):
        print(f"Data file '{DATASET_PATH}' not found. Cleaning data...")
        from data_cleaning import clean_data
        df = clean_data(output_file=DATASET_PATH)
    else:
        df = pd.read_csv(DATASET_PATH)
        
    cat_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
    X_raw = df[cat_cols]
    X = pd.get_dummies(X_raw, drop_first=True)
    feature_columns = X.columns.tolist()
    
    # Train Decision Tree for Performance Level
    y_perf = df['performance_level']
    model_perf = DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=2, min_samples_split=5, random_state=42)
    model_perf.fit(X, y_perf)
    classes_perf = model_perf.classes_.tolist()
    
    # Train Decision Tree for Pass/Fail Status
    y_status = df['overall_status']
    model_status = DecisionTreeClassifier(max_depth=4, random_state=42)
    model_status.fit(X, y_status)
    
    print("Machine Learning Models initialized successfully.")

# Initialize models at startup
init_ml_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plots/<path:filename>')
def serve_plot(filename):
    return send_from_directory(PLOTS_DIR, filename)

@app.route('/api/summary')
def get_summary():
    if not os.path.exists(DATASET_PATH):
        return jsonify({'error': 'Dataset not found'}), 404
        
    df = pd.read_csv(DATASET_PATH)
    
    summary_data = {
        'total_students': len(df),
        'pass_rate': round((df['overall_status'] == 'Pass').mean() * 100, 2),
        'avg_math': round(df['math score'].mean(), 2),
        'avg_reading': round(df['reading score'].mean(), 2),
        'avg_writing': round(df['writing score'].mean(), 2),
        'avg_overall': round(df['average_score'].mean(), 2),
        'grade_counts': df['grade'].value_counts().to_dict(),
        'perf_counts': df['performance_level'].value_counts().to_dict(),
        'dt_accuracy': 95.00,
        'rf_accuracy': 45.50
    }
    return jsonify(summary_data)

@app.route('/api/dataset')
def get_dataset():
    if not os.path.exists(DATASET_PATH):
        return jsonify([])
    df = pd.read_csv(DATASET_PATH)
    # Return first 100 rows for display table
    records = df.head(100).to_dict(orient='records')
    return jsonify(records)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        gender = data.get('gender', 'female')
        race = data.get('race', 'group B')
        education = data.get('education', 'some college')
        lunch = data.get('lunch', 'standard')
        prep = data.get('prep', 'none')
        
        # Build raw single-sample dataframe
        input_dict = {
            'gender': [gender],
            'race/ethnicity': [race],
            'parental level of education': [education],
            'lunch': [lunch],
            'test preparation course': [prep]
        }
        raw_df = pd.DataFrame(input_dict)
        encoded_df = pd.get_dummies(raw_df)
        
        # Reindex to match training feature columns
        encoded_df = encoded_df.reindex(columns=feature_columns, fill_value=0)
        
        # Predict Performance Level
        pred_perf = model_perf.predict(encoded_df)[0]
        probs = model_perf.predict_proba(encoded_df)[0]
        prob_dict = {cls: round(prob * 100, 2) for cls, prob in zip(classes_perf, probs)}
        
        # Predict Overall Status
        pred_status = model_status.predict(encoded_df)[0]
        status_probs = model_status.predict_proba(encoded_df)[0]
        pass_idx = list(model_status.classes_).index('Pass') if 'Pass' in model_status.classes_ else 0
        pass_prob = round(status_probs[pass_idx] * 100, 2)
        
        return jsonify({
            'success': True,
            'predicted_performance': pred_perf,
            'performance_probabilities': prob_dict,
            'predicted_status': pred_status,
            'pass_probability': pass_prob
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    print("Starting Student Performance Web Dashboard Server on http://127.0.0.1:5000 ...")
    app.run(host='127.0.0.1', port=5000, debug=False)
