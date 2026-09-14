import os
import pandas as pd
import numpy as np

def clean_data(input_file="StudentsPerformance.csv", output_file="cleaned_students_performance.csv"):
    print("=" * 60)
    print("STEP 1: DATA CLEANING & FEATURE ENGINEERING")
    print("=" * 60)
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")
        
    df = pd.read_csv(input_file)
    print(f"Loaded dataset successfully with shape: {df.shape}")
    
    # Check for missing values
    missing_vals = df.isnull().sum().sum()
    print(f"Missing values found: {missing_vals}")
    
    # Check for duplicates
    duplicate_rows = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicate_rows}")
    if duplicate_rows > 0:
        df = df.drop_duplicates()
        print(f"Duplicates dropped. New shape: {df.shape}")
        
    # Feature Engineering
    # 1. Total and Average Score
    df['total_score'] = df['math score'] + df['reading score'] + df['writing score']
    df['average_score'] = np.round(df['total_score'] / 3.0, 2)
    
    # 2. Individual Subject Pass/Fail (Cutoff = 40)
    df['pass_math'] = df['math score'] >= 40
    df['pass_reading'] = df['reading score'] >= 40
    df['pass_writing'] = df['writing score'] >= 40
    
    # 3. Overall Status
    df['overall_status'] = np.where(
        (df['pass_math']) & (df['pass_reading']) & (df['pass_writing']), 
        'Pass', 
        'Fail'
    )
    
    # 4. Grading System
    def assign_grade(score):
        if score >= 90:
            return 'A+'
        elif score >= 80:
            return 'A'
        elif score >= 70:
            return 'B'
        elif score >= 60:
            return 'C'
        elif score >= 50:
            return 'D'
        else:
            return 'F'
            
    df['grade'] = df['average_score'].apply(assign_grade)
    
    # 5. Performance Level (Target variable for ML)
    def assign_performance_level(score):
        if score >= 75:
            return 'High'
        elif score >= 55:
            return 'Medium'
        else:
            return 'Low'
            
    df['performance_level'] = df['average_score'].apply(assign_performance_level)
    
    # Save cleaned dataframe
    df.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved to: {os.path.abspath(output_file)}")
    
    # Summary report
    print("\n--- Summary Statistics ---")
    print(f"Total Students: {len(df)}")
    print(f"Overall Pass Rate: {(df['overall_status'] == 'Pass').mean() * 100:.2f}%")
    print("\nGrade Breakdown:")
    print(df['grade'].value_counts().sort_index())
    print("\nPerformance Level Breakdown:")
    print(df['performance_level'].value_counts())
    print("=" * 60 + "\n")
    
    return df

if __name__ == "__main__":
    clean_data()