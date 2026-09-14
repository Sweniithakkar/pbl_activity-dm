import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Import mlxtend for Association Rule Mining
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False

def run_analysis(input_file="cleaned_students_performance.csv", plots_dir="plots"):
    print("=" * 60)
    print("STEP 2: EXPLORATORY DATA ANALYSIS & ASSOCIATION MINING")
    print("=" * 60)
    
    if not os.path.exists(input_file):
        print(f"File '{input_file}' not found. Running data cleaning first...")
        from data_cleaning import clean_data
        df = clean_data(output_file=input_file)
    else:
        df = pd.read_csv(input_file)
        
    os.makedirs(plots_dir, exist_ok=True)
    
    # Visual Styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    sns.set_palette("muted")
    plt.rcParams.update({
        'font.sans-serif': 'Segoe UI',
        'font.family': 'sans-serif',
        'figure.titlesize': 16,
        'axes.titlesize': 14,
        'axes.labelsize': 12
    })
    
    # 1. Descriptive Statistics
    print("\n--- Numerical Summary Statistics ---")
    score_cols = ['math score', 'reading score', 'writing score', 'average_score']
    desc_stats = df[score_cols].describe().round(2)
    print(desc_stats)
    
    print("\n--- Average Scores by Gender ---")
    print(df.groupby('gender')[score_cols].mean().round(2))
    
    print("\n--- Average Scores by Test Preparation Course ---")
    print(df.groupby('test preparation course')[score_cols].mean().round(2))
    
    print("\n--- Average Scores by Lunch Type (Socioeconomic Indicator) ---")
    print(df.groupby('lunch')[score_cols].mean().round(2))
    
    print("\n--- Average Scores by Parental Level of Education ---")
    print(df.groupby('parental level of education')[score_cols].mean().round(2).sort_values(by='average_score', ascending=False))
    
    # 2. Plot Generation
    print(f"\nGenerating visual analysis plots in '{plots_dir}/'...")
    
    # Plot 1: Score Distributions
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    subjects = [('math score', 'Math Score', '#3498db'), 
                ('reading score', 'Reading Score', '#2ecc71'), 
                ('writing score', 'Writing Score', '#e74c3c')]
    
    for i, (col, title, color) in enumerate(subjects):
        sns.histplot(df[col], kde=True, ax=axes[i], color=color, bins=20, edgecolor='black', alpha=0.6)
        axes[i].set_title(f'Distribution of {title}')
        axes[i].set_xlabel('Score')
        axes[i].set_ylabel('Student Count')
        axes[i].axvline(df[col].mean(), color='darkred', linestyle='--', label=f'Mean: {df[col].mean():.1f}')
        axes[i].legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "1_score_distributions.png"), dpi=300)
    plt.close()
    
    # Plot 2: Correlation Heatmap
    plt.figure(figsize=(7, 6))
    corr = df[score_cols].corr()
    sns.heatmap(corr, annot=True, cmap='Blues', fmt='.3f', linewidths=1, square=True)
    plt.title('Correlation Heatmap of Test Scores')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "2_correlation_heatmap.png"), dpi=300)
    plt.close()
    
    # Plot 3: Demographic Impact (Parental Education & Gender)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    edu_order = [
        "some high school", "high school", "some college",
        "associate's degree", "bachelor's degree", "master's degree"
    ]
    sns.boxplot(data=df, x='parental level of education', y='average_score', hue='parental level of education', order=edu_order, ax=axes[0], palette='Set2', legend=False)
    axes[0].set_title('Average Score by Parental Level of Education')
    axes[0].set_xticks(range(len(edu_order)))
    axes[0].set_xticklabels(edu_order, rotation=30, ha='right')
    axes[0].set_ylabel('Average Score')
    
    sns.boxplot(data=df, x='gender', y='average_score', hue='gender', ax=axes[1], palette='Set1', legend=False)
    axes[1].set_title('Average Score Distribution by Gender')
    axes[1].set_ylabel('Average Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "3_demographic_performance.png"), dpi=300)
    plt.close()
    
    # Plot 4: Test Prep & Lunch Impact
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.barplot(data=df, x='test preparation course', y='average_score', hue='lunch', ax=axes[0], palette='viridis', errorbar=None)
    axes[0].set_title('Average Score: Test Prep vs Lunch Type')
    axes[0].set_ylabel('Average Score')
    for p in axes[0].patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            axes[0].annotate(f'{height:.1f}', (p.get_x() + p.get_width() / 2., height / 2),
                             ha='center', va='center', fontsize=11, color='white', weight='bold')

    sns.violinplot(data=df, x='lunch', y='math score', hue='test preparation course', split=True, ax=axes[1], palette='Pastel1')
    axes[1].set_title('Math Score by Lunch & Test Prep')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "4_test_prep_and_lunch_impact.png"), dpi=300)
    plt.close()
    
    # Plot 5: Performance Levels Breakdown
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    perf_counts = df['performance_level'].value_counts()
    axes[0].pie(perf_counts, labels=perf_counts.index, autopct='%1.1f%%', colors=['#2ecc71', '#3498db', '#e74c3c'], startangle=90, explode=(0.05, 0.05, 0.05))
    axes[0].set_title('Performance Level Distribution')
    
    grade_counts = df['grade'].value_counts().sort_index()
    sns.barplot(x=grade_counts.index, y=grade_counts.values, hue=grade_counts.index, ax=axes[1], palette='crest', legend=False)
    axes[1].set_title('Grade Distribution (A+ to F)')
    axes[1].set_xlabel('Grade')
    axes[1].set_ylabel('Count')
    for p in axes[1].patches:
        axes[1].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=10, xytext=(0, 2), textcoords='offset points')
                         
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "5_grade_distribution.png"), dpi=300)
    plt.close()
    
    # 3. Association Rule Mining
    if MLXTEND_AVAILABLE:
        print("\n--- Association Rule Mining (Apriori Algorithm) ---")
        cat_cols = ['gender', 'parental level of education', 'lunch', 'test preparation course', 'performance_level']
        df_cat = df[cat_cols].copy()
        
        # One-Hot Encoding for categorical features
        df_encoded = pd.get_dummies(df_cat)
        
        # Apriori algorithm
        frequent_itemsets = apriori(df_encoded, min_support=0.08, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
        
        # Filter rules targeting performance level
        perf_rules = rules[rules['consequents'].apply(lambda x: any('performance_level' in item for item in x))].copy()
        perf_rules = perf_rules.sort_values(by='lift', ascending=False)
        
        print(f"Discovered {len(rules)} total association rules. Displaying top rules predicting Performance Level:")
        
        display_rules = []
        for idx, row in perf_rules.head(8).iterrows():
            ant = ", ".join(list(row['antecedents']))
            con = ", ".join(list(row['consequents']))
            display_rules.append({
                'Antecedents (If)': ant,
                'Consequents (Then)': con,
                'Support': f"{row['support']:.3f}",
                'Confidence': f"{row['confidence']:.3f}",
                'Lift': f"{row['lift']:.3f}"
            })
            
        rules_df = pd.DataFrame(display_rules)
        if not rules_df.empty:
            print(rules_df.to_string(index=False))
            
            # Plot 6: Association Rules Scatter
            plt.figure(figsize=(8, 5))
            sns.scatterplot(data=rules, x='support', y='confidence', hue='lift', size='lift', sizes=(40, 200), palette='coolwarm')
            plt.title('Association Rules: Support vs Confidence (Color/Size = Lift)')
            plt.xlabel('Support')
            plt.ylabel('Confidence')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "6_association_rules.png"), dpi=300)
            plt.close()
    else:
        print("\nmlxtend library not available. Skipping Association Rule Mining.")
        
    print(f"\nEDA completed successfully. All plots saved to '{plots_dir}/'.")
    print("=" * 60 + "\n")
    return df

if __name__ == "__main__":
    run_analysis()