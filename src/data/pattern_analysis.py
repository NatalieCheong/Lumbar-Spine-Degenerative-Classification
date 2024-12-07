import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main function for analyzing patterns in the lumbar spine data"""
    base_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification'

    # Load necessary data
    train_df = pd.read_csv(f"{base_path}/train.csv")
    coords_df = pd.read_csv(f"{base_path}/train_label_coordinates.csv")
    series_df = pd.read_csv(f"{base_path}/train_series_descriptions.csv")

    # Analyze patterns
    print("1. Analyzing class distribution patterns...")
    analyze_class_distributions(train_df)

    print("\n2. Analyzing condition co-occurrence...")
    analyze_condition_cooccurrence(train_df)

    print("\n3. Analyzing condition patterns across vertebral levels...")
    analyze_level_patterns(train_df)

    print("\n4. Analyzing series descriptions and their relationships to conditions...")
    analyze_series_patterns(series_df, coords_df)

def analyze_class_distributions(train_df):
    """Analyze the distribution of severity classes across different conditions"""
    # Get condition columns
    condition_cols = [col for col in train_df.columns
                     if any(cond in col for cond in ['foraminal', 'subarticular', 'canal'])]

    # Create figure for distribution plots
    plt.figure(figsize=(15, 8))
    severity_counts = {}

    # Count severity distributions for each condition type
    for col in condition_cols:
        severity_counts[col] = train_df[col].value_counts()

    # Convert to DataFrame for easier plotting
    severity_df = pd.DataFrame(severity_counts).fillna(0)

    # Plot overall severity distribution
    severity_df.transpose().plot(kind='bar', stacked=True)
    plt.title('Severity Distribution Across All Conditions and Levels')
    plt.xlabel('Condition and Level')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nSeverity Distribution Summary:")
    total_annotations = severity_df.sum().sum()
    for severity in ['Normal/Mild', 'Moderate', 'Severe']:
        count = severity_df.loc[severity].sum()
        percentage = (count/total_annotations) * 100
        print(f"{severity}: {count:.0f} cases ({percentage:.1f}%)")

def analyze_condition_cooccurrence(train_df):
    """Analyze how different conditions co-occur"""
    # Create separate DataFrames for each condition type
    conditions = {
        'canal': [col for col in train_df.columns if 'canal' in col],
        'foraminal': [col for col in train_df.columns if 'foraminal' in col],
        'subarticular': [col for col in train_df.columns if 'subarticular' in col]
    }

    # Create co-occurrence matrix
    plt.figure(figsize=(12, 8))
    cooccurrence_matrix = np.zeros((3, 3))
    condition_types = list(conditions.keys())

    for i, cond1 in enumerate(condition_types):
        for j, cond2 in enumerate(condition_types):
            # Count cases where both conditions are severe
            severe_cases1 = train_df[conditions[cond1]] == 'Severe'
            severe_cases2 = train_df[conditions[cond2]] == 'Severe'
            cooccurrence = (severe_cases1.any(axis=1) & severe_cases2.any(axis=1)).sum()
            cooccurrence_matrix[i, j] = cooccurrence

    # Plot co-occurrence heatmap
    sns.heatmap(cooccurrence_matrix,
                annot=True,
                fmt='g',
                xticklabels=condition_types,
                yticklabels=condition_types)
    plt.title('Co-occurrence of Severe Cases Between Conditions')
    plt.tight_layout()
    plt.show()

    print("\nKey Co-occurrence Patterns:")
    for i, cond1 in enumerate(condition_types):
        for j, cond2 in enumerate(condition_types):
            if i < j:
                print(f"{cond1.capitalize()} + {cond2.capitalize()}: {cooccurrence_matrix[i,j]:.0f} severe cases")

def analyze_level_patterns(train_df):
    """Analyze patterns across vertebral levels"""
    levels = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
    conditions = ['canal', 'foraminal', 'subarticular']

    plt.figure(figsize=(15, 6))

    # Create matrix of severity by level
    level_severity = np.zeros((len(conditions), len(levels)))

    for i, condition in enumerate(conditions):
        for j, level in enumerate(levels):
            # Get columns for this condition and level
            cols = [col for col in train_df.columns if condition in col and level in col]
            # Calculate percentage of severe cases
            severe_cases = (train_df[cols] == 'Severe').sum().sum()
            total_cases = train_df[cols].notna().sum().sum()
            level_severity[i, j] = (severe_cases / total_cases * 100) if total_cases > 0 else 0

    # Plot level severity patterns
    sns.heatmap(level_severity,
                annot=True,
                fmt='.1f',
                xticklabels=levels,
                yticklabels=conditions,
                cmap='YlOrRd')
    plt.title('Percentage of Severe Cases by Level and Condition')
    plt.xlabel('Vertebral Level')
    plt.ylabel('Condition Type')
    plt.tight_layout()
    plt.show()

    print("\nLevel-wise Pattern Summary:")
    for condition in conditions:
        print(f"\n{condition.capitalize()} patterns:")
        for level in levels:
            cols = [col for col in train_df.columns if condition in col and level in col]
            severe_count = (train_df[cols] == 'Severe').sum().sum()
            total_count = train_df[cols].notna().sum().sum()
            if total_count > 0:
                percentage = (severe_count / total_count) * 100
                print(f"  {level}: {percentage:.1f}% severe cases")

def analyze_series_patterns(series_df, coords_df):
    """Analyze patterns in series descriptions and their relationship to conditions"""
    # Count series descriptions
    series_counts = series_df['series_description'].value_counts()

    plt.figure(figsize=(12, 6))
    series_counts.plot(kind='bar')
    plt.title('Distribution of Series Types')
    plt.xlabel('Series Description')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Analyze which series types are used for different conditions
    condition_series = coords_df.merge(series_df,
                                     left_on=['study_id', 'series_id'],
                                     right_on=['study_id', 'series_id'])

    print("\nSeries Usage by Condition:")
    for condition in condition_series['condition'].unique():
        print(f"\n{condition}:")
        series_for_condition = condition_series[
            condition_series['condition'] == condition
        ]['series_description'].value_counts()

        for series_type, count in series_for_condition.items():
            print(f"  {series_type}: {count} annotations")

if __name__ == "__main__":
    main()
