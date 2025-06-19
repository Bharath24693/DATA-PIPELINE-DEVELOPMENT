import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Extract: Load data from CSV
def extract_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    print("Extracting data...")
    return pd.read_csv(file_path)

# Transform: Clean and preprocess data
def transform_data(df, target_column='target'):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]

    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    X_processed = preprocessor.fit_transform(X)

    # Get final column names
    cat_encoded_cols = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)
    all_columns = numeric_cols + list(cat_encoded_cols)

    X_df = pd.DataFrame(X_processed, columns=all_columns)
    return X_df, y

# Load: Save processed data to CSV
def load_data(X, y, output_file='processed_data.csv'):
    print(f" Saving to {output_file}...")
    X['target'] = y
    X.to_csv(output_file, index=False)
    print(" Data saved successfully.")

# Run ETL pipeline
def run_pipeline():
    print(" Pipeline started")  # Debug line to confirm running
    input_file = 'input_data.csv'
    output_file = 'processed_data.csv'
    target_column = 'target'

    df = extract_data(input_file)
    X_processed, y = transform_data(df, target_column)
    load_data(X_processed, y, output_file)

# Main execution
if _name_ == "_main_":
    run_pipeline()
