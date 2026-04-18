import pandas as pd

def clean_data(df):
    """
    Perform data cleaning:
    - Fix column names
    - Handle missing values
    - Remove duplicates
    - Fix data types
    """

    # 1. Fix column names
    df.columns = df.columns.str.strip()

    # 2. Check missing values (before)
    print("Missing values before cleaning:\n", df.isnull().sum())

    # 3. Separate numeric and categorical columns
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(exclude=['number']).columns

    # 4. Fill missing values
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    # 5. Remove duplicates
    duplicate_count = df.duplicated().sum()
    print(f"\nRemoving {duplicate_count} duplicate rows...")
    df = df.drop_duplicates().reset_index(drop=True)

    # 6. Fix data types (specific columns)
    numeric_cols = [
        'satisfaction_level',
        'last_evaluation',
        'number_project',
        'average_montly_hours',
        'Work_accident',
        'left'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 7. Handle NaNs after conversion
    existing_numeric_cols = [col for col in numeric_cols if col in df.columns]
    df[existing_numeric_cols] = df[existing_numeric_cols].fillna(
        df[existing_numeric_cols].median()
    )

    # 8. Final check
    print("\nData Cleaning Completed.")
    print("\nMissing values after cleaning:\n", df.isnull().sum())

    return df