import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def ordinal_encode(df):
    """
    Apply ordinal encoding to ordered categorical features.
    """
    if 'salary' in df.columns:
        df['salary'] = df['salary'].map({
            'low': 1,
            'medium': 2,
            'high': 3
        })
    return df


def split_data(df, target_col='left', test_size=0.2, random_state=42):
    """
    Split dataset into train and test sets with stratification.
    """
    y = df[target_col]
    X = df.drop(target_col, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def encode_and_scale(X_train, X_test):
    """
    Apply One-Hot Encoding and Scaling after splitting.
    """

    # 1. Define columns (adjust if dataset differs)
    categorical_cols =["Departments"]
    numeric_cols = [
        "average_montly_hours",
        "time_spend_company",
        "number_project"
    ]

    print("Categorical:", categorical_cols)
    print("Numeric:", numeric_cols)

    # 2. One-Hot Encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    if categorical_cols:
        X_train_cat = encoder.fit_transform(X_train[categorical_cols])
        X_test_cat = encoder.transform(X_test[categorical_cols])

        X_train_cat_df = pd.DataFrame(
            X_train_cat,
            columns=encoder.get_feature_names_out(categorical_cols),
            index=X_train.index
        )

        X_test_cat_df = pd.DataFrame(
            X_test_cat,
            columns=encoder.get_feature_names_out(categorical_cols),
            index=X_test.index
        )

        # Drop original categorical columns
        X_train = X_train.drop(columns=categorical_cols)
        X_test = X_test.drop(columns=categorical_cols)

        # Merge encoded features
        X_train = pd.concat([X_train, X_train_cat_df], axis=1)
        X_test = pd.concat([X_test, X_test_cat_df], axis=1)

    # 3. Scaling
    scaler = RobustScaler()

    if numeric_cols:
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, encoder, scaler