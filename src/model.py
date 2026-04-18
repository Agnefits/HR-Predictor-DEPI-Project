# ML Models Build
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# ANN (DL) Model Build
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
# ML Model Selection
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# ML Models Build
def compute_class_weight_scale(y_train):
    """
    Compute imbalance ratio for XGBoost.
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    if pos == 0:
        return 1.0

    return neg / pos


def build_models(y_train):
    """
    Initialize ML models with proper imbalance handling.
    """
    scale = compute_class_weight_scale(y_train)

    log_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    )

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight='balanced_subsample',
        random_state=42
    )

    xgb_model = XGBClassifier(
        scale_pos_weight=scale,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        eval_metric='logloss',
        random_state=42
    )

    return log_model, rf_model, xgb_model


# ANN (DL) Model Build
def train_models(models, X_train, y_train):
    """
    Train all models and return trained versions.
    """
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


def build_ann(X_train, y_train, X_val=None, y_val=None,
              epochs=32, batch_size=32):
    """
    Build, train, and return ANN model + predictions.
    """

    # 1. Build model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # 2. Compile
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # 3. Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # 4. Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if X_val is not None else None,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    return model, history


def predict_ann(model, X_test, threshold=0.5):
    """
    Generate predictions for ANN.
    """
    y_prob = model.predict(X_test).flatten()
    y_pred = (y_prob > threshold).astype(int)

    return y_pred, y_prob

# ML Model Selection
def get_feature_importance(rf_model, feature_names):
    """
    Return sorted feature importances and indices.
    """
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    return importances, indices

def select_top_features(importances, indices, feature_names, threshold=0.01):
    """
    Select features above importance threshold.
    """
    selected = [feature_names[i] for i in indices if importances[i] > threshold]
    return selected

def tune_random_forest(X_train, y_train, n_iter=30, cv=5):
    """
    RandomizedSearchCV for Random Forest.
    """
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    rf_random = RandomizedSearchCV(
        RandomForestClassifier(
            class_weight='balanced_subsample',
            random_state=42
        ),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    rf_random.fit(X_train, y_train)
    return rf_random