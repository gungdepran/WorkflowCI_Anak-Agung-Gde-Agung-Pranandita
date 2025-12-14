import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import mlflow
import dagshub
import xgboost as xgb

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import estimator_html_repr
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


DAGSHUB_USER = "gungdepran"
DAGSHUB_REPO = "Model_tracking_asah"

print("Menghubungkan ke DagsHub...")
dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)
mlflow.set_experiment("Sentiment Analysis - XGBoost Tuning")


print("Loading data...")
try:
    df = pd.read_csv('preprocessing/preprocessed_Mobile_JKN.csv')
except FileNotFoundError:
    df = pd.read_csv('preprocessed_Mobile_JKN.csv')


def get_sentiment(score):
    if score <= 3:
        return 'negatif'
    else:
        return 'positif'

df['sentiment_category'] = df['score'].apply(get_sentiment)


min_count = df['sentiment_category'].value_counts().min()
df_balanced = df.groupby('sentiment_category').apply(
    lambda x: x.sample(min_count, random_state=42)
).reset_index(drop=True)


le = LabelEncoder()
df_balanced['label'] = le.fit_transform(df_balanced['sentiment_category'])

X = df_balanced['clean_content'].values.astype(str)
y = df_balanced['label'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Data siap! Train shape: {X_train.shape}, Test shape: {X_test.shape}")

print("Memulai Hyperparameter Tuning...")

space = {
    "n_estimators": hp.quniform("n_estimators", 50, 300, 1),
    "max_depth": hp.quniform("max_depth", 3, 10, 1),
    "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
    "subsample": hp.uniform("subsample", 0.5, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
}

def objective(params):
    params["n_estimators"] = int(params["n_estimators"])
    params["max_depth"] = int(params["max_depth"])

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("xgb", xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            **params
        ))
    ])

    score = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy").mean()
    return {"loss": -score, "status": STATUS_OK}

# Jalankan Tuning
trials = Trials()
best_params = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=10, 
    trials=trials,
    rstate=np.random.default_rng(42)
)

# Konversi tipe data hasil tuning agar sesuai dengan XGBoost
best_params["n_estimators"] = int(best_params["n_estimators"])
best_params["max_depth"] = int(best_params["max_depth"])

print(f"Best Params found: {best_params}")


with mlflow.start_run(run_name="Best_XGBoost_Model"):
    
    # Pipeline dengan Best Params
    final_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ("xgb", xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            **best_params
        ))
    ])
    
    # Train Model
    final_pipeline.fit(X_train, y_train)
    
    # Evaluasi
    y_pred = final_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Final Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))

    # Logging Metrics & Params ke DagsHub
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", accuracy)
    
    # Logging Model
    mlflow.sklearn.log_model(final_pipeline, "model")
    
    # confusion Matrix Image
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig("training_confusion_matrix.png")
    mlflow.log_artifact("training_confusion_matrix.png") 
    plt.close() 
    
    # Metric Info JSON
    metrics_data = {
        "accuracy": accuracy,
        "best_params": best_params,
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    with open("metric_info.json", "w") as f:
        json.dump(metrics_data, f, indent=4)
    mlflow.log_artifact("metric_info.json")
    
    # Estimator HTML Representation
    with open("estimator.html", "w", encoding='utf-8') as f:
        f.write(estimator_html_repr(final_pipeline))
    mlflow.log_artifact("estimator.html")

    print("Proses Selesai. Silakan cek DagsHub > Experiments.")

    mlflow.sklearn.save_model(final_pipeline, "model_output_lokal")
