import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Load CSV
df = pd.read_csv("form_field_labels.csv")
df = df.dropna(subset=["word", "x", "y", "width", "height", "is_label"])
# Feature columns
X = df[["word", "x", "y", "width", "height"]]
y = df["is_label"]

# Simple text+numeric pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from joblib import dump


text_vectorizer = CountVectorizer()
preprocessor = ColumnTransformer([
    ("text", text_vectorizer, "word"),
    ("numeric", StandardScaler(), ["x", "y", "width", "height"]),
])

pipeline = Pipeline([
    ("features", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
pipeline.fit(X_train, y_train)
dump(pipeline, 'label_classifier_model.pkl', compress=3)

print("Model accuracy:", pipeline.score(X_test, y_test))




