import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# PHASE 1: DATA LOADING & SCHEMA VALIDATION
# ==========================================

def load_and_validate_data(file_path):
    """
    Loads the dataset and ensures essential columns are present.
    In production engineering, this prevents silent downstream failures.
    """
    print("--- Phase 1: Loading & Validating Data ---")
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
        print(f"Columns found in dataset: {list(df.columns)}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    # Define the essential schema columns expected for our pipeline
    required_columns = ['prompt_text', 'response_text', 'language', 'domain', 'model_name', 'hallucination_type', 'hallucination_label']

    # Check for missing columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CRITICAL: Dataset is missing required columns: {missing_cols}")

    print("Schema validation passed. Required features are present.\n")
    return df


# ==========================================
# PHASE 2: MULTILINGUAL TEXT PREPROCESSING
# ==========================================

def clean_multilingual_text(text):
    """
    Cleans text without breaking non-English scripts.
    Crucial for Featherless AI: Standard Regex [A-Za-z] strips Arabic, Hindi, etc.
    We use Unicode-aware matching (\w) to preserve non-Latin alphabets.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase (safe for English/French/German, ignored by Arabic/Hindi)
    text = text.lower()

    # Strip URLs and web links
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove excessive punctuation but keep word characters from all scripts (\w) and spaces (\s)
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)

    # Remove redundant whitespace whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocess_pipeline(df):
    """
    Applies text cleaning and handles missing target variables safely.
    """
    print("--- Phase 2: Running Text Preprocessing Pipeline ---")

    # Run text cleaning on prompt and response features
    df['cleaned_prompt'] = df['prompt_text'].apply(clean_multilingual_text)
    df['cleaned_response'] = df['response_text'].apply(clean_multilingual_text)

    # Combine prompt and response to give the classification model complete context
    df['combined_text'] = df['cleaned_prompt'] + " " + df['cleaned_response']

    # Ensure our target binary variable is structured cleanly (1 for Hallucinated, 0 for Not)
    # Mapping in case data uses text strings like 'Yes'/'No' or 'hallucinated'/'not'
    if df['hallucination_label'].dtype == 'object':
        df['target_binary'] = df['hallucination_label'].str.lower().map({'yes': 1, 'hallucinated': 1, True: 1, 'no': 0, 'not': 0, False: 0})
    else:
        df['target_binary'] = df['hallucination_label'].astype(int)

    print("Text processing complete. Created 'combined_text' and 'target_binary' fields.\n")
    return df


# ==========================================
# PHASE 3: MODEL TRAINING & EVALUATION
# ==========================================

def train_hallucination_classifier(df):
    """
    Trains a baseline classifier to programmatically spot hallucinations.
    Uses sublinear TF-IDF scaling to handle text length variations across LLMs.
    """
    print("--- Phase 3: Building & Evaluating Classifier ---")

    # Separate features and target label
    X = df['combined_text']
    y = df['target_binary']

    # Split data: 80% for training parameters, 20% held out for evaluation integrity
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Vectorize text into numerical tokens
    # 'ngram_range=(1,2)' catches single words and two-word pairings (e.g., 'not true')
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Logistic Regression baseline (Highly interpretable for data diagnostics)
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train_vec, y_train)

    # Predict outputs on unseen testing partition
    predictions = model.predict(X_test_vec)

    # Performance Evaluation Summary
    print("--- Model Accuracy Evaluation ---")
    print(f"Accuracy Score: {accuracy_score(y_test, predictions):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['Not Hallucinated', 'Hallucinated']))

    return model, vectorizer


# ==========================================
# PHASE 4: REVENUE & PRODUCTION DIAGNOSTICS
# ==========================================

def run_production_analytics(df):
    """
    Computes cross-model, multilingual, and high-stakes domain breakdowns.
    This provides the operational metrics hiring managers want to see.
    """
    print("\n--- Phase 4: Running Operational Diagnostics ---")

    # 1. Cross-Model Hallucination Comparison
    print("\n[Diagnostic 1] Hallucination Rate By Model Framework:")
    model_analysis = df.groupby('model_name')['target_binary'].mean().sort_values(ascending=False)
    for model, rate in model_analysis.items():
        print(f" * {model}: {rate:.1%}")

    # 2. Multilingual Degradation Tracking
    print("\n[Diagnostic 2] Hallucination Rate By Language context:")
    lang_analysis = df.groupby('language')['target_binary'].mean().sort_values(ascending=False)
    for lang, rate in lang_analysis.items():
        print(f" * {lang}: {rate:.1%}")

    # 3. High-Stakes Domain Vulnerability Check
    print("\n[Diagnostic 3] Risk Level across High-Stakes Domains:")
    domain_analysis = df.groupby('domain')['target_binary'].mean().sort_values(ascending=False)
    for domain, rate in domain_analysis.items():
        marker = "🚨 HIGH RISK" if domain in ['Medicine', 'Law', 'Finance'] and rate > 0.3 else "Normal"
        print(f" * {domain}: {rate:.1%}- [{marker}]")

    # 4. Mitigation Strategy Effectiveness Matrix
    if 'mitigation_strategy' in df.columns:
        print("\n[Diagnostic 4] Mitigation Performance Matrix:")
        mit_analysis = df.groupby('mitigation_strategy')['target_binary'].mean().sort_values()
        for strategy, rate in mit_analysis.items():
            print(f" * {strategy}: Reduced failure rate down to {rate:.1%}")


# ==========================================
# PIPELINE EXECUTION ENTRYPOINT
# ==========================================

if __name__ == "__main__":
    # Specify your local path or colab uploaded filename
    DATASET_PATH = '/content/llm_hallucination_dataset_v1.csv'

    # Run full sequence sequentially
    processed_df = load_and_validate_data(DATASET_PATH)

    if processed_df is not None:
        processed_df = preprocess_pipeline(processed_df)
        trained_model, text_vectorizer = train_hallucination_classifier(processed_df)
        run_production_analytics(processed_df)
        print("\nPipeline executed cleanly.")

###### ----------------------------------------------------------- ######

#The results:

--- Phase 2: Running Text Preprocessing Pipeline ---
Text processing complete. Created 'combined_text' and 'target_binary' fields.

--- Phase 3: Building & Evaluating Classifier ---
--- Model Accuracy Evaluation ---
Accuracy Score: 100.00%

Classification Report:
                  precision    recall  f1-score   support

Not Hallucinated       1.00      1.00      1.00        26
    Hallucinated       1.00      1.00      1.00        14

        accuracy                           1.00        40
       macro avg       1.00      1.00      1.00        40
    weighted avg       1.00      1.00      1.00        40


--- Phase 4: Running Operational Diagnostics ---

[Diagnostic 1] Hallucination Rate By Model Framework:
 * Gemini-1.5-Pro: 50.0%
 * Claude-3.5-Sonnet: 38.7%
 * GPT-4o: 31.0%
 * Llama-3.1-70B: 29.2%
 * Mistral-Large: 28.9%

[Diagnostic 2] Hallucination Rate By Language context:
 * Spanish: 55.0%
 * French: 38.5%
 * Mandarin: 37.5%
 * English: 34.2%
 * German: 22.2%
 * Arabic: 21.4%
 * Hindi: 18.2%

[Diagnostic 3] Risk Level across High-Stakes Domains:
 * History: 69.6%- [Normal]
 * Technology: 52.9%- [Normal]
 * Law: 42.1%- [🚨 HIGH RISK]
 * Finance: 36.7%- [🚨 HIGH RISK]
 * Medicine: 31.6%- [🚨 HIGH RISK]
 * Science: 13.3%- [Normal]
 * General: 0.0%- [Normal]
 * Politics: 0.0%- [Normal]

[Diagnostic 4] Mitigation Performance Matrix:
 * CoT-Prompting: Reduced failure rate down to 31.7%
 * Self-Consistency: Reduced failure rate down to 34.1%
 * RAG: Reduced failure rate down to 35.7%
 * Structured-Prompt: Reduced failure rate down to 37.8%

Pipeline executed cleanly.
<>:47: SyntaxWarning: invalid escape sequence '\w'
<>:47: SyntaxWarning: invalid escape sequence '\w'
/tmp/ipykernel_9852/1533316056.py:47: SyntaxWarning: invalid escape sequence '\w'
  We use Unicode-aware matching (\w) to preserve non-Latin alphabets.
