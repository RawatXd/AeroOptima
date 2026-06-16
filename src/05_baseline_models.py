import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score

# Load train/test
train_df = pd.read_csv('DataSet/train.csv')
test_df = pd.read_csv('DataSet/test.csv')

# Define features to use (excluding target, raw delay values, and leakage-prone columns)
features = ['MONTH', 'DAY_OF_WEEK', 'DEP_HOUR', 'IS_PEAK_HOUR', 'IS_WEEKEND',
            'ORIGIN_DAILY_FLIGHTS', 'ROUTE_AVG_DELAY', 'CARRIER_AVG_DELAY',
            'OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST']
target = 'IS_DELAYED'

X_train = train_df[features].copy()
X_test = test_df[features].copy()
y_train = train_df[target]
y_test = test_df[target]

# Encode categorical columns (carrier, origin, dest are text)
categorical_cols = ['OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    
    # Handle unseen categories in test set safely
    known_classes = set(le.classes_)
    X_test[col] = X_test[col].apply(lambda x: x if x in known_classes else 'UNKNOWN')
    
    le_classes_list = list(le.classes_)
    if 'UNKNOWN' not in le_classes_list:
        le_classes_list.append('UNKNOWN')
    le.classes_ = pd.array(le_classes_list)
    
    X_test[col] = X_test[col].map(lambda x: list(le.classes_).index(x))
    encoders[col] = le

# Model 1: Logistic Regression
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
log_reg.fit(X_train, y_train)
log_reg_preds = log_reg.predict(X_test)
log_reg_probs = log_reg.predict_proba(X_test)[:, 1]

print("=== Logistic Regression ===")
print(classification_report(y_test, log_reg_preds))
print("ROC-AUC:", roc_auc_score(y_test, log_reg_probs))

# Model 2: Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', 
                              random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]

print("\n=== Random Forest ===")
print(classification_report(y_test, rf_preds))
print("ROC-AUC:", roc_auc_score(y_test, rf_probs))