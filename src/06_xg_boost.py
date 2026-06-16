import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score

# Load train/test
train_df = pd.read_csv('DataSet/train.csv')
test_df = pd.read_csv('DataSet/test.csv')

features = ['MONTH', 'DAY_OF_WEEK', 'DEP_HOUR', 'IS_PEAK_HOUR', 'IS_WEEKEND',
            'ORIGIN_DAILY_FLIGHTS', 'ROUTE_AVG_DELAY', 'CARRIER_AVG_DELAY',
            'OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST']
target = 'IS_DELAYED'

X_train = train_df[features].copy()
X_test = test_df[features].copy()
y_train = train_df[target]
y_test = test_df[target]

# Encode categoricals — use a fast dict-based mapping instead of the slow .index() lookup
categorical_cols = ['OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])

    class_to_idx = {cls: idx for idx, cls in enumerate(le.classes_)}
    unknown_idx = len(le.classes_)  # reserve one index for unseen categories

    X_test[col] = X_test[col].map(lambda x: class_to_idx.get(x, unknown_idx))
    encoders[col] = le

# Handle class imbalance via scale_pos_weight (negative/positive ratio)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

print("=== XGBoost ===")
print(classification_report(y_test, xgb_preds))
print("ROC-AUC:", roc_auc_score(y_test, xgb_probs))

# Feature importance — you'll want this for your README and for interview talk
importance = pd.DataFrame({
    'feature': features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:\n", importance)