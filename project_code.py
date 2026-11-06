# Name: Matthew Muth | Number: 0402

import polars as pl
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier
import pandas as pd

train_df = pl.read_csv("training.csv.gz", null_values=['NA'], infer_schema_length=None)
test_df = pl.read_csv("testing.csv.gz", null_values=['NA'], infer_schema_length=None)

def parse_approach(x):
    if x is None:
        return None
    inner = x.strip('{}')
    if inner == '':
        return None
    parts = inner.split(',')
    result = []
    for p in parts:
        p = p.strip()
        if p == '' or p.upper() == 'NA':
            result.append(None)
        else:
            try:
                result.append(float(p))
            except ValueError:
                result.append(None)
    return result

def preprocess(df):
    df = df.with_columns(
        pl.col('closestdefapproach')
          .map_elements(parse_approach, return_dtype=pl.List(pl.Float64))
          .alias('closestdefapproach')
    )

    # extract features from the parsed array
    df = df.with_columns([
        pl.col('closestdefapproach').list.get(0).alias('def_dist_at_1s'),
        pl.col('closestdefapproach').list.get(1).alias('def_dist_at_075s'),
        pl.col('closestdefapproach').list.get(2).alias('def_dist_at_05s'),
        pl.col('closestdefapproach').list.get(3).alias('def_dist_at_025s'),
        pl.col('closestdefapproach').list.min().alias('def_dist_min_pre_release'),
        pl.col('closestdefapproach').list.len().alias('def_approach_len'),
        (pl.col('closestdefapproach').list.first() - pl.col('closestdefdist'))
            .alias('def_closing_speed_to_release'),
    ])

    df = df.with_columns([
        pl.col(c).fill_null(99) for c in ['distcont1', 'distcont2', 'distcont3', 'distcont4']
    ])

    # Check for null values in defender appraoch and fill with median
    approach_feature_cols = ['def_dist_at_1s', 'def_dist_at_075s', 'def_dist_at_05s', 'def_dist_at_025s']
    df = df.with_columns([
        pl.col(c).is_null().alias(f'missing_{c}') for c in approach_feature_cols
    ])
    df = df.with_columns([
        pl.col(c).fill_null(pl.col(c).median()) for c in approach_feature_cols
    ])

    # Calibrating court dimensions and creating shot zones
    rim_x, rim_y = -41.75, 0

    df = df.with_columns([
        (((pl.col('locationx') - rim_x)**2 + (pl.col('locationy') - rim_y)**2).sqrt())
            .alias('calc_distance'),
        (pl.arctan2(pl.col('locationy') - rim_y, pl.col('locationx') - rim_x))
            .alias('shot_angle_rad'),
    ])

    df = df.with_columns(
        pl.when(pl.col('three') & (pl.col('locationy').abs() > 22.0) & (pl.col('locationx') < -40))
          .then(pl.lit('corner_three'))
        .when(pl.col('three'))
          .then(pl.lit('key_three'))
        .when(pl.col('calc_distance') <= 4)
          .then(pl.lit('rim'))
        .when(pl.col('calc_distance') <= 8)
          .then(pl.lit('short_mid'))
        .when(pl.col('calc_distance') <= 16)
          .then(pl.lit('mid_range'))
        .otherwise(pl.lit('long_two'))
        .alias('shot_zone')
    )

    return df

train_df = preprocess(train_df)
test_df = preprocess(test_df)


target_col = 'outcome'

drop_cols = [
    'shot_id', 'outcome', 'closestdef_id', 'closestdefapproach', 'calc_distance',
    'contester1_id', 'contester2_id', 'contester3_id', 'contester4_id', 'shooter_id', 'def_approach_len',
    'missing_def_dist_at_1s', 'missing_def_dist_at_075s',
    'missing_def_dist_at_05s', 'missing_def_dist_at_025s'
]

feature_cols = [c for c in train_df.columns if c not in drop_cols]
categorical_cols = ['season_id', 'off_team_id', 'def_team_id', 'gamestate', 'shottype', 'month', 'shot_zone']

train_encoded = train_df.select(feature_cols).to_dummies(columns=categorical_cols)
test_encoded = test_df.select(feature_cols).to_dummies(columns=categorical_cols)

train_cols = train_encoded.columns
test_encoded = test_encoded.select([
    pl.col(c) if c in test_encoded.columns else pl.lit(0).alias(c)
    for c in train_cols
])

X_train = train_encoded.to_numpy()
X_test = test_encoded.to_numpy()

y_train = train_df[target_col].to_numpy()
y_test = test_df[target_col].to_numpy() if target_col in test_df.columns else None

# param_grid = {
#     'n_estimators': [200, 400, 600],
#     'max_depth': [3, 4, 5, 6, 8],
#     'learning_rate': [0.01, 0.03, 0.05, 0.1],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0],
# }

# search = RandomizedSearchCV(
#     XGBClassifier(eval_metric='logloss'),
#     param_distributions=param_grid,
#     n_iter=30,
#     scoring='neg_log_loss',
#     cv=5,
#     random_state=42,
#     n_jobs=-1
# )
# search.fit(X_train, y_train)

# print(search.best_params_)
# best_model = search.best_estimator_

# Training the model
model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.6,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

train_probs = model.predict_proba(X_train)[:, 1]
test_probs = model.predict_proba(X_test)[:, 1]

test_df = test_df.with_columns(pl.Series('make_prob', test_probs))

# print("=== Full Train ===")
# print("Log loss:", log_loss(y_train, train_probs))
# print("Brier score:", brier_score_loss(y_train, train_probs))
# print("AUC:", roc_auc_score(y_train, train_probs))

# importances = pd.Series(model.feature_importances_, index=train_cols).sort_values(ascending=False)
# print(importances.head(15))
# print(test_df['make_prob'].describe())

test_df.select(['shot_id', 'make_prob']).write_csv('submission.csv')