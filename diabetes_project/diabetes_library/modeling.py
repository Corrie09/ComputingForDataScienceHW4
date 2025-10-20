from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

FEATURES = [
    "age", "height", "weight", "aids", "cirrhosis",
    "hepatic_failure", "immunosuppression", "leukemia",
    "lymphoma", "solid_tumor_with_metastasis"
]
TARGET = "diabetes_mellitus"

def split_data(df, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    return train_test_split(df, test_size=test_size, random_state=random_state)


def train_model(train_df):
    """Train logistic regression model."""
    X = train_df[FEATURES]
    y = train_df[TARGET]
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def add_predictions(df, model):
    """Add predicted probabilities to dataframe."""
    df["predictions"] = model.predict_proba(df[FEATURES])[:, 1]
    return df
