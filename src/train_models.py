from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV


class ModelTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def get_models(self, preprocessor):
        """
        Create model pipelines that include preprocessing.
        """
        models = {
            "DummyClassifier": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", DummyClassifier(strategy="most_frequent"))
                ]
            ),
            "LogisticRegression": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", LogisticRegression(max_iter=1000, random_state=self.random_state))
                ]
            ),
            "DecisionTree": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", DecisionTreeClassifier(random_state=self.random_state))
                ]
            ),
            "RandomForest": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", RandomForestClassifier(
                        n_estimators=200,
                        random_state=self.random_state,
                        n_jobs=-1
                    ))
                ]
            )
        }
        return models

    def train_models(self, models, X_train, y_train):
        """
        Train all models and return trained pipelines.
        """
        trained_models = {}

        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model

        return trained_models
