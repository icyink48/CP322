from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV


class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def get_models(self, preprocessor):
        """
        Return a dictionary of machine learning pipelines.
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
            "RandomForest": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", RandomForestClassifier(
                        n_estimators=200,
                        random_state=self.random_state,
                        n_jobs=-1
                    ))
                ]
            ),
            "LinearSVM": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", LinearSVC(random_state=self.random_state, max_iter=5000))
                ]
            )
        }

        return models

    def train_models(self, models, X_train, y_train):
        """
        Train all models and return the trained versions.
        """
        trained_models = {}

        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model

        return trained_models

    def get_tuned_random_forest(self, preprocessor):
        """
        Return a GridSearchCV object for tuning Random Forest.
        """
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestClassifier(
                    random_state=self.random_state,
                    n_jobs=-1
                ))
            ]
        )

        param_grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2]
        }

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=-1
        )

        return grid_search
