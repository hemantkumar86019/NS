import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact

from networksecurity.utils.main_utils.utils import evaluate_models
from networksecurity.utils.main_utils.utils import load_numpy_array_data, load_object, save_object
from networksecurity.utils.ml_utils.metric.classification_metric import get_classificaton_score
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

import mlflow

# import dagshub
# dagshub.init(repo_owner='hemantkumar86019', repo_name='NS', mlflow=True)



class ModelTrainer:
    def __init__(
            self,
            data_transformation_artifact:DataTransformationArtifact,
            model_trainer_config:ModelTrainerConfig
        ):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        

    def track_mlflow(self, best_model, classification_metric):
        with mlflow.start_run():
            f1_score = classification_metric.f1_score 
            precision_score = classification_metric.precision_score
            recall_score = classification_metric.recall_score

            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision_score", precision_score)
            mlflow.log_metric("recall_score", recall_score)
            mlflow.sklearn.log_model(best_model, name = "model") 
        

    def train_model(self, X_train, y_train, X_test, y_test):
        models = {
            'KNeighbors Classifier': KNeighborsClassifier(),
            'Random Forest': RandomForestClassifier(verbose = 1),
            'Decision Tree': DecisionTreeClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(verbose = 1),
            'Logistic Regression': LogisticRegression(verbose = 1),
            'AdaBoost': AdaBoostClassifier()
        }

        params = {
            "KNeighbors Classifier": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"]
            },
            "Random Forest": {"n_estimators": [8, 16, 32, 64, 128, 256]},

            "Decision Tree": {"criterion": ["gini", "entropy", "log_loss"]},

            "Gradient Boosting": {
                "n_estimators": [8, 16, 32, 64, 128, 256],
                "learning_rate": [0.05, 0.1, 0.2],
                "subsample": [0.8, 1.0]
            },
            "Logistic Regression": {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["lbfgs", "liblinear"],
                "penalty": ["l2"]
            },
            "AdaBoost": {
                "n_estimators": [8, 16, 32, 64, 128, 256],
                "learning_rate": [0.5, 1.0, 1.5]
            }
        }

        model_report:dict = evaluate_models(
            X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models, params = params
            )
        
        ## to get best model score
        best_model_score = max(sorted(model_report.values()))

        ## to get best model name
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]

        best_model = models[best_model_name]

        print(f"Best Model: {best_model_name} with Score: {best_model_score}")

        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classificaton_score(y_true = y_train, y_pred = y_train_pred)
        ## track the experiments with mlflow
        self.track_mlflow(best_model, classification_train_metric)

        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classificaton_score(y_true = y_test, y_pred = y_test_pred)
        ## track the experiments with mlflow
        self.track_mlflow(best_model, classification_test_metric)


        preprocessor = load_object(file_path = self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok = True)

        network_model = NetworkModel(preprocessor = preprocessor, model = best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj = network_model)

        ## saving file in final_models
        save_object('final_models/model.pkl', best_model)

        ## Model Trainer Artifact
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path = self.model_trainer_config.trained_model_file_path,
            train_metric_artifact = classification_train_metric,
            test_metric_artifact = classification_test_metric
        )

        logging.info(f"Model Trainer artifact: {model_trainer_artifact}")

        return model_trainer_artifact
        


    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            ## loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)