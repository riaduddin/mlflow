import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn

test_size = 0.2
# random_state = 42
n_estimators = 100
max_depth = 5

def performance_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1


if __name__ == "__main__":
    data= pd.read_csv('synthetic_dataset.csv')
    train, test= train_test_split(data, test_size=test_size)
    X_train, y_train = train.drop('label', axis=1), train['label']
    X_test, y_test = test.drop('label', axis=1), test['label']
    
    mlflow.sklearn.autolog()
    my_exp= mlflow.set_experiment("RandomForest_Experiment")
    with mlflow.start_run(run_name="RandomForest_Run"):

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy, precision, recall, f1 = performance_metrics(y_test, predictions)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        
        mlflow.log_metric("F1 Score", f1)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)

        mlflow.sklearn.log_model(model, name="random_forest_model")
