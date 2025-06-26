import time

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, \
    recall_score, f1_score


def compute_regression_metrics(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }


def compute_classification_metrics(y_true, y_pred, average='macro'):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=average, zero_division=0)
    }


def evaluate_model(model, X, y, is_classification_task=False):
    start_time = time.time()
    y_pred = model.predict(X)
    end_time = time.time()

    if is_classification_task:
        metrics = compute_classification_metrics(y, y_pred)
    else:
        metrics = compute_regression_metrics(y, y_pred)

    metrics["Runtime (s)"] = end_time - start_time
    return metrics


def evaluate_model_and_print(model, model_name, X_train, y_train, X_test, y_test, is_classification_task=False):
    print("========== {} ==========".format(model_name))

    print("Training model")
    train_metrics = evaluate_model(model, X_train, y_train, is_classification_task=is_classification_task)
    print(train_metrics)

    print("Testing model")
    test_metrics = evaluate_model(model, X_test, y_test, is_classification_task=is_classification_task)
    print(test_metrics)
