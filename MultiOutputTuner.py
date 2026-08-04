from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.base import clone

from interpret import show
from interpret.blackbox import ShapKernel, LimeTabular

from numpy import column_stack
from BaselineRegressor import BaselineRegressor
from dataLoader import column_names
import numpy as np
import pickle
import json
from joblib import Parallel, delayed
from skorch import NeuralNetRegressor


def toFloat32(x):
    return x.astype(np.float32)
               
class MultiOutputTuner:
    def __init__(self, base_estimator, param_grid, cv=5, n_jobs=None, scoring="neg_mean_squared_error", verbose=0, refit=True, reduce = 1):
        self.base_estimator = base_estimator
        self.param_grid = param_grid
        self.cv = cv
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.verbose = verbose
        self.refit = refit
        self.reduce = reduce

    def fit(self, X, y, reduce = False):
        n_outputs = y.shape[1]

        def fit_single_column(i):

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('toFloat32', FunctionTransformer(toFloat32)),
                ('PCA', PCA(n_components = self.reduce) if self.reduce !=1 else "passthrough"),
                ('model', clone(self.base_estimator))
            ])
            
            grid = GridSearchCV(
                pipeline,
                self.param_grid,
                cv=self.cv,
                n_jobs=1,
                scoring=self.scoring,
                verbose=self.verbose,
                refit = self.refit
            )
        
            y_col = y[:, i]
        
            if isinstance(self.base_estimator, NeuralNetRegressor):
                y_col = y_col.reshape(-1, 1)
        
            grid.fit(X, y_col)
        
            if self.verbose > 0:
                print(f"Column: {i} \t score: {grid.best_score_} \t params: {grid.best_params_}")
        
            return grid.best_estimator_
    
        self.models = Parallel(n_jobs=self.n_jobs)(
            delayed(fit_single_column)(i) for i in range(n_outputs)
        )
        return self
        
    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return column_stack(predictions)

    def explain_local(self, method: str, X_train, X_test, rows: int, col: int):
        model = self.models[col]
        method = method.lower()

        with open("wavelengths.json", "r") as f:
            wavelengths = json.load(f)
            feature_names = list(wavelengths["hsi_satellite_wavelengths"].values())

        
        if method == "shap":
        
            explainer = ShapKernel(
                model.predict, 
                X_train,
                feature_names = feature_names
            )
            
            explanation = explainer.explain_local(X_test[:rows])
        
            show(explanation)
        
        elif method == "lime":
        
            explainer = LimeTabular(
                model, 
                X_train,
                feature_names = feature_names
            )
            explanation = explainer.explain_local(X_test[:rows])
        
            show(explanation)
        
        else:
            raise ValueError("Unsupported method. Use 'shap' or 'lime'.")
        
    def save(self, fileName: str):
        with open(fileName, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(fileName: str):
        with open(fileName, 'rb') as file:
            return pickle.load(file)

def score(X_train, y_train, X_test, y_test, model):
    # Fit the baseline model
    baseline_reg = BaselineRegressor()
    baseline_reg = baseline_reg.fit(X_train, y_train)
    baseline_predictions = baseline_reg.predict(X_test)

    # Generate baseline values to be used in score computation
    baselines = np.mean((y_test - baseline_predictions) ** 2, axis=0)

    # Generate predictions slightly different from baseline predictions
    np.random.seed(0)
    predictions = np.zeros_like(y_test)
    for column_index in range(predictions.shape[1]):
        class_mean_value = baseline_reg.mean[column_index]
        predictions[:, column_index] = np.random.uniform(low=class_mean_value - class_mean_value * 0.2,
                                                         high=class_mean_value + class_mean_value * 0.2,
                                                         size=len(predictions))

    # Calculate MSE for each class
    mse_baseline = np.mean((y_test - predictions) ** 2, axis=0)
    print(f"Baseline mse: {mse_baseline}")

    # Calculate the score for each class individually
    scores_baseline = mse_baseline / baselines

    # Calculate the final score
    final_score = np.mean(scores_baseline)

    for score, class_name in zip(scores_baseline, column_names):
        print(f"Class {class_name} baseline score: {score}")

    print(f"Final baseline score: {final_score}")

    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2, axis = 0)
    print(f"Model mse: {mse}")
    scores = mse / baselines
    print(f"Final testing model score: {scores}")