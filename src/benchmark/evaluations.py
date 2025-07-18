from typing import Callable
from typing import List

from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from dr_model_benchmark.common.analysis.evaluations import compute_metric
from dr_model_benchmark.common.enums import MetricType
from dr_model_benchmark.common.enums import TargetType
from dr_model_benchmark.datarobot.predictions import PredictionOutputs

from benchmark.entities import CVEvaluationResult
from benchmark.entities import Dataset
from benchmark.entities import EvaluationResult
from benchmark.models import ModelWrapper


def get_sklearn_scorer(
    target_type: TargetType,
    metric_type: MetricType,
) -> Callable:
    is_multiclass = target_type == TargetType.MULTICLASS
    scorer_args = {}
    if target_type.is_classification():
        scorer_args.update({"response_method": "predict_proba"})
    if metric_type == MetricType.AUC and is_multiclass:  # FIXME
        scorer_args.update({"multi_class": "ovr"})

    return make_scorer(get_sklearn_score_func(metric_type), **scorer_args)


def get_sklearn_score_func(metric_type: MetricType) -> Callable:
    metric_handler = {
        MetricType.AUC: roc_auc_score,
        MetricType.RMSE: root_mean_squared_error,
        MetricType.LOGLOSS: log_loss,
    }
    return metric_handler[metric_type]


def evaluate_on_inference_result(
    target_type: TargetType,
    metric_type: MetricType,
    prediction_outputs: PredictionOutputs,
) -> EvaluationResult:
    score = compute_metric(prediction_outputs, metric_type, target_type==TargetType.MULTICLASS)
    return EvaluationResult(metric_type, score)


def evaluate_with_cv(
    model_wrapper: ModelWrapper,
    dataset: Dataset,
    num_of_folds: int,
    target_type: TargetType,
    metric_type: MetricType,
) -> List[CVEvaluationResult]:
    cv = KFold(n_splits=num_of_folds, shuffle=True, random_state=1234)
    cv_scores = cross_val_score(
        estimator=model_wrapper.pipeline,
        X=dataset.get_train_data_x(),
        y=dataset.get_train_data_y(),
        cv=cv,
        scoring=get_sklearn_scorer(target_type, metric_type),
    )

    return [
        CVEvaluationResult(EvaluationResult(metric_type, cv_score), idx)
        for idx, cv_score in enumerate(cv_scores)
    ]
