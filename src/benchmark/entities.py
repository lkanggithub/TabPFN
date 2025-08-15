from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd

from dr_model_benchmark.common.analysis.entities import ModelScoreMetrics
from dr_model_benchmark.common.analysis.entities import ModelTimeProfiles
from dr_model_benchmark.common.analysis.entities import TestResultV2
from dr_model_benchmark.common.analysis.enums import Partition
from dr_model_benchmark.common.enums import MetricType
from dr_model_benchmark.common.profile.entities import Seconds
from dr_model_benchmark.common.profile.entities import TimeProfile
from dr_model_benchmark.common.profile.enums import TimeProfileType


class Dataset:
    def __init__(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_name: str,
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.target_name = target_name

    def num_unique_train_data_y(self) -> int:
        y = self.get_train_data_y()
        return len(np.unique(y))

    def get_train_data_x(self) -> np.ndarray:
        columns = list(self.train_data.columns)
        columns.remove(self.target_name)
        return self.train_data[columns].values

    def get_train_data_y(self) -> np.ndarray:
        return self.train_data[self.target_name].values

    def get_test_data_x(self) -> np.ndarray:
        columns = list(self.test_data.columns)
        columns.remove(self.target_name)
        return self.test_data[columns].values

    def get_test_data_y(self) -> np.ndarray:
        return self.test_data[self.target_name].values


@dataclass
class EvaluationResult:
    metric_type: MetricType
    score: Optional[float] = None


@dataclass
class CVEvaluationResult:
    evaluation_result: EvaluationResult
    cv_fold: int


@dataclass
class TabPFNTestReport:
    name: str
    cv_evaluation_results: List[CVEvaluationResult]
    test_set_evaluation_results: List[EvaluationResult]
    train_time_profile: List[TimeProfile]
    test_set_predict_time_profile: List[TimeProfile]

    def get_cv_metric_type(self) -> MetricType:  # FIXME
        return self.cv_evaluation_results[0].evaluation_result.metric_type

    def median_cv_score(self) -> np.floating:
        return np.median(
            [eval_result.evaluation_result.score for eval_result in self.cv_evaluation_results]
        )

    def median_test_set_score(self) -> np.floating:
        return np.median([eval_result.score for eval_result in self.test_set_evaluation_results])

    def median_train_time(self) -> np.floating:
        return np.median(
            [time_profile.time_ellipse.to_float() for time_profile in self.train_time_profile]
        )

    def total_train_time(self) -> np.floating:
        return np.sum(
            [time_profile.time_ellipse.to_float() for time_profile in self.train_time_profile]
        )

    def median_test_set_predict_time(self) -> np.floating:
        return np.median(
            [
                time_profile.time_ellipse.to_float()
                for time_profile in self.test_set_predict_time_profile
            ]
        )

    def total_test_set_predict_time(self) -> np.floating:
        return np.sum(
            [
                time_profile.time_ellipse.to_float()
                for time_profile in self.test_set_predict_time_profile
            ]
        )

    def to_test_result(self) -> TestResultV2:
        model_score_metrics = []
        if self.cv_evaluation_results:
            model_score_metrics.extend(
                [
                    ModelScoreMetrics(
                        self.get_cv_metric_type(),
                        Partition.CV,
                        self.median_cv_score().item(),
                    )
                ]
            )
        model_score_metrics.extend(
            [
                ModelScoreMetrics(
                    evaluate_result.metric_type, Partition.TEST, evaluate_result.score,
                )
                for evaluate_result in self.test_set_evaluation_results
            ]
        )
        model_time_profiles = [
            ModelTimeProfiles(
                TimeProfileType.TOTAL_CLOCK_TIME,
                Partition.TRAIN,
                Seconds(self.total_train_time()),
            ),
            ModelTimeProfiles(
                TimeProfileType.TOTAL_CLOCK_TIME,
                Partition.TEST,
                Seconds(self.total_test_set_predict_time()),
            ),
        ]
        return TestResultV2(
            dataset_name=self.name,
            model_score_metrics=model_score_metrics,
            model_time_profiles=model_time_profiles,
        )
