import logging
import traceback
from pathlib import Path
from typing import List

import click
import pandas as pd
from dr_model_benchmark.tools.openml.utils import get_openml_study
from dr_model_benchmark.tools.openml.utils import get_openml_task
from dr_model_benchmark.tools.openml.utils import get_train_test_sets_of_openml_dataset
from dr_model_benchmark.common.analysis.entities import TestResultV2
from dr_model_benchmark.common.enums import DeviceType
from dr_model_benchmark.common.enums import MetricType
from dr_model_benchmark.common.enums import PartitionType
from dr_model_benchmark.common.enums import TargetType
from dr_model_benchmark.common.profile.entities import TimeProfile
from dr_model_benchmark.common.profile.utils import TimeProfiler

from benchmark.entities import Dataset
from benchmark.entities import TabPFNTestReport
from benchmark.models import ModelWrapper
from benchmark.evaluations import evaluate_with_cv
from benchmark.evaluations import evaluate_on_inference_result
from benchmark.models import get_tabpfn_model_wrapper


logger = logging.getLogger(__name__)


def infer_classification_target_type(
    classification_train_data: pd.DataFrame, target_name: str
) -> TargetType:
    target_col = classification_train_data[target_name]
    num_of_values = len(target_col.unique())
    return TargetType.BINARY if num_of_values == 2 else TargetType.MULTICLASS


@click.command()
@click.option(
    "--openml_study_id",
    type=int,
    required=True,
    help="OpenML study id (study contains tasks and datasets)",
)
@click.option(
    "--training_metric",
    type=click.Choice([metric_type.name for metric_type in MetricType]),
    required=True,
    help="Metric used for training",
)
@click.option(
    "--evaluation_metrics",
    type=str,
    required=True,
    help="Metric used for evaluation",
)
@click.option(
    "--output_report_path",
    type=str,
    required=True,
    help="OpenML datasets will be downloaded here",
)
@click.option(
    "--folder_of_pretrained_models",
    type=str,
    required=False,
    default="",
    help="",
)
@click.option(
    "--use_tabpfn_extension",
    type=bool,
    required=False,
    default=False,
    help="",
)
@click.option(
    "--device_type",
    type=click.Choice([device_type.name for device_type in DeviceType]),
    required=False,
    default=DeviceType.AUTO,
    help="",
)
def run_cli(
    openml_study_id: int,
    training_metric: str,
    evaluation_metrics: str,
    output_report_path: str,
    folder_of_pretrained_models: str,
    use_tabpfn_extension: bool,
    device_type: str,
) -> None:
    folder_of_pretrained_models = (
        Path(folder_of_pretrained_models) if folder_of_pretrained_models else None
    )

    openml_study = get_openml_study(openml_study_id)
    logger.info(f"Total {len(openml_study.tasks)} task(s) to test.")
    dataset_test_reports: List[TabPFNTestReport] = []
    for openml_task_id in openml_study.tasks:
        openml_task = get_openml_task(openml_task_id)
        openml_dataset = openml_task.get_dataset()
        train_dataframe, test_dataframe = get_train_test_sets_of_openml_dataset(openml_task)
        task_target_name = openml_task.target_name
        target_type = infer_classification_target_type(train_dataframe, task_target_name)
        dataset = Dataset(train_dataframe, test_dataframe, task_target_name)
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {openml_dataset.name}")
        logger.info(f"Processing task {openml_dataset.name}")

        # cross validation
        training_metric_type = MetricType.from_string(training_metric)
        model_wrapper = get_tabpfn_model_wrapper(
            target_type,
            folder_of_pretrained_models,
            use_tabpfn_extension,
            False,
            DeviceType.from_string(device_type),
        )
        try:
            cv_evaluation_results = evaluate_with_cv(
                model_wrapper,
                dataset,
                5,
                target_type,
                training_metric_type,
            )
            # train
            model_wrapper = get_tabpfn_model_wrapper(
                target_type,
                folder_of_pretrained_models,
                use_tabpfn_extension,
                False,
                DeviceType.from_string(device_type),
            )
            train_fit_time_profile = TimeProfile(PartitionType.TRAIN.name)
            with TimeProfiler(train_fit_time_profile):
                model_wrapper.fit(dataset)
            # test with holdout
            holdout_predict_time_profile = TimeProfile(PartitionType.HOLDOUT.name)
            with TimeProfiler(holdout_predict_time_profile):
                prediction_outputs = model_wrapper.inference(dataset)
        except:
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> CV fails: {openml_dataset.name}")
            traceback.print_stack()
            continue

        evaluation_metric_types = [
            MetricType.from_string(metric) for metric in evaluation_metrics.split(",")
        ]
        holdout_evaluation_results = [
            evaluate_on_inference_result(
                target_type,
                evaluation_metric_type,
                prediction_outputs,
            )
            for evaluation_metric_type in evaluation_metric_types
        ]

        # analysis and report
        dataset_test_reports.append(
            TabPFNTestReport(
                openml_dataset.name,
                cv_evaluation_results,
                holdout_evaluation_results,
                [train_fit_time_profile],
                [holdout_predict_time_profile],
            )
        )

    # report
    TestResultV2.to_csv(
        [
            TabPFNTestReport.to_test_result(dataset_test_report)
            for dataset_test_report in dataset_test_reports
        ],
        Path(output_report_path),
    )


if __name__ == "__main__":
    run_cli()
