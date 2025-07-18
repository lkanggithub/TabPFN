import logging
from pathlib import Path
from typing import List

import click

from dr_model_benchmark.common.analysis.entities import TestResult
from dr_model_benchmark.common.entities import DataRobotMBTestDatasetConfig
from dr_model_benchmark.common.enums import DeviceType
from dr_model_benchmark.common.enums import PartitionType
from dr_model_benchmark.common.profile.entities import TimeProfile
from dr_model_benchmark.common.profile.utils import TimeProfiler

from benchmark.entities import Dataset
from benchmark.entities import TabPFNTestReport
from benchmark.evaluations import evaluate_on_inference_result
from benchmark.evaluations import evaluate_with_cv
from benchmark.models import get_tabpfn_model_wrapper


logger = logging.getLogger(__name__)


def parse_dataset_name_from_dataset_path(dataset_path: str) -> str:
    dataset_path = Path(dataset_path)
    return dataset_path.parent.name.strip()


@click.command()
@click.option(
    "--datarobot_mbtest_yaml_path",
    type=str,
    required=False,
    default="",
    help="",
)
@click.option(
    "--output_report_path",
    type=str,
    required=True,
    help="",
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
    "--inference_only",
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
def run_job(
    datarobot_mbtest_yaml_path: str,
    output_report_path: str,
    folder_of_pretrained_models: str,
    use_tabpfn_extension: bool,
    inference_only: bool,
    device_type: str,
) -> None:
    dr_mbtest_dataset_config_list = DataRobotMBTestDatasetConfig.load_from_yaml(
        Path(datarobot_mbtest_yaml_path)
    )
    logger.info(f"Total datasets to test: {len(dr_mbtest_dataset_config_list)}")

    dataset_test_reports: List[TabPFNTestReport] = []
    for dr_dataset_config in dr_mbtest_dataset_config_list:
        logger.info(f"Test dataset: {dr_dataset_config.train_dataset_path}")
        metric_type = dr_dataset_config.metric
        # prepare dataset
        dataset = Dataset(
            dr_dataset_config.train_dataset_path,
            dr_dataset_config.pred_dataset_path,
            dr_dataset_config.target,
        )
        # setup model
        folder_of_pretrained_models = (
            Path(folder_of_pretrained_models) if folder_of_pretrained_models else None
        )
        model_wrapper = get_tabpfn_model_wrapper(
            dr_dataset_config.rtype,
            folder_of_pretrained_models,
            use_tabpfn_extension,
            inference_only,
            DeviceType.from_string(device_type),
        )
        # evaluate with cv
        cv_evaluation_results = evaluate_with_cv(
            model_wrapper,
            dataset,
            dr_dataset_config.reps,
            dr_dataset_config.rtype,
            metric_type,
        )
        # train
        train_fit_time_profile = TimeProfile(PartitionType.TRAIN.name)
        with TimeProfiler(train_fit_time_profile):
            model_wrapper.fit(dataset)
        # test with holdout
        holdout_predict_time_profile = TimeProfile(PartitionType.HOLDOUT.name)
        with TimeProfiler(holdout_predict_time_profile):
            inference_results = model_wrapper.inference(dataset, include_true_prediction=True)
        holdout_evaluation_result = evaluate_on_inference_result(
            dr_dataset_config.rtype,
            metric_type,
            inference_results,
        )
        # analysis and report
        dataset_test_reports.append(
            TabPFNTestReport(
                parse_dataset_name_from_dataset_path(dr_dataset_config.train_dataset_path),
                metric_type,
                cv_evaluation_results,
                [holdout_evaluation_result],
                [train_fit_time_profile],
                [holdout_predict_time_profile],
            )
        )

    # report
    TestResult.to_csv(
        [
            TabPFNTestReport.to_test_result(dataset_test_report)
            for dataset_test_report in dataset_test_reports
        ],
        Path(output_report_path),
    )


if __name__ == "__main__":
    run_job()
