import logging
from pathlib import Path
from typing import List

import click
import pandas as pd

from dr_model_benchmark.common.analysis.entities import TestResultV2
from dr_model_benchmark.common.analysis.enums import Partition
from dr_model_benchmark.common.enums import DeviceType
from dr_model_benchmark.common.profile.entities import TimeProfile
from dr_model_benchmark.common.profile.utils import TimeProfiler
from dr_model_benchmark.common.entities import DataRobotMBTestDatasetConfig

from benchmark.entities import Dataset
from benchmark.entities import TabPFNTestReport
from benchmark.evaluations import evaluate_with_cv
from benchmark.evaluations import evaluate_on_inference_result
from benchmark.models import get_tabpfn_model_wrapper


logger = logging.getLogger(__name__)


def get_dataset_name(dataset_path: Path) -> str:  # FIXME
    dataset_file_name = dataset_path.name
    return dataset_file_name.split("_train.csv")[0]


@click.command()
@click.option(
    "--datarobot_mbtest_yaml_path",
    type=str,
    required=True,
    help="Path to a DataRobot mbtest yaml file",
)
@click.option(
    "--output_report_path",
    type=str,
    required=True,
    help="Test results will be created here",
)
@click.option(
    "--folder_of_pretrained_models",
    type=str,
    required=False,
    default="",
    help="Folder of pretrained models",
)
@click.option(
    "--use_tabpfn_extension",
    type=bool,
    required=False,
    default=False,
    help="Set to True if it uses TabPFN extension",
)
@click.option(
    "--device_type",
    type=click.Choice([device_type.name for device_type in DeviceType]),
    required=False,
    default=DeviceType.AUTO,
    help="Device type",
)
@click.option(
    "--dataset_names_to_exclude",
    type=str,
    required=False,
    default="",
    help="names of datasets to be excluded from testings",
)
@click.option(
    "--run_cv",
    type=bool,
    required=False,
    default=False,
    help="If True, CV is run",
)
def run_cli(
    datarobot_mbtest_yaml_path: str,
    output_report_path: str,
    folder_of_pretrained_models: str,
    use_tabpfn_extension: bool,
    device_type: str,
    dataset_names_to_exclude: str,
    run_cv: bool,
) -> None:
    datarobot_mbtest_configs = DataRobotMBTestDatasetConfig.load_from_yaml(
        Path(datarobot_mbtest_yaml_path)
    )
    folder_of_pretrained_models = (
        Path(folder_of_pretrained_models) if folder_of_pretrained_models else None
    )

    logger.info(f"Total {len(datarobot_mbtest_configs)} task(s) to test.")
    dataset_test_reports: List[TabPFNTestReport] = []
    dataset_names_to_exclude = set(dataset_names_to_exclude.split(","))
    for mbtest_config in datarobot_mbtest_configs:
        dataset_name = get_dataset_name(Path(mbtest_config.train_dataset_path))  # FIXME
        if dataset_name in dataset_names_to_exclude:
            logger.info(f"Skipped task: {dataset_name}.")
            continue

        train_dataframe = pd.read_csv(mbtest_config.train_dataset_path)
        test_dataframe = pd.read_csv(mbtest_config.pred_dataset_path)
        target_name = mbtest_config.target
        target_type = mbtest_config.rtype
        dataset = Dataset(train_dataframe, test_dataframe, target_name)
        logger.info(f"Processing task {dataset_name}")

        # cross validation
        try:
            cv_evaluation_results = []
            if run_cv:
                model_wrapper = get_tabpfn_model_wrapper(
                    target_type,
                    folder_of_pretrained_models,
                    use_tabpfn_extension,
                    False,
                    DeviceType.from_string(device_type),
                )
                cv_evaluation_results = evaluate_with_cv(
                    model_wrapper,
                    dataset,
                    5,
                    target_type,
                    mbtest_config.metric,
                )
            # train
            model_wrapper = get_tabpfn_model_wrapper(
                target_type,
                folder_of_pretrained_models,
                use_tabpfn_extension,
                False,
                DeviceType.from_string(device_type),
            )
            train_fit_time_profile = TimeProfile(Partition.TRAIN.name)
            with TimeProfiler(train_fit_time_profile):
                model_wrapper.fit(dataset)
            # test with holdout
            holdout_predict_time_profile = TimeProfile(Partition.TEST.name)
            with TimeProfiler(holdout_predict_time_profile):
                prediction_outputs = model_wrapper.inference(dataset)
        except:
            logger.exception(f"CV fails: {dataset_name}")
            continue

        holdout_evaluation_results = [
            evaluate_on_inference_result(
                target_type,
                mbtest_config.metric,
                prediction_outputs,
            )
        ]

        # analysis and report
        dataset_test_reports.append(
            TabPFNTestReport(
                dataset_name,
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
