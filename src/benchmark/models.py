from pathlib import Path
from typing import Optional

import numpy as np

from tabpfn import TabPFNClassifier
from tabpfn import TabPFNRegressor
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor

from dr_model_benchmark.common.enums import DeviceType
from dr_model_benchmark.common.enums import TargetType
from dr_model_benchmark.datarobot.predictions import PredictionOutputs

from benchmark.entities import Dataset


class ModelWrapper:
    def __init__(
        self,
        model: TabPFNClassifier | TabPFNRegressor | AutoTabPFNClassifier | AutoTabPFNRegressor,
        inference_only: Optional[bool] = False,
    ):
        self.model = model
        self.is_regressor = isinstance(self.model, TabPFNRegressor) or isinstance(
            self.model, AutoTabPFNRegressor
        )
        self.inference_only = inference_only

    @staticmethod
    def is_binary_classification_output(prediction_proba_values: np.ndarray) -> bool:
        return prediction_proba_values.shape[1] == 2

    def fit(self, dataset: Dataset) -> "ModelWrapper":
        if self.inference_only:
            return self

        self.model.fit(dataset.get_train_data_x(), dataset.get_train_data_y())
        return self

    def inference(self, dataset: Dataset) -> PredictionOutputs:
        inference_input_data = dataset.get_test_data_x()
        prediction_values = self.model.predict(inference_input_data)
        prediction_proba_values = (
            self.model.predict_proba(inference_input_data)
            if not self.is_regressor
            else None
        )
        class_labels = self.model.classes_ if not self.is_regressor else None

        prediction_proba_values = (
            prediction_proba_values[:, 1]
            if self.is_binary_classification_output(prediction_proba_values)
            else prediction_proba_values
        )

        return PredictionOutputs(
            actual_values=dataset.get_test_data_y(),
            prediction_values=prediction_values,
            prediction_proba_values=prediction_proba_values,
            class_labels=class_labels,
        )


def get_tabpfn_model(
    target_type: TargetType,
    folder_of_pretrained_models: Optional[Path] = None,
    device: Optional[DeviceType] = DeviceType.AUTO,
) -> TabPFNClassifier | TabPFNRegressor:
    model_path = folder_of_pretrained_models if folder_of_pretrained_models else "auto"
    device = device.to_torch_device_type_string()
    return (
        TabPFNRegressor(model_path=model_path, device=device, ignore_pretraining_limits=True)
        if target_type == TargetType.REGRESSION
        else TabPFNClassifier(model_path=model_path, device=device, ignore_pretraining_limits=True)
    )


def get_tabpfn_extension_model(
    target_type: TargetType,
    device: Optional[DeviceType] = DeviceType.AUTO,
) -> AutoTabPFNClassifier | AutoTabPFNRegressor:
    device = device.to_torch_device_type_string()
    return (
        AutoTabPFNRegressor(device=device)
        if target_type == TargetType.REGRESSION
        else AutoTabPFNClassifier(device=device)
    )


def get_tabpfn_model_wrapper(
    target_type: TargetType,
    folder_of_pretrained_models: Optional[Path] = None,
    use_tabpfn_extension: Optional[bool] = False,
    inference_only: Optional[bool] = False,
    device: Optional[DeviceType] = DeviceType.AUTO,
) -> ModelWrapper:
    model = (
        get_tabpfn_extension_model(target_type, device)
        if use_tabpfn_extension
        else get_tabpfn_model(target_type, folder_of_pretrained_models, device)
    )
    return ModelWrapper(
        model=model,
        inference_only=inference_only,
    )
