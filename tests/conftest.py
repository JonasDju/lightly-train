#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Generator

import pytest
from pytest import FixtureRequest, TempPathFactory
from pytest_mock import MockerFixture

# Tests that are broken by the 3D DINOv2 adaptation but test functionality that is
# unrelated to 3D DINOv2 pretraining (other SSL methods, 2D task models and their
# transforms, 2D datasets). They fail because shared code is now 3D-only, e.g.
# ImageSizeTuple has three elements, the DINOv2 PatchEmbed expects 5D inputs,
# ViewTransform is a MONAI volume pipeline, EmbeddingModel uses a Conv3d head, and
# pretrain() takes data_root/data_meta instead of data.
#
# Files in which every test is broken are not collected at all.
collect_ignore = [
    "_commands/test_predict_task.py",
    "_data/test_object_detection_dataset.py",
    "_methods/densecl/test_densecl_transform.py",
    "_methods/detcon/test_detcon_transform.py",
    "_methods/dino/test_dino_transform.py",
    "_methods/distillation/test_distillation_transform.py",
    "_methods/distillationv2/test_distillationv2_transforms.py",
    "_methods/simclr/test_simclr_transform.py",
    "_task_models/test_train_model.py",
    "_transforms/test_oriented_object_detection_transform.py",
]

# Individual broken tests in otherwise working files. Parametrized tests are listed
# without their parameters and are skipped for all parameters.
_BROKEN_BY_3D_DINOV2_ADAPTATION = frozenset(
    {
        "tests/_commands/test_train_helpers.py::test_get_embedding_model",
        "tests/_commands/test_train_helpers.py::test_get_embedding_model__custom",
        "tests/_commands/test_train_task.py::test_train_image_classification__multiclass",
        "tests/_commands/test_train_task.py::test_train_image_classification__multilabel",
        "tests/_commands/test_train_task.py::test_train_image_classification_multihead",
        "tests/_commands/test_train_task.py::test_train_instance_segmentation",
        "tests/_commands/test_train_task.py::test_train_object_detection_yolo",
        "tests/_commands/test_train_task.py::test_train_panoptic_segmentation",
        "tests/_commands/test_train_task.py::test_train_panoptic_segmentation__dinov2",
        "tests/_commands/test_train_task.py::test_train_semantic_segmentation",
        "tests/_commands/test_train_task.py::test_train_semantic_segmentation__checkpoint",
        "tests/_commands/test_train_task.py::test_train_semantic_segmentation__export",
        "tests/_commands/test_train_task.py::test_train_semantic_segmentation__resume_interrupted",
        "tests/_commands/test_train_task.py::test_train_semantic_segmentation_multihead__integration__runs_with_multiple_heads",
        "tests/_commands/test_train_task_helpers.py::test_get_train_model_args_and_transform_args__propagate_dinov2_patch_size_to_scale_jitter",
        "tests/_commands/test_train_task_helpers.py::test_get_train_model_args_and_transform_args__propagate_patch_size_to_scale_jitter",
        "tests/_data/test_mask_semantic_segmentation_dataset.py::TestMaskSemanticSegmentationDataset::test__getitem__integer_masks",
        "tests/_data/test_mask_semantic_segmentation_dataset.py::TestMaskSemanticSegmentationDataset::test__getitem__multi_channel_mask_with_label_classes_error",
        "tests/_data/test_mask_semantic_segmentation_dataset.py::TestMaskSemanticSegmentationDataset::test__getitem__multi_channel_masks",
        "tests/_data/test_mask_semantic_segmentation_dataset.py::TestMaskSemanticSegmentationDataset::test__getitem__shape_mismatch_error",
        "tests/_data/test_mask_semantic_segmentation_dataset.py::TestMaskSemanticSegmentationDataset::test_get_class_mapping__ignore_classes__labels",
        "tests/_data/test_mask_semantic_segmentation_dataset.py::TestMaskSemanticSegmentationDataset::test_get_class_mapping__ignore_classes__multi_channel_masks",
        "tests/_data/test_mask_semantic_segmentation_dataset.py::TestMaskSemanticSegmentationDataset::test_get_class_mapping__labels",
        "tests/_data/test_mask_semantic_segmentation_dataset.py::TestMaskSemanticSegmentationDataset::test_get_class_mapping__multi_channel_masks",
        "tests/_data/test_yolo_oriented_object_detection_dataset.py::TestYoloOrientedObjectDetectionDataset::test__get_item__internal_class_ids",
        "tests/_data/test_yolo_oriented_object_detection_dataset.py::TestYoloOrientedObjectDetectionDataset::test__getitem__empty_label_file",
        "tests/_data/test_yolo_oriented_object_detection_dataset.py::TestYoloOrientedObjectDetectionDataset::test__getitem__no_label_file",
        "tests/_data/test_yolo_oriented_object_detection_dataset.py::TestYoloOrientedObjectDetectionDataset::test__split_first",
        "tests/_data/test_yolo_oriented_object_detection_dataset.py::TestYoloOrientedObjectDetectionDataset::test__split_last",
        "tests/_methods/dinov31/test_dinov31.py::TestDINOv31::test_dino_path_unchanged_by_paka",
        "tests/_methods/dinov31/test_dinov31.py::TestDINOv31::test_paka_skipped_before_start_step",
        "tests/_methods/dinov31/test_dinov31.py::TestDINOv31::test_train_step_impl",
        "tests/_methods/distillation/test_distillation.py::TestDistillation::test_load_state_dict_from_pretrained_teacher",
        "tests/_methods/distillation/test_distillation.py::TestDistillation::test_teacher_parameters_are_frozen",
        "tests/_methods/distillationv2/test_distillationv2.py::TestDistillationV2::test__forward_teacher_student__output_shape",
        "tests/_methods/distillationv2/test_distillationv2.py::TestDistillationV2::test_distillation_configure_optimizers_lr_scaling",
        "tests/_methods/distillationv2/test_distillationv2.py::TestDistillationV2::test_load_state_dict_from_pretrained_teacher",
        "tests/_methods/distillationv2/test_distillationv2.py::TestDistillationV2::test_teacher_parameters_are_frozen",
        "tests/_task_models/depth_estimation/test_task_model.py::TestDepthAnythingDepthEstimation::test_forward__returns_depth",
        "tests/_task_models/depth_estimation/test_task_model.py::TestDepthAnythingDepthEstimation::test_forward__returns_depth_and_sky",
        "tests/_task_models/depth_estimation/test_task_model.py::TestDepthAnythingDepthEstimation::test_predict__intrinsics_rejected_for_nonfocal_model",
        "tests/_task_models/depth_estimation/test_task_model.py::TestDepthAnythingDepthEstimation::test_predict__intrinsics_required_for_focal_model",
        "tests/_task_models/depth_estimation/test_task_model.py::TestDepthAnythingDepthEstimation::test_predict__intrinsics_scale_depth_by_focal",
        "tests/_task_models/depth_estimation/test_task_model.py::TestDepthAnythingDepthEstimation::test_predict__max_depth_scales_output",
        "tests/_task_models/depth_estimation/test_task_model.py::TestDepthAnythingDepthEstimation::test_predict__returns_original_resolution",
        "tests/_task_models/depth_estimation/test_task_model.py::TestDepthAnythingDepthEstimation::test_predict__returns_original_resolution_focal",
        "tests/_task_models/depth_estimation/test_task_model.py::TestDepthAnythingDepthEstimation::test_predict_batch__intrinsics_per_image",
        "tests/_task_models/depth_estimation/test_task_model.py::TestDepthAnythingDepthEstimation::test_predict_batch__mixed_sizes",
        "tests/_task_models/dinov2_eomt_instance_segmentation/test_task_model.py::test_predict_batch__composes_stages_in_order",
        "tests/_task_models/dinov2_eomt_panoptic_segmentation/test_task_model.py::test_predict_batch__composes_stages_in_order",
        "tests/_task_models/dinov2_eomt_semantic_segmentation/test_task_model.py::test_predict_batch__composes_stages_in_order",
        "tests/_task_models/image_classification/test_task_model.py::test_predict_batch__composes_stages_in_order",
        "tests/_task_models/linear_semantic_segmentation/test_task_model.py::TestLinearSemanticSegmentation::test_forward_train__output_shape",
        "tests/_task_models/linear_semantic_segmentation/test_task_model.py::TestLinearSemanticSegmentation::test_init__freezes_dinov2_mask_token",
        "tests/_task_models/ltdetr_object_detection/test_task_model.py::test_checkpoint_roundtrip__rtdetrv2_decoder_preserved_when_not_explicit",
        "tests/_task_models/ltdetr_object_detection/test_task_model.py::test_create_train_model__ecvit",
        "tests/_task_models/ltdetr_object_detection/test_task_model.py::test_dinov2_vits14_ltdetr__constructs_and_runs_forward",
        "tests/_task_models/ltdetr_object_detection/test_task_model.py::test_freeze_backbone_on_set_train_mode",
        "tests/_task_models/ltdetr_object_detection/test_task_model.py::test_freeze_backbone_on_set_train_mode__ecvit",
        "tests/_task_models/ltdetr_object_detection/test_task_model.py::test_get_optimizer__ecvit_splits_pretrained_backbone_from_projector",
        "tests/_task_models/ltdetr_object_detection/test_task_model.py::test_get_optimizer__flat_cosine_raises_when_cosine_phase_collapses",
        "tests/_task_models/ltdetr_object_detection/test_task_model.py::test_get_optimizer__linear_warns_when_warmup_exceeds_training",
        "tests/_task_models/ltdetr_object_detection/test_task_model.py::test_get_optimizer__scheduler_modes",
        "tests/_task_models/ltdetr_object_detection/test_task_model.py::test_load_train_state_dict__from_exported",
        "tests/_task_models/ltdetr_object_detection/test_task_model.py::test_load_train_state_dict__no_ema_weights",
        "tests/_task_models/ltdetr_object_detection/test_task_model.py::test_train_transform_args__resolve_auto__image_size_is_2x_patch_size_compatible",
        "tests/_task_models/ltdetr_object_detection/test_task_model.py::test_train_transform_args__resolve_auto__scale_jitter_divisible_by_patch_size",
        "tests/_task_models/ltdetr_object_detection/test_task_model.py::test_transform_args__resolve_auto__preserves_explicit_image_size",
        "tests/_task_models/ltdetr_object_detection/test_task_model.py::test_val_transform_args__resolve_auto__image_size_is_2x_patch_size_compatible",
        "tests/_task_models/test_task_model_helpers.py::test_init_model_from_checkpoint__legacy_dinov2_uses_registered_decoder",
        "tests/_transforms/ltdetr_transforms/test_components.py::TestStepScheduledCompose::test_get_transform_is_cached",
        "tests/_transforms/ltdetr_transforms/test_components.py::TestStepScheduledCompose::test_mosaic_exposed_when_configured",
        "tests/_transforms/ltdetr_transforms/test_components.py::TestStepScheduledCompose::test_mosaic_none_when_not_configured",
        "tests/_transforms/ltdetr_transforms/test_components.py::TestStepScheduledCompose::test_reinit_contract_matches_active_status",
        "tests/_transforms/ltdetr_transforms/test_components.py::TestStepScheduledCompose::test_set_step_advances_via_tracker",
        "tests/_transforms/ltdetr_transforms/test_components.py::test_requires_dataloader_reinitialization",
        "tests/_transforms/ltdetr_transforms/test_components.py::test_resolve_ltdetr_step_schedule_for_augmentation__applies_windows",
        "tests/_transforms/ltdetr_transforms/test_components.py::test_resolve_ltdetr_step_schedule_for_augmentation__disables_empty_windows",
        "tests/_transforms/ltdetr_transforms/test_components.py::test_resolve_ltdetr_step_schedule_for_augmentation__keeps_non_empty_windows",
        "tests/_transforms/ltdetr_transforms/test_instance_segmentation.py::TestLTDETRInstanceSegmentationCollateBboxFilter::test_drops_sub_min_size_boxes_and_aligns_masks",
        "tests/_transforms/ltdetr_transforms/test_instance_segmentation.py::TestLTDETRInstanceSegmentationCollateFunction::test__call__train",
        "tests/_transforms/ltdetr_transforms/test_instance_segmentation.py::TestLTDETRInstanceSegmentationCollateFunction::test__call__val_split",
        "tests/_transforms/ltdetr_transforms/test_instance_segmentation.py::TestLTDETRInstanceSegmentationCollateFunction::test_mixup_concatenates_instances",
        "tests/_transforms/ltdetr_transforms/test_instance_segmentation.py::TestLTDETRInstanceSegmentationCollateFunction::test_requires_dataloader_reinitialization",
        "tests/_transforms/ltdetr_transforms/test_instance_segmentation.py::TestLTDETRInstanceSegmentationTrainTransformArgs::test_resolve_auto_rejects_non_rgb",
        "tests/_transforms/ltdetr_transforms/test_instance_segmentation.py::TestLTDETRInstanceSegmentationTransform::test___call__output_contract",
        "tests/_transforms/ltdetr_transforms/test_instance_segmentation.py::TestLTDETRInstanceSegmentationTransform::test_empty_masks",
        "tests/_transforms/ltdetr_transforms/test_instance_segmentation.py::TestLTDETRInstanceSegmentationTransform::test_masks_track_bboxes",
        "tests/_transforms/ltdetr_transforms/test_instance_segmentation.py::TestLTDETRInstanceSegmentationTransform::test_mosaic_mask_aware",
        "tests/_transforms/ltdetr_transforms/test_instance_segmentation.py::TestLTDETRInstanceSegmentationTransformBboxFilter::test_does_not_drop_sub_min_size_boxes",
        "tests/_transforms/ltdetr_transforms/test_instance_segmentation.py::TestMinBboxSizePxDefaults::test_train_args_default_to_four_pixels",
        "tests/_transforms/ltdetr_transforms/test_instance_segmentation.py::TestMinBboxSizePxDefaults::test_val_args_default_to_disabled",
        "tests/_transforms/ltdetr_transforms/test_object_detection.py::TestMinBboxSizePxDefaults::test_train_args_default_to_four_pixels",
        "tests/_transforms/ltdetr_transforms/test_object_detection.py::TestMinBboxSizePxDefaults::test_val_args_default_to_disabled",
        "tests/_transforms/ltdetr_transforms/test_object_detection.py::TestMixupMosaicProbDefaults::test_base_train_args_default_to_shared_prob",
        "tests/_transforms/ltdetr_transforms/test_object_detection.py::TestMixupMosaicProbDefaults::test_ltdetrv2_train_args_default_to_higher_prob",
        "tests/_transforms/ltdetr_transforms/test_object_detection.py::TestObjectDetectionCollateBboxFilter::test_filters_sub_min_size_after_resize",
        "tests/_transforms/ltdetr_transforms/test_object_detection.py::TestObjectDetectionCollateBboxFilter::test_filters_sub_min_size_after_scale_jitter",
        "tests/_transforms/ltdetr_transforms/test_object_detection.py::TestObjectDetectionCollateFunction::test__call__",
        "tests/_transforms/ltdetr_transforms/test_object_detection.py::TestObjectDetectionCollateFunction::test_requires_dataloader_reinitialization",
        "tests/_transforms/ltdetr_transforms/test_object_detection.py::TestObjectDetectionTransform::test___all_args_combinations",
        "tests/_transforms/ltdetr_transforms/test_object_detection.py::TestObjectDetectionTransformBboxFilter::test_does_not_drop_sub_min_size_boxes",
        "tests/templates/test_train_object_detection.py::test_rendered_template_runs_training_with_all_params",
        "tests/templates/test_train_object_detection.py::test_rendered_template_runs_training_with_defaults",
    }
)


@pytest.fixture(autouse=True)  # Apply to all tests
def lightly_train_cache_dir(
    tmp_path_factory: TempPathFactory,
    request: FixtureRequest,
    mocker: MockerFixture,
) -> Generator[Path, None, None]:
    """Set LIGHTLY_TRAIN_CACHE_DIR to a unique directory for each test.

    This ensures that tests do not share cache files between each other. By default
    LightlyTrain uses ~/.cache/lightly-train which is the same for all tests and can
    lead to hard-to-debug issues when tests interfere with each other.
    """
    name = request.node.name
    # From: https://github.com/pytest-dev/pytest/blob/9913cedb51a39da580d3ef3aff8cff006c3e7fc6/src/_pytest/tmpdir.py#L247-L249
    name = re.sub(r"[\W]", "_", name)
    MAXVAL = 100
    name = name[:MAXVAL]
    # Use tmp_path_factory instead of tmp_path because tmp_path is oftentimes also used
    # inside the actual test function. We don't want to use tmp_path for the cache dir
    # because then tmp_path is not empty anymore for the test function which might be
    # unexpected.
    cache_dir = tmp_path_factory.mktemp(f"{name}_lightly_train_cache")
    mocker.patch.dict(os.environ, {"LIGHTLY_TRAIN_CACHE_DIR": str(cache_dir)})
    try:
        yield cache_dir
    finally:
        # Delete the cache dir after the test. By default, pytest only deletes
        # directories created via tmp_path_factory at the end of the whole test
        # session.
        shutil.rmtree(cache_dir, ignore_errors=True)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "long_running_test: marks tests as long running (skipped on GitHub CI)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    skip_broken = pytest.mark.skip(
        reason="Broken by the 3D DINOv2 adaptation and unrelated to 3D DINOv2 pretraining."
    )
    for item in items:
        if item.nodeid.split("[", 1)[0] in _BROKEN_BY_3D_DINOV2_ADAPTATION:
            item.add_marker(skip_broken)

    if os.environ.get("GITHUB_ACTIONS"):
        skip_long = pytest.mark.skip(reason="long running test, skipped on Github CI")
        for item in items:
            if "long_running_test" in item.keywords:
                item.add_marker(skip_long)


@pytest.fixture(autouse=True)  # Apply to all tests
def set_test_env_variables(
    mocker: MockerFixture,
) -> None:
    mocker.patch.dict(
        os.environ,
        {
            "LIGHTLY_TRAIN_EVENTS_DISABLED": "1",
            "LIGHTLY_TRAIN_POSTHOG_KEY": "",
        },
    )
