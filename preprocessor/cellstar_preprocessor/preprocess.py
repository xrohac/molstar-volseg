import asyncio
import logging
import shutil
from tracemalloc import start
import typing
from argparse import ArgumentError
from enum import Enum
from pathlib import Path
import time as times

import typer
import zarr
from cellstar_db.file_system.annotations_context import AnnnotationsEditContext
from cellstar_db.file_system.db import FileSystemVolumeServerDB
from cellstar_db.file_system.volume_and_segmentation_context import (
    VolumeAndSegmentationContext,
)
from cellstar_db.models import (
    DescriptionData,
    GeometricSegmentationData,
    SegmentAnnotationData,
)
from cellstar_preprocessor.flows.common import (
    open_json_file,
    open_zarr_structure_from_path,
    process_extra_data,
)
from cellstar_preprocessor.flows.constants import (
    GEOMETRIC_SEGMENTATIONS_ZATTRS,
    INIT_ANNOTATIONS_DICT,
    INIT_METADATA_DICT,
    LATTICE_SEGMENTATION_DATA_GROUPNAME,
    MESH_SEGMENTATION_DATA_GROUPNAME,
    RAW_GEOMETRIC_SEGMENTATION_INPUT_ZATTRS,
    VOLUME_DATA_GROUPNAME,
)
from cellstar_preprocessor.flows.segmentation.collect_custom_annotations import (
    collect_custom_annotations,
)
from cellstar_preprocessor.flows.segmentation.extract_annotations_from_geometric_segmentation import (
    extract_annotations_from_geometric_segmentation,
)
from cellstar_preprocessor.flows.segmentation.extract_annotations_from_sff_segmentation import (
    extract_annotations_from_sff_segmentation,
)
from cellstar_preprocessor.flows.segmentation.extract_metadata_from_mask import (
    extract_metadata_from_mask,
)
from cellstar_preprocessor.flows.segmentation.extract_metadata_from_nii_segmentation import (
    extract_metadata_from_nii_segmentation,
)
from cellstar_preprocessor.flows.segmentation.extract_metadata_from_sff_segmentation import (
    extract_metadata_from_sff_segmentation,
)
from cellstar_preprocessor.flows.segmentation.extract_metadata_geometric_segmentation import (
    extract_metadata_geometric_segmentation,
)
from cellstar_preprocessor.flows.segmentation.extract_ome_tiff_segmentation_annotations import (
    extract_ome_tiff_segmentation_annotations,
)
from cellstar_preprocessor.flows.segmentation.extract_ometiff_segmentation_metadata import (
    extract_ometiff_segmentation_metadata,
)
from cellstar_preprocessor.flows.segmentation.geometric_segmentation_preprocessing import (
    geometric_segmentation_preprocessing,
)
from cellstar_preprocessor.flows.segmentation.helper_methods import (
    check_if_omezarr_has_labels,
)
from cellstar_preprocessor.flows.segmentation.mask_annotation_creation import (
    mask_annotation_creation,
)
from cellstar_preprocessor.flows.segmentation.mask_segmentation_preprocessing import (
    mask_segmentation_preprocessing,
)
from cellstar_preprocessor.flows.segmentation.nii_segmentation_downsampling import (
    nii_segmentation_downsampling,
)
from cellstar_preprocessor.flows.segmentation.nii_segmentation_preprocessing import (
    nii_segmentation_preprocessing,
)
from cellstar_preprocessor.flows.segmentation.ome_zarr_labels_preprocessing import (
    ome_zarr_labels_preprocessing,
)
from cellstar_preprocessor.flows.segmentation.segmentation_downsampling import (
    sff_segmentation_downsampling,
)

from cellstar_preprocessor.flows.segmentation.sff_preprocessing import sff_preprocessing
from cellstar_preprocessor.flows.volume.extract_metadata_from_map import (
    extract_metadata_from_map,
)
from cellstar_preprocessor.flows.volume.extract_nii_metadata import extract_nii_metadata
from cellstar_preprocessor.flows.volume.extract_ome_tiff_image_annotations import (
    extract_ome_tiff_image_annotations,
)
from cellstar_preprocessor.flows.volume.extract_ometiff_image_metadata import (
    extract_ometiff_image_metadata,
)
from cellstar_preprocessor.flows.volume.extract_omezarr_annotations import (
    extract_omezarr_annotations,
)
from cellstar_preprocessor.flows.volume.extract_omezarr_metadata import (
    extract_ome_zarr_metadata,
)
from cellstar_preprocessor.flows.volume.map_preprocessing import map_preprocessing
from cellstar_preprocessor.flows.volume.nii_preprocessing import nii_preprocessing
from cellstar_preprocessor.flows.volume.ome_zarr_image_preprocessing import (
    ome_zarr_image_preprocessing,
)
from cellstar_preprocessor.flows.volume.ometiff_image_processing import (
    ometiff_image_processing,
)
from cellstar_preprocessor.flows.volume.ometiff_segmentation_processing import (
    ometiff_segmentation_processing,
)
from cellstar_preprocessor.flows.volume.process_allencel_metadata_csv import (
    process_allencell_metadata_csv,
)
from cellstar_preprocessor.flows.volume.quantize_internal_volume import (
    quantize_internal_volume,
)
from cellstar_preprocessor.flows.volume.volume_downsampling import volume_downsampling
from cellstar_preprocessor.flows.volume.volume_downsampling_gpu import volume_downsampling_gpu
from cellstar_preprocessor.flows.segmentation.mask_segmentation_preprocessing_gpu import (
    mask_segmentation_preprocessing_gpu,
)
from cellstar_preprocessor.flows.segmentation.segmentation_downsampling_gpu import (
    sff_segmentation_downsampling_gpu,
)

from cellstar_preprocessor.model.input import (
    DownsamplingParams,
    EntryData,
    InputKind,
    Inputs,
    PreprocessorInput,
    QuantizationDtype,
    StoringParams,
    VolumeParams,
)
from cellstar_preprocessor.model.segmentation import InternalSegmentation
from cellstar_preprocessor.model.volume import InternalVolume
from cellstar_preprocessor.tools.convert_app_specific_segm_to_sff.convert_app_specific_segm_to_sff import (
    convert_app_specific_segm_to_sff,
)
from pydantic import BaseModel
from typing_extensions import Annotated


class PreprocessorMode(str, Enum):
    add = "add"
    extend = "extend"


class InputT(BaseModel):
    input_path: Path


class OMETIFFImageInput(InputT):
    pass


class OMETIFFSegmentationInput(InputT):
    pass


class ExtraDataInput(InputT):
    pass


class MAPInput(InputT):
    pass


class SFFInput(InputT):
    pass


class OMEZARRInput(InputT):
    pass


class CustomAnnotationsInput(InputT):
    pass


class NIIVolumeInput(InputT):
    pass


class NIISegmentationInput(InputT):
    pass


class MaskInput(InputT):
    pass


class GeometricSegmentationInput(InputT):
    pass


class TaskBase(typing.Protocol):
    def execute(self) -> None: ...


class CustomAnnotationsCollectionTask(TaskBase):
    # NOTE: for this to work, custom annotations json must contain only the keys that
    # need to be updated
    def __init__(
        self, input_path: Path, intermediate_zarr_structure_path: Path
    ) -> None:
        self.input_path = input_path
        self.intermediate_zarr_structure_path = intermediate_zarr_structure_path

    def execute(self) -> None:
        collect_custom_annotations(
            self.input_path, self.intermediate_zarr_structure_path
        )


class QuantizeInternalVolumeTask(TaskBase):
    def __init__(self, internal_volume: InternalVolume):
        self.internal_volume = internal_volume

    def execute(self) -> None:
        quantize_internal_volume(internal_volume=self.internal_volume)


# class SaveAnnotationsTask(TaskBase):
#     def __init__(self, intermediate_zarr_structure_path: Path):
#         self.intermediate_zarr_structure_path = intermediate_zarr_structure_path

#     def execute(self) -> None:
#         root = open_zarr_structure_from_path(self.intermediate_zarr_structure_path)
#         save_dict_to_json_file(
#             root.attrs["annotations_dict"],
#             ANNOTATION_METADATA_FILENAME,
#             self.intermediate_zarr_structure_path,
#         )


# class SaveMetadataTask(TaskBase):
#     def __init__(self, intermediate_zarr_structure_path: Path):
#         self.intermediate_zarr_structure_path = intermediate_zarr_structure_path

#     def execute(self) -> None:
#         root = open_zarr_structure_from_path(self.intermediate_zarr_structure_path)
#         save_dict_to_json_file(
#             root.attrs["metadata_dict"],
#             GRID_METADATA_FILENAME,
#             self.intermediate_zarr_structure_path,
#         )

# class SaveGeometricSegmentationSets(TaskBase):
#     def __init__(self, intermediate_zarr_structure_path: Path):
#         self.intermediate_zarr_structure_path = intermediate_zarr_structure_path

#     def execute(self) -> None:
#         root = open_zarr_structure_from_path(self.intermediate_zarr_structure_path)
#         save_dict_to_json_file(
#             root.attrs[GEOMETRIC_SEGMENTATIONS_ZATTRS],
#             GEOMETRIC_SEGMENTATION_FILENAME,
#             self.intermediate_zarr_structure_path,
#         )


class SFFAnnotationCollectionTask(TaskBase):
    def __init__(self, internal_segmentation: InternalSegmentation):
        self.internal_segmentation = internal_segmentation

    def execute(self) -> None:
        annotations_dict = extract_annotations_from_sff_segmentation(
            internal_segmentation=self.internal_segmentation
        )


class MaskAnnotationCreationTask(TaskBase):
    def __init__(self, internal_segmentation: InternalSegmentation):
        self.internal_segmentation = internal_segmentation

    def execute(self) -> None:
        # annotations_dict = extract_annotations_from_sff_segmentation(
        #     internal_segmentation=self.internal_segmentation
        # )
        mask_annotation_creation(internal_segmentation=self.internal_segmentation)


class NIIMetadataCollectionTask(TaskBase):
    def __init__(self, internal_volume: InternalVolume):
        self.internal_volume = internal_volume

    def execute(self) -> None:
        volume = self.internal_volume
        metadata_dict = extract_nii_metadata(internal_volume=volume)


class MAPMetadataCollectionTask(TaskBase):
    def __init__(self, internal_volume: InternalVolume):
        self.internal_volume = internal_volume

    def execute(self) -> None:
        volume = self.internal_volume
        metadata_dict = extract_metadata_from_map(internal_volume=volume)


class OMEZARRAnnotationsCollectionTask(TaskBase):
    def __init__(self, internal_volume: InternalVolume):
        self.internal_volume = internal_volume

    def execute(self) -> None:
        annotations_dict = extract_omezarr_annotations(
            internal_volume=self.internal_volume
        )


class OMEZARRMetadataCollectionTask(TaskBase):
    def __init__(self, internal_volume: InternalVolume):
        self.internal_volume = internal_volume

    def execute(self) -> None:
        metadata_dict = extract_ome_zarr_metadata(internal_volume=self.internal_volume)


class OMEZARRImageProcessTask(TaskBase):
    def __init__(self, internal_volume: InternalVolume):
        self.internal_volume = internal_volume

    def execute(self) -> None:
        ome_zarr_image_preprocessing(self.internal_volume)


class OMEZARRLabelsProcessTask(TaskBase):
    def __init__(self, internal_segmentation: InternalSegmentation):
        self.internal_segmentation = internal_segmentation

    def execute(self) -> None:
        ome_zarr_labels_preprocessing(internal_segmentation=self.internal_segmentation)


class SFFMetadataCollectionTask(TaskBase):
    def __init__(self, internal_segmentation: InternalSegmentation):
        self.internal_segmentation = internal_segmentation

    def execute(self) -> None:
        metadata_dict = extract_metadata_from_sff_segmentation(
            internal_segmentation=self.internal_segmentation
        )


class MaskMetadataCollectionTask(TaskBase):
    def __init__(self, internal_segmentation: InternalSegmentation):
        self.internal_segmentation = internal_segmentation

    def execute(self) -> None:
        # metadata_dict = extract_metadata_from_sff_segmentation(
        #     internal_segmentation=self.internal_segmentation
        # )
        metadata_dict = extract_metadata_from_mask(
            internal_segmentation=self.internal_segmentation
        )


class GeometricSegmentationMetadataCollectionTask(TaskBase):
    def __init__(self, internal_segmentation: InternalSegmentation):
        self.internal_segmentation = internal_segmentation

    def execute(self) -> None:
        metadata_dict = extract_metadata_geometric_segmentation(
            internal_segmentation=self.internal_segmentation
        )


class NIISegmentationMetadataCollectionTask(TaskBase):
    def __init__(self, internal_segmentation: InternalSegmentation):
        self.internal_segmentation = internal_segmentation

    def execute(self) -> None:
        metadata_dict = extract_metadata_from_nii_segmentation(
            internal_segmentation=self.internal_segmentation
        )


class MAPProcessVolumeTask(TaskBase):
    def __init__(self, internal_volume: InternalVolume):
        self.internal_volume = internal_volume

    def execute(self) -> None:
        volume = self.internal_volume
        map_preprocessing(volume)
        # volume_downsampling(volume)
        volume_downsampling_gpu(volume)


class NIIProcessVolumeTask(TaskBase):
    def __init__(self, internal_volume: InternalVolume):
        self.internal_volume = internal_volume

    def execute(self) -> None:
        volume = self.internal_volume

        nii_preprocessing(volume)
        # in processing part do
        volume_downsampling(volume)


class OMETIFFImageProcessingTask(TaskBase):
    def __init__(self, internal_volume: InternalVolume):
        self.internal_volume = internal_volume

    def execute(self) -> None:
        volume = self.internal_volume
        ometiff_image_processing(internal_volume=volume)
        volume_downsampling(internal_volume=volume)


class OMETIFFSegmentationProcessingTask(TaskBase):
    def __init__(self, internal_segmentation: InternalSegmentation):
        self.internal_segmentation = internal_segmentation

    def execute(self) -> None:
        segmentation = self.internal_segmentation
        ometiff_segmentation_processing(internal_segmentation=segmentation)
        sff_segmentation_downsampling(segmentation)


class OMETIFFImageMetadataExtractionTask(TaskBase):
    def __init__(self, internal_volume: InternalVolume):
        self.internal_volume = internal_volume

    def execute(self) -> None:
        volume = self.internal_volume
        extract_ometiff_image_metadata(internal_volume=volume)


class OMETIFFSegmentationMetadataExtractionTask(TaskBase):
    def __init__(self, internal_segmentation: InternalSegmentation):
        self.internal_segmentation = internal_segmentation

    def execute(self) -> None:
        internal_segmentation = self.internal_segmentation
        extract_ometiff_segmentation_metadata(
            internal_segmentation=internal_segmentation
        )


class OMETIFFImageAnnotationsExtractionTask(TaskBase):
    def __init__(self, internal_volume: InternalVolume):
        self.internal_volume = internal_volume

    def execute(self) -> None:
        volume = self.internal_volume
        extract_ome_tiff_image_annotations(internal_volume=volume)


class OMETIFFSegmentationAnnotationsExtractionTask(TaskBase):
    def __init__(self, internal_segmentation: InternalSegmentation):
        self.internal_segmentation = internal_segmentation

    def execute(self) -> None:
        internal_segmentation = self.internal_segmentation
        extract_ome_tiff_segmentation_annotations(
            internal_segmentation=internal_segmentation
        )


class ProcessExtraDataTask(TaskBase):
    def __init__(self, path: Path, intermediate_zarr_structure_path: Path):
        self.path = path
        self.intermediate_zarr_structure = intermediate_zarr_structure_path

    def execute(self) -> None:
        process_extra_data(self.path, self.intermediate_zarr_structure)


class AllencellMetadataCSVProcessingTask(TaskBase):
    def __init__(
        self, path: Path, cell_id: int, intermediate_zarr_structure_path: Path
    ):
        self.path = path
        self.cell_id = cell_id
        self.intermediate_zarr_structure = intermediate_zarr_structure_path

    def execute(self) -> None:
        process_allencell_metadata_csv(
            self.path, self.cell_id, self.intermediate_zarr_structure
        )


class NIIProcessSegmentationTask(TaskBase):
    def __init__(self, internal_segmentation: InternalSegmentation):
        self.internal_segmentation = internal_segmentation

    def execute(self) -> None:
        segmentation = self.internal_segmentation

        nii_segmentation_preprocessing(internal_segmentation=segmentation)

        nii_segmentation_downsampling(internal_segmentation=segmentation)


class SFFProcessSegmentationTask(TaskBase):
    def __init__(self, internal_segmentation: InternalSegmentation):
        self.internal_segmentation = internal_segmentation

    def execute(self) -> None:
        segmentation = self.internal_segmentation

        sff_preprocessing(segmentation)

        sff_segmentation_downsampling(segmentation)


class MaskProcessSegmentationTask(TaskBase):
    def __init__(self, internal_segmentation: InternalSegmentation):
        self.internal_segmentation = internal_segmentation

    def execute(self) -> None:
        segmentation = self.internal_segmentation
        # mask_segmentation_preprocessing(internal_segmentation=segmentation)
        mask_segmentation_preprocessing_gpu(internal_segmentation=segmentation)
        # sff_segmentation_downsampling(segmentation)
        sff_segmentation_downsampling_gpu(segmentation)

class ProcessGeometricSegmentationTask(TaskBase):
    def __init__(self, internal_segmentation: InternalSegmentation):
        self.internal_segmentation = internal_segmentation

    def execute(self) -> None:
        segmentation = self.internal_segmentation

        geometric_segmentation_preprocessing(internal_segmentation=segmentation)


class GeometricSegmentationAnnotationsCollectionTask(TaskBase):
    def __init__(self, internal_segmentation: InternalSegmentation):
        self.internal_segmentation = internal_segmentation

    def execute(self) -> None:
        segmentation = self.internal_segmentation

        extract_annotations_from_geometric_segmentation(
            internal_segmentation=segmentation
        )


class Preprocessor:
    def __init__(self, preprocessor_input: PreprocessorInput):
        if not preprocessor_input:
            raise ArgumentError("No input parameters are provided")
        self.preprocessor_input = preprocessor_input
        self.intermediate_zarr_structure = None
        self.internal_volume = None
        self.internal_segmentation = None

    def store_internal_volume(self, internal_volume: InternalVolume):
        self.internal_volume = internal_volume

    def get_internal_volume(self):
        return self.internal_volume

    def store_internal_segmentation(self, internal_segmentation: InternalSegmentation):
        self.internal_segmentation = internal_segmentation

    def get_internal_segmentation(self):
        return self.internal_segmentation

    def _process_inputs(self, inputs: list[InputT]) -> list[TaskBase]:
        tasks = []
        nii_segmentation_inputs: list[NIISegmentationInput] = []
        mask_segmentation_inputs: list[MaskInput] = []
        for i in inputs:
            if isinstance(i, ExtraDataInput):
                tasks.append(
                    ProcessExtraDataTask(
                        path=i.input_path,
                        intermediate_zarr_structure_path=self.intermediate_zarr_structure,
                    )
                )
            elif isinstance(i, MAPInput):
                self.store_internal_volume(
                    internal_volume=InternalVolume(
                        intermediate_zarr_structure_path=self.intermediate_zarr_structure,
                        volume_input_path=i.input_path,
                        params_for_storing=self.preprocessor_input.storing_params,
                        volume_force_dtype=self.preprocessor_input.volume.force_volume_dtype,
                        quantize_dtype_str=self.preprocessor_input.volume.quantize_dtype_str,
                        downsampling_parameters=self.preprocessor_input.downsampling,
                        entry_data=self.preprocessor_input.entry_data,
                        quantize_downsampling_levels=self.preprocessor_input.volume.quantize_downsampling_levels,
                    )
                )
                tasks.append(
                    MAPProcessVolumeTask(internal_volume=self.get_internal_volume())
                )
                tasks.append(
                    MAPMetadataCollectionTask(
                        internal_volume=self.get_internal_volume()
                    )
                )
            elif isinstance(i, SFFInput):
                self.store_internal_segmentation(
                    internal_segmentation=InternalSegmentation(
                        intermediate_zarr_structure_path=self.intermediate_zarr_structure,
                        segmentation_input_path=i.input_path,
                        params_for_storing=self.preprocessor_input.storing_params,
                        downsampling_parameters=self.preprocessor_input.downsampling,
                        entry_data=self.preprocessor_input.entry_data,
                    )
                )
                tasks.append(
                    SFFProcessSegmentationTask(
                        internal_segmentation=self.get_internal_segmentation()
                    )
                )
                tasks.append(
                    SFFMetadataCollectionTask(
                        internal_segmentation=self.get_internal_segmentation()
                    )
                )
                tasks.append(
                    SFFAnnotationCollectionTask(
                        internal_segmentation=self.get_internal_segmentation()
                    )
                )

            elif isinstance(i, MaskInput):
                mask_segmentation_inputs.append(i)

            elif isinstance(i, OMEZARRInput):
                self.store_internal_volume(
                    internal_volume=InternalVolume(
                        intermediate_zarr_structure_path=self.intermediate_zarr_structure,
                        volume_input_path=i.input_path,
                        params_for_storing=self.preprocessor_input.storing_params,
                        volume_force_dtype=self.preprocessor_input.volume.force_volume_dtype,
                        quantize_dtype_str=self.preprocessor_input.volume.quantize_dtype_str,
                        downsampling_parameters=self.preprocessor_input.downsampling,
                        entry_data=self.preprocessor_input.entry_data,
                        quantize_downsampling_levels=self.preprocessor_input.volume.quantize_downsampling_levels,
                    )
                )
                tasks.append(OMEZARRImageProcessTask(self.get_internal_volume()))
                if check_if_omezarr_has_labels(
                    internal_volume=self.get_internal_volume()
                ):
                    self.store_internal_segmentation(
                        internal_segmentation=InternalSegmentation(
                            intermediate_zarr_structure_path=self.intermediate_zarr_structure,
                            segmentation_input_path=i.input_path,
                            params_for_storing=self.preprocessor_input.storing_params,
                            downsampling_parameters=self.preprocessor_input.downsampling,
                            entry_data=self.preprocessor_input.entry_data,
                        )
                    )
                    tasks.append(
                        OMEZARRLabelsProcessTask(self.get_internal_segmentation())
                    )

                tasks.append(
                    OMEZARRMetadataCollectionTask(
                        internal_volume=self.get_internal_volume()
                    )
                )
                tasks.append(
                    OMEZARRAnnotationsCollectionTask(self.get_internal_volume())
                )

            elif isinstance(i, GeometricSegmentationInput):
                self.store_internal_segmentation(
                    internal_segmentation=InternalSegmentation(
                        intermediate_zarr_structure_path=self.intermediate_zarr_structure,
                        segmentation_input_path=i.input_path,
                        params_for_storing=self.preprocessor_input.storing_params,
                        downsampling_parameters=self.preprocessor_input.downsampling,
                        entry_data=self.preprocessor_input.entry_data,
                    )
                )
                tasks.append(
                    ProcessGeometricSegmentationTask(self.get_internal_segmentation())
                )
            elif isinstance(i, OMETIFFImageInput):
                self.store_internal_volume(
                    internal_volume=InternalVolume(
                        intermediate_zarr_structure_path=self.intermediate_zarr_structure,
                        volume_input_path=i.input_path,
                        params_for_storing=self.preprocessor_input.storing_params,
                        volume_force_dtype=self.preprocessor_input.volume.force_volume_dtype,
                        quantize_dtype_str=self.preprocessor_input.volume.quantize_dtype_str,
                        downsampling_parameters=self.preprocessor_input.downsampling,
                        entry_data=self.preprocessor_input.entry_data,
                        quantize_downsampling_levels=self.preprocessor_input.volume.quantize_downsampling_levels,
                    )
                )
                tasks.append(
                    OMETIFFImageProcessingTask(
                        internal_volume=self.get_internal_volume()
                    )
                )
                tasks.append(
                    OMETIFFImageMetadataExtractionTask(
                        internal_volume=self.get_internal_volume()
                    )
                )
                # TODO: remove - after processing segmentation
                tasks.append(
                    OMETIFFImageAnnotationsExtractionTask(
                        internal_volume=self.get_internal_volume()
                    )
                )
            elif isinstance(i, OMETIFFSegmentationInput):
                self.store_internal_segmentation(
                    internal_segmentation=InternalSegmentation(
                        intermediate_zarr_structure_path=self.intermediate_zarr_structure,
                        segmentation_input_path=i.input_path,
                        params_for_storing=self.preprocessor_input.storing_params,
                        downsampling_parameters=self.preprocessor_input.downsampling,
                        entry_data=self.preprocessor_input.entry_data,
                    )
                )
                tasks.append(
                    OMETIFFSegmentationProcessingTask(self.get_internal_segmentation())
                )
                tasks.append(
                    OMETIFFSegmentationMetadataExtractionTask(
                        internal_segmentation=self.get_internal_segmentation()
                    )
                )
                tasks.append(
                    OMETIFFSegmentationAnnotationsExtractionTask(
                        internal_segmentation=self.get_internal_segmentation()
                    )
                )
            elif isinstance(i, NIIVolumeInput):
                self.store_internal_volume(
                    internal_volume=InternalVolume(
                        intermediate_zarr_structure_path=self.intermediate_zarr_structure,
                        volume_input_path=i.input_path,
                        params_for_storing=self.preprocessor_input.storing_params,
                        volume_force_dtype=self.preprocessor_input.volume.force_volume_dtype,
                        quantize_dtype_str=self.preprocessor_input.volume.quantize_dtype_str,
                        downsampling_parameters=self.preprocessor_input.downsampling,
                        entry_data=self.preprocessor_input.entry_data,
                        quantize_downsampling_levels=self.preprocessor_input.volume.quantize_downsampling_levels,
                    )
                )
                tasks.append(
                    NIIProcessVolumeTask(internal_volume=self.get_internal_volume())
                )
                tasks.append(
                    NIIMetadataCollectionTask(
                        internal_volume=self.get_internal_volume()
                    )
                )

            elif isinstance(i, NIISegmentationInput):
                nii_segmentation_inputs.append(i)
            elif isinstance(i, CustomAnnotationsInput):
                tasks.append(
                    CustomAnnotationsCollectionTask(
                        input_path=i.input_path,
                        intermediate_zarr_structure_path=self.intermediate_zarr_structure,
                    )
                )

        if (
            self.get_internal_volume()
            and self.preprocessor_input.volume.quantize_dtype_str
        ):
            tasks.append(
                QuantizeInternalVolumeTask(internal_volume=self.get_internal_volume())
            )

        if nii_segmentation_inputs:
            nii_segmentation_input_paths = [
                i.input_path for i in nii_segmentation_inputs
            ]
            self.store_internal_segmentation(
                internal_segmentation=InternalSegmentation(
                    intermediate_zarr_structure_path=self.intermediate_zarr_structure,
                    segmentation_input_path=nii_segmentation_input_paths,
                    params_for_storing=self.preprocessor_input.storing_params,
                    downsampling_parameters=self.preprocessor_input.downsampling,
                    entry_data=self.preprocessor_input.entry_data,
                )
            )
            tasks.append(
                NIIProcessSegmentationTask(
                    internal_segmentation=self.get_internal_segmentation()
                )
            )
            tasks.append(
                NIISegmentationMetadataCollectionTask(
                    internal_segmentation=self.get_internal_segmentation()
                )
            )

        if mask_segmentation_inputs:
            mask_segmentation_input_paths = [
                i.input_path for i in mask_segmentation_inputs
            ]
            self.store_internal_segmentation(
                internal_segmentation=InternalSegmentation(
                    intermediate_zarr_structure_path=self.intermediate_zarr_structure,
                    segmentation_input_path=mask_segmentation_input_paths,
                    params_for_storing=self.preprocessor_input.storing_params,
                    downsampling_parameters=self.preprocessor_input.downsampling,
                    entry_data=self.preprocessor_input.entry_data,
                )
            )
            tasks.append(
                MaskProcessSegmentationTask(
                    internal_segmentation=self.get_internal_segmentation()
                )
            )
            tasks.append(
                MaskMetadataCollectionTask(
                    internal_segmentation=self.get_internal_segmentation()
                )
            )

            tasks.append(
                MaskAnnotationCreationTask(
                    internal_segmentation=self.get_internal_segmentation()
                )
            )

        if any(isinstance(i, GeometricSegmentationInput) for i in inputs):
            # tasks.append(SaveGeometricSegmentationSets(self.intermediate_zarr_structure))
            tasks.append(
                GeometricSegmentationAnnotationsCollectionTask(
                    self.get_internal_segmentation()
                )
            )
            tasks.append(
                GeometricSegmentationMetadataCollectionTask(
                    self.get_internal_segmentation()
                )
            )

        # tasks.append(SaveMetadataTask(self.intermediate_zarr_structure))
        # tasks.append(SaveAnnotationsTask(self.intermediate_zarr_structure))

        return tasks

    def _execute_tasks(self, tasks: list[TaskBase]):
        for task in tasks:
            task.execute()

    def __check_if_inputs_exists(self, raw_inputs_list: list[tuple[Path, InputKind]]):
        for input_item in raw_inputs_list:
            p = input_item[0]
            assert p.exists(), f'Input file {p} does not exist'

    def _analyse_preprocessor_input(self) -> list[InputT]:
        raw_inputs_list = self.preprocessor_input.inputs.files
        analyzed_inputs: list[InputT] = []

        self.__check_if_inputs_exists(raw_inputs_list)
        
        for input_item in raw_inputs_list:
            if input_item[1] == InputKind.extra_data:
                analyzed_inputs.append(ExtraDataInput(input_path=input_item[0]))
            elif input_item[1] == InputKind.map:
                analyzed_inputs.append(MAPInput(input_path=input_item[0]))
            elif input_item[1] == InputKind.sff:
                analyzed_inputs.append(SFFInput(input_path=input_item[0]))
            elif input_item[1] == InputKind.mask:
                analyzed_inputs.append(MaskInput(input_path=input_item[0]))
            elif input_item[1] == InputKind.omezarr:
                analyzed_inputs.append(OMEZARRInput(input_path=input_item[0]))
            elif input_item[1] == InputKind.geometric_segmentation:
                analyzed_inputs.append(
                    GeometricSegmentationInput(input_path=input_item[0])
                )
            elif input_item[1] == InputKind.custom_annotations:
                analyzed_inputs.append(CustomAnnotationsInput(input_path=input_item[0]))
            elif input_item[1] == InputKind.application_specific_segmentation:
                sff_path = convert_app_specific_segm_to_sff(input_item[0])
                analyzed_inputs.append(SFFInput(input_path=sff_path))
                # TODO: remove app specific segm file?
            elif input_item[1] == InputKind.nii_volume:
                analyzed_inputs.append(NIIVolumeInput(input_path=input_item[0]))
            elif input_item[1] == InputKind.nii_segmentation:
                analyzed_inputs.append(NIISegmentationInput(input_path=input_item[0]))
            elif input_item[1] == InputKind.ometiff_image:
                analyzed_inputs.append(OMETIFFImageInput(input_path=input_item[0]))
            elif input_item[1] == InputKind.ometiff_segmentation:
                analyzed_inputs.append(
                    OMETIFFSegmentationInput(input_path=input_item[0])
                )
            else:
                raise Exception('Input kind is not recognized')


        return analyzed_inputs

    async def entry_exists(self):
        new_db_path = Path(self.preprocessor_input.db_path)
        if new_db_path.is_dir() == False:
            new_db_path.mkdir(parents=True)

        db = FileSystemVolumeServerDB(new_db_path, store_type="zip")

        exists = await db.contains(
            namespace=self.preprocessor_input.entry_data.source_db,
            key=self.preprocessor_input.entry_data.entry_id,
        )

        return exists

    async def initialization(self, mode: PreprocessorMode):
        self.intermediate_zarr_structure = (
            self.preprocessor_input.working_folder
            / self.preprocessor_input.entry_data.entry_id
        )
        try:
            # delete previous intermediate zarr structure
            shutil.rmtree(self.intermediate_zarr_structure, ignore_errors=True)
            assert (
                self.intermediate_zarr_structure.exists() == False
            ), f"intermediate_zarr_structure: {self.intermediate_zarr_structure} already exists"
            store: zarr.storage.DirectoryStore = zarr.DirectoryStore(
                str(self.intermediate_zarr_structure)
            )
            root = zarr.group(store=store)

            # first initialize metadata and annotations dicts as empty
            # or as dicts read from db if mode is "extend"
            if mode == PreprocessorMode.extend:
                new_db_path = Path(self.preprocessor_input.db_path)
                db = FileSystemVolumeServerDB(new_db_path, store_type="zip")
                volume_metadata = await db.read_metadata(
                    self.preprocessor_input.entry_data.source_db,
                    self.preprocessor_input.entry_data.entry_id,
                )
                root.attrs["metadata_dict"] = volume_metadata.json_metadata()
                root.attrs["annotations_dict"] = await db.read_annotations(
                    self.preprocessor_input.entry_data.source_db,
                    self.preprocessor_input.entry_data.entry_id,
                )

            elif mode == PreprocessorMode.add:
                root.attrs["metadata_dict"] = INIT_METADATA_DICT

                root.attrs["annotations_dict"] = INIT_ANNOTATIONS_DICT
            else:
                raise Exception("Preprocessor mode is not supported")
            # init GeometricSegmentationData in zattrs
            root.attrs[GEOMETRIC_SEGMENTATIONS_ZATTRS] = []
            root.attrs[RAW_GEOMETRIC_SEGMENTATION_INPUT_ZATTRS] = {}

        except Exception as e:
            logging.error(e, stack_info=True, exc_info=True)
            raise e

        # self._analyse_preprocessor_input()

    def preprocessing(self):
        inputs = self._analyse_preprocessor_input()
        tasks = self._process_inputs(inputs)
        self._execute_tasks(tasks)
        return

    def store_to_db(self, mode: PreprocessorMode):
        new_db_path = Path(self.preprocessor_input.db_path)
        if new_db_path.is_dir() == False:
            new_db_path.mkdir()

        db = FileSystemVolumeServerDB(new_db_path, store_type="zip")

        # call it once and get context
        # get segmentation_ids from metadata
        # using its method
        root = open_zarr_structure_from_path(self.intermediate_zarr_structure)

        segmentation_lattice_ids = []
        segmentation_mesh_ids = []
        geometric_segmentation_ids = []

        if LATTICE_SEGMENTATION_DATA_GROUPNAME in root:
            segmentation_lattice_ids = list(
                root[LATTICE_SEGMENTATION_DATA_GROUPNAME].group_keys()
            )
        if MESH_SEGMENTATION_DATA_GROUPNAME in root:
            segmentation_mesh_ids = list(
                root[MESH_SEGMENTATION_DATA_GROUPNAME].group_keys()
            )
        if GEOMETRIC_SEGMENTATIONS_ZATTRS in root.attrs:
            geometric_segm_attrs: list[GeometricSegmentationData] = root.attrs[
                GEOMETRIC_SEGMENTATIONS_ZATTRS
            ]
            geometric_segmentation_ids = [
                g["segmentation_id"] for g in geometric_segm_attrs
            ]

        with db.edit_data(
            namespace=self.preprocessor_input.entry_data.source_db,
            key=self.preprocessor_input.entry_data.entry_id,
            working_folder=self.preprocessor_input.working_folder,
        ) as db_edit_context:
            db_edit_context: VolumeAndSegmentationContext
            # adding volume
            if VOLUME_DATA_GROUPNAME in root:
                db_edit_context.add_volume()
            # adding segmentations
            for id in segmentation_lattice_ids:
                db_edit_context.add_segmentation(id=id, kind="lattice")
            for id in segmentation_mesh_ids:
                db_edit_context.add_segmentation(id=id, kind="mesh")
            for id in geometric_segmentation_ids:
                db_edit_context.add_segmentation(id=id, kind="geometric_segmentation")
        if mode == PreprocessorMode.add:
            print(f"Entry {self.preprocessor_input.entry_data.entry_id} stored to the database")
        else:
            print(f"Entry {self.preprocessor_input.entry_data.entry_id} in the database was expanded")


async def main_preprocessor(
    mode: PreprocessorMode,
    quantize_dtype_str: typing.Optional[QuantizationDtype],
    quantize_downsampling_levels: typing.Optional[str],
    force_volume_dtype: typing.Optional[str],
    max_size_per_downsampling_lvl_mb: typing.Optional[float],
    min_downsampling_level: typing.Optional[int],
    max_downsampling_level: typing.Optional[int],
    remove_original_resolution: bool,
    entry_id: str,
    source_db: str,
    source_db_id: str,
    source_db_name: str,
    working_folder: str,
    db_path: str,
    input_paths: list[str],
    input_kinds: list[InputKind],
    min_size_per_downsampling_lvl_mb: typing.Optional[float] = 5.0,
):
    if quantize_downsampling_levels:
        quantize_downsampling_levels = quantize_downsampling_levels.split(" ")
        quantize_downsampling_levels = tuple(
            [int(level) for level in quantize_downsampling_levels]
        )

    preprocessor_input = PreprocessorInput(
        inputs=Inputs(files=[]),
        volume=VolumeParams(
            quantize_dtype_str=quantize_dtype_str,
            quantize_downsampling_levels=quantize_downsampling_levels,
            force_volume_dtype=force_volume_dtype,
        ),
        downsampling=DownsamplingParams(
            min_size_per_downsampling_lvl_mb=min_size_per_downsampling_lvl_mb,
            max_size_per_downsampling_lvl_mb=max_size_per_downsampling_lvl_mb,
            min_downsampling_level=min_downsampling_level,
            max_downsampling_level=max_downsampling_level,
            remove_original_resolution=remove_original_resolution,
        ),
        entry_data=EntryData(
            entry_id=entry_id,
            source_db=source_db,
            source_db_id=source_db_id,
            source_db_name=source_db_name,
        ),
        working_folder=Path(working_folder),
        storing_params=StoringParams(),
        db_path=Path(db_path),
    )

    for input_path, input_kind in zip(input_paths, input_kinds):
        preprocessor_input.inputs.files.append((Path(input_path), input_kind))

    preprocessor = Preprocessor(preprocessor_input)
    if mode == PreprocessorMode.add:
        if await preprocessor.entry_exists():
            raise Exception(
                f"Entry {preprocessor_input.entry_data.entry_id} from {preprocessor_input.entry_data.source_db} source already exists in database {preprocessor_input.db_path}"
            )
    else:
        if not await preprocessor.entry_exists():
            raise Exception(
                f"Entry {preprocessor_input.entry_data.entry_id} from {preprocessor_input.entry_data.source_db} source does not exist in database {preprocessor_input.db_path}"
            )
        assert mode == PreprocessorMode.extend, "Preprocessor mode is not supported"

    await preprocessor.initialization(mode=mode)
    preprocessor.preprocessing()
    preprocessor.store_to_db(mode)


app = typer.Typer()


# NOTE: works as adding, i.e. if entry already has volume/segmentation
# it will not add anything, throwing error instead (group exists in destination)
@app.command("preprocess")
def main(
    mode: PreprocessorMode = PreprocessorMode.add.value,
    quantize_dtype_str: Annotated[
        typing.Optional[QuantizationDtype], typer.Option(None)
    ] = None,
    quantize_downsampling_levels: Annotated[
        typing.Optional[str], typer.Option(None, help="Space-separated list of numbers")
    ] = None,
    force_volume_dtype: Annotated[typing.Optional[str], typer.Option(None)] = None,
    max_size_per_downsampling_lvl_mb: Annotated[
        typing.Optional[float], typer.Option(None)
    ] = None,
    min_size_per_downsampling_lvl_mb: Annotated[
        typing.Optional[float], typer.Option(None)
    ] = 5.0,
    min_downsampling_level: Annotated[typing.Optional[int], typer.Option(None)] = None,
    max_downsampling_level: Annotated[typing.Optional[int], typer.Option(None)] = None,
    remove_original_resolution: Annotated[
        typing.Optional[bool], typer.Option(None)
    ] = False,
    entry_id: str = typer.Option(default=...),
    source_db: str = typer.Option(default=...),
    source_db_id: str = typer.Option(default=...),
    source_db_name: str = typer.Option(default=...),
    working_folder: str = typer.Option(default=...),
    db_path: str = typer.Option(default=...),
    input_path: list[str] = typer.Option(default=...),
    input_kind: list[InputKind] = typer.Option(default=...),
    # add_segmentation_to_entry: bool = typer.Option(default=False),
    # add_custom_annotations: bool = typer.Option(default=False),
):
    asyncio.run(
        main_preprocessor(
            mode=mode,
            entry_id=entry_id,
            source_db=source_db,
            source_db_id=source_db_id,
            source_db_name=source_db_name,
            working_folder=working_folder,
            db_path=db_path,
            input_paths=input_path,
            input_kinds=input_kind,
            quantize_dtype_str=quantize_dtype_str,
            quantize_downsampling_levels=quantize_downsampling_levels,
            force_volume_dtype=force_volume_dtype,
            max_size_per_downsampling_lvl_mb=max_size_per_downsampling_lvl_mb,
            min_size_per_downsampling_lvl_mb=min_size_per_downsampling_lvl_mb,
            min_downsampling_level=min_downsampling_level,
            max_downsampling_level=max_downsampling_level,
            remove_original_resolution=remove_original_resolution,
            # add_segmentation_to_entry=add_segmentation_to_entry,
            # add_custom_annotations=add_custom_annotations
        )
    )


@app.command("delete")
def delete_entry(
    entry_id: str = typer.Option(default=...),
    source_db: str = typer.Option(default=...),
    db_path: str = typer.Option(default=...),
):
    print(f"Deleting db entry: {entry_id} {source_db}")
    new_db_path = Path(db_path)
    if new_db_path.is_dir() == False:
        new_db_path.mkdir()

    db = FileSystemVolumeServerDB(new_db_path, store_type="zip")
    asyncio.run(db.delete(namespace=source_db, key=entry_id))


@app.command("remove-volume")
def remove_volume(
    entry_id: str = typer.Option(default=...),
    source_db: str = typer.Option(default=...),
    db_path: str = typer.Option(default=...),
    working_folder: str = typer.Option(default=...),
):
    print(f"Deleting volumes for entry: {entry_id} {source_db}")
    new_db_path = Path(db_path)
    if new_db_path.is_dir() == False:
        new_db_path.mkdir()

    db = FileSystemVolumeServerDB(new_db_path, store_type="zip")

    with db.edit_data(
        namespace=source_db, key=entry_id, working_folder=Path(working_folder)
    ) as db_edit_context:
        db_edit_context: VolumeAndSegmentationContext
        db_edit_context.remove_volume()


@app.command("remove-segmentation")
def remove_segmentation(
    entry_id: str = typer.Option(default=...),
    source_db: str = typer.Option(default=...),
    id: str = typer.Option(default=...),
    kind: str = typer.Option(default=...),
    db_path: str = typer.Option(default=...),
    working_folder: str = typer.Option(default=...),
):
    print(f"Deleting segmentation for entry: {entry_id} {source_db}")
    new_db_path = Path(db_path)
    if new_db_path.is_dir() == False:
        new_db_path.mkdir()

    db = FileSystemVolumeServerDB(new_db_path, store_type="zip")

    with db.edit_data(
        namespace=source_db, key=entry_id, working_folder=Path(working_folder)
    ) as db_edit_context:
        db_edit_context: VolumeAndSegmentationContext
        db_edit_context.remove_segmentation(id=id, kind=kind)


@app.command("remove-segment-annotations")
def remove_segment_annotations(
    entry_id: str = typer.Option(default=...),
    source_db: str = typer.Option(default=...),
    id: list[str] = typer.Option(default=...),
    db_path: str = typer.Option(default=...),
):
    print(f"Deleting annotation for entry: {entry_id} {source_db}")
    new_db_path = Path(db_path)
    if new_db_path.is_dir() == False:
        new_db_path.mkdir()

    db = FileSystemVolumeServerDB(new_db_path, store_type="zip")

    with db.edit_annotations(
        namespace=source_db, key=entry_id
    ) as db_edit_annotations_context:
        db_edit_annotations_context: AnnnotationsEditContext
        asyncio.run(db_edit_annotations_context.remove_segment_annotations(ids=id))


@app.command("remove-descriptions")
def remove_descriptions(
    entry_id: str = typer.Option(default=...),
    source_db: str = typer.Option(default=...),
    id: list[str] = typer.Option(default=...),
    db_path: str = typer.Option(default=...),
):
    print(f"Deleting descriptions for entry: {entry_id} {source_db}")
    new_db_path = Path(db_path)
    if new_db_path.is_dir() == False:
        new_db_path.mkdir()

    db = FileSystemVolumeServerDB(new_db_path, store_type="zip")

    with db.edit_annotations(
        namespace=source_db, key=entry_id
    ) as db_edit_annotations_context:
        db_edit_annotations_context: AnnnotationsEditContext
        asyncio.run(db_edit_annotations_context.remove_descriptions(ids=id))


@app.command("edit-segment-annotations")
def edit_segment_annotations(
    entry_id: str = typer.Option(default=...),
    source_db: str = typer.Option(default=...),
    # id: list[str] = typer.Option(default=...),
    data_json_path: str = typer.Option(default=...),
    db_path: str = typer.Option(default=...),
):
    # print(f"Deleting descriptions for entry: {entry_id} {source_db}")
    new_db_path = Path(db_path)
    if new_db_path.is_dir() == False:
        new_db_path.mkdir()

    db = FileSystemVolumeServerDB(new_db_path, store_type="zip")

    parsedSegmentAnnotations: list[SegmentAnnotationData] = open_json_file(
        Path(data_json_path)
    )

    with db.edit_annotations(
        namespace=source_db, key=entry_id
    ) as db_edit_annotations_context:
        db_edit_annotations_context: AnnnotationsEditContext
        asyncio.run(
            db_edit_annotations_context.add_or_modify_segment_annotations(
                parsedSegmentAnnotations
            )
        )


@app.command("edit-descriptions")
def edit_descriptions(
    entry_id: str = typer.Option(default=...),
    source_db: str = typer.Option(default=...),
    # id: list[str] = typer.Option(default=...),
    data_json_path: str = typer.Option(default=...),
    db_path: str = typer.Option(default=...),
):
    # print(f"Deleting descriptions for entry: {entry_id} {source_db}")
    new_db_path = Path(db_path)
    if new_db_path.is_dir() == False:
        new_db_path.mkdir()

    db = FileSystemVolumeServerDB(new_db_path, store_type="zip")

    parsedDescriptionData: list[DescriptionData] = open_json_file(Path(data_json_path))

    with db.edit_annotations(
        namespace=source_db, key=entry_id
    ) as db_edit_annotations_context:
        db_edit_annotations_context: AnnnotationsEditContext
        asyncio.run(
            db_edit_annotations_context.add_or_modify_descriptions(
                parsedDescriptionData
            )
        )


if __name__ == "__main__":
    # solutions how to run it async - two last https://github.com/tiangolo/typer/issues/85
    # currently using last one
    # typer.run(main)

    # could try https://github.com/tiangolo/typer/issues/88#issuecomment-1732469681
    app()


# NOTE: for testing:
# python preprocessor/preprocessor/preprocess.py --input-path temp/v2_temp_static_entry_files_dir/idr/idr-6001247/6001247.zarr --input-kind omezarr
# python preprocessor/preprocessor/preprocess.py --input-path test-data/preprocessor/sample_volumes/emdb_sff/EMD-1832.map --input-kind map --input-path test-data/preprocessor/sample_segmentations/emdb_sff/emd_1832.hff --input-kind sff
