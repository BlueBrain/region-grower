"""Synthesize the morphologies.

- launch TMD(?) synthesis in parallel
- write each synthesized morphology to a separate file
- assign morphology names to MVD3/sonata
- assign identity cell rotations to MVD3/sonata
- optional axon grafting "on-the-fly"
"""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of region-grower.
# See https://github.com/BlueBrain/region-grower for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from shutil import which
from typing import Optional

import dask.dataframe as dd
import dask.distributed
import morphio
import numpy as np
import pandas as pd
import yaml
from diameter_synthesis.validators import validate_model_params
from jsonschema import validate
from morph_tool.exceptions import NoDendriteException
from morphio.mut import Morphology
from neurots.utils import convert_from_legacy_neurite_type
from neurots.validator import validate_neuron_distribs
from neurots.validator import validate_neuron_params
from pkg_resources import resource_stream
from voxcell import RegionMap
from voxcell.cell_collection import CellCollection
from voxcell.nexus.voxelbrain import Atlas

from region_grower import RegionGrowerError
from region_grower import SkipSynthesisError
from region_grower.atlas_helper import AtlasHelper
from region_grower.context import CellState
from region_grower.context import ComputationParameters
from region_grower.context import SpaceContext
from region_grower.context import SpaceWorker
from region_grower.context import SynthesisParameters
from region_grower.morph_io import MorphLoader
from region_grower.morph_io import MorphWriter
from region_grower.utils import assign_morphologies
from region_grower.utils import check_na_morphologies
from region_grower.utils import load_morphology_list

LOGGER = logging.getLogger(__name__)

morphio.set_maximum_warnings(0)  # suppress MorphIO warnings on writing files

_SERIALIZED_COLUMNS = [
    "layer_depths",
    "orientation",
    "tmd_parameters",
    "tmd_distributions",
]


class RegionMapper:
    """Mapper between region acronyms and regions names in synthesis config files."""

    def __init__(self, synthesis_regions, region_map, known_regions=None):
        """Constructor.

        Args:
            synthesis_regions (list[str]): list of regions available from synthesis
            region_map (voxcell.RegionMap): RegionMap object related to hierarchy.json
            known_regions (list[str]): the list of all the known regions
        """
        self.region_map = region_map
        self.synthesis_regions = synthesis_regions
        self._mapper = {}
        self._inverse_mapper = {}
        for synthesis_region in self.synthesis_regions:
            for region_id in self.region_map.find(
                synthesis_region, attr="acronym", with_descendants=True
            ):
                region_acronym = self.region_map.get(region_id, "acronym")
                self._mapper[region_acronym] = synthesis_region
                if synthesis_region not in self._inverse_mapper:
                    self._inverse_mapper[synthesis_region] = set()
                self._inverse_mapper[synthesis_region].add(region_acronym)

        if known_regions is not None:
            self._inverse_mapper["default"] = set(
                sorted(set(known_regions).difference(self._mapper.keys()))
            )
        if "default" not in self._inverse_mapper:
            self._inverse_mapper["default"] = set()

    def __getitem__(self, key):
        """Make this class behave like a dict with a default value."""
        return self._mapper.get(key, "default")

    @property
    def mapper(self):
        """Access the internal mapper."""
        return self._mapper

    @property
    def inverse_mapper(self):
        """Access the internal inverse mapper."""
        return self._inverse_mapper


def _parallel_wrapper(
    row,
    computation_parameters,
    cortical_depths,
    rotational_jitter_std,
    scaling_jitter_std,
    min_hard_scale,
    tmd_parameters,
    tmd_distributions,
):
    # pylint: disable=too-many-locals
    try:
        current_cell = CellState(
            position=np.array([row["x"], row["y"], row["z"]]),
            orientation=np.array([row["orientation"]]),
            mtype=row["mtype"],
            depth=row["current_depth"],
        )
        current_space_context = SpaceContext(
            layer_depths=row["layer_depths"],
            cortical_depths=cortical_depths[row["synthesis_region"]],
        )

        axon_scale = row.get("axon_scale", None)
        if axon_scale is not None and np.isnan(axon_scale):
            axon_scale = None
        region = (
            row["synthesis_region"]
            if row["synthesis_region"] in tmd_distributions
            else row["region"]
        )
        current_synthesis_parameters = SynthesisParameters(
            tmd_distributions=tmd_distributions[region][row["mtype"]],
            tmd_parameters=tmd_parameters[region][row["mtype"]],
            axon_morph_name=row.get("axon_name", None),
            axon_morph_scale=axon_scale,
            rotational_jitter_std=rotational_jitter_std,
            scaling_jitter_std=scaling_jitter_std,
            recenter=True,
            seed=row["seed"],
            min_hard_scale=min_hard_scale,
        )
        space_worker = SpaceWorker(
            current_cell,
            current_space_context,
            current_synthesis_parameters,
            computation_parameters,
        )
        new_cell = space_worker.synthesize()
        res = space_worker.completion(new_cell)

        res["debug_infos"] = dict(space_worker.debug_infos)
    except (SkipSynthesisError, RegionGrowerError, NoDendriteException) as exc:  # pragma: no cover
        LOGGER.error("Skip %s because of the following error: %s", row.name, exc)
        res = {
            "name": None,
            "apical_points": None,
            "apical_sections": None,
            "apical_NRN_sections": None,
            "debug_infos": None,
        }
    return pd.Series(res)


class SynthesizeMorphologies:
    """Synthesize morphologies.

    The synthesis steps are the following:

    - load CellCollection
    - load and check TMD parameters / distributions
    - prepare morphology output folder
    - fetch atlas data
    - check axon morphology list
    - call TNS to synthesize each cell and write it to the output folder
    - write the global results (the new CellCollection and the apical points)

    Args:
        input_cells: the path to the MVD3/sonata file.
        tmd_parameters: the path to the JSON file containing the TMD parameters.
        tmd_distributions: the path to the JSON file containing the TMD distributions.
        atlas: the path to the Atlas directory.
        out_cells: the path to the MVD3/sonata file in which the properties of the synthesized
            cells are written.
        out_apical: the path to the YAML file in which the apical points are written.
        out_morph_dir: the path to the directory in which the synthesized morphologies are
            written.
        out_morph_ext: the file extensions used to write the synthesized morphologies.
        morph_axon: the path to the TSV file containing the name of the morphology that
            should be used to graft the axon on each synthesized morphology.
        base_morph_dir: the path containing the morphologies listed in the TSV file given in
            ``morph_axon``.
        atlas_cache: the path to the directory used for the atlas cache.
        seed: the starting seed to use (note that the GID of each cell is added to this seed
            to ensure all cells has different seeds).
        out_apical_nrn_sections: the path to the YAML file in which the apical section IDs
            used by Neuron are written.
        max_files_per_dir: the maximum number of file in each directory (will create
            subdirectories if needed).
        overwrite: if set to False, the directory given to ``out_morph_dir`` must be empty.
        max_drop_ratio: the maximum ratio that
        scaling_jitter_std: the std of the scaling jitter.
        rotational_jitter_std: the std of the rotational jitter.
        nb_processes: the number of processes when MPI is not used.
        with_mpi: initialize and use MPI when set to True.
        min_depth: minimum depth from atlas computation
        max_depth: maximum depth from atlas computation
        skip_write: set to True to bypass writing to disk for debugging/testing
        min_hard_scale: the scale value below which a neurite is removed
    """

    MAX_SYNTHESIS_ATTEMPTS_COUNT = 10
    NEW_COLUMNS = [
        "name",
        "apical_points",
        "apical_sections",
        "apical_NRN_sections",
        "debug_infos",
    ]

    def __init__(
        self,
        input_cells,
        tmd_parameters,
        tmd_distributions,
        atlas,
        out_cells,
        out_apical=None,
        out_morph_dir="out",
        out_morph_ext=None,
        morph_axon=None,
        base_morph_dir=None,
        atlas_cache=None,
        seed=0,
        out_apical_nrn_sections=None,
        max_files_per_dir=None,
        overwrite=False,
        max_drop_ratio=0,
        scaling_jitter_std=None,
        rotational_jitter_std=None,
        out_debug_data=None,
        nb_processes=None,
        with_mpi=False,
        min_depth=25,
        max_depth=5000,
        skip_write=False,
        min_hard_scale=0.2,
        region_structure=None,
        container_path=None,
        hide_progress_bar=False,
        dask_config=None,
        chunksize=None,
    ):  # pylint: disable=too-many-arguments, too-many-locals, too-many-statements
        self.seed = seed
        self.scaling_jitter_std = scaling_jitter_std
        self.rotational_jitter_std = rotational_jitter_std
        self.with_NRN_sections = out_apical_nrn_sections is not None
        if self.with_NRN_sections and not set(["asc", "swc"]).intersection(out_morph_ext):
            raise ValueError(
                """The 'out_morph_ext' parameter must contain one of ["asc", "swc"] when """
                f"'with_NRN_sections' is set to True (current value is {list(out_morph_ext)})."
            )
        self.out_apical_nrn_sections = out_apical_nrn_sections
        self.out_cells = out_cells
        self.out_apical = out_apical
        self.out_debug_data = out_debug_data
        self.min_hard_scale = min_hard_scale
        self.container_path = container_path
        self._progress_bar = not bool(hide_progress_bar)
        self.atlas = None
        self.with_mpi = with_mpi
        self.nb_processes = nb_processes
        self.dask_config = dask_config
        self.chunksize = chunksize
        self._parallel_client = None
        self._init_parallel(mpi_only=True)

        LOGGER.info(
            "Loading atlas from '%s' using the following cache dir: '%s' and the following "
            "region_structure file: '%s'",
            atlas,
            atlas_cache,
            region_structure,
        )
        # The atlas is loaded after the self._init_parallel() call so that when using MPI the atlas
        # is loaded only in the scheduler process
        self.atlas = AtlasHelper(
            Atlas.open(atlas, cache_dir=atlas_cache), region_structure_path=region_structure
        )

        LOGGER.info("Loading CellCollection from %s", input_cells)
        self.cells = CellCollection.load(input_cells)
        if self.cells.size() == 0:
            LOGGER.info("The CellCollection is empty, synthesis will create empty results")

        LOGGER.info("Loading TMD parameters from %s", tmd_parameters)
        with open(tmd_parameters, "r", encoding="utf-8") as f:
            self.tmd_parameters = convert_from_legacy_neurite_type(json.load(f))

        LOGGER.info("Loading TMD distributions from %s", tmd_distributions)
        with open(tmd_distributions, "r", encoding="utf-8") as f:
            self.tmd_distributions = convert_from_legacy_neurite_type(json.load(f))

        # Set default values to tmd_parameters and tmd_distributions
        self.set_default_params_and_distrs()

        self.regions = [r for r in self.atlas.region_structure if r != "default"]
        self.set_cortical_depths()

        LOGGER.info("Preparing morphology output folder in %s", out_morph_dir)
        self.morph_writer = MorphWriter(out_morph_dir, out_morph_ext or ["swc"], skip_write)
        self.morph_writer.prepare(
            num_files=len(self.cells.positions),
            max_files_per_dir=max_files_per_dir,
            overwrite=overwrite,
        )

        LOGGER.info("Preparing internal representation of cells")
        self.cells_data = self.cells.as_dataframe()
        self.cells_data.index -= 1  # Index must start from 0

        self.region_mapper = RegionMapper(
            self.regions,
            RegionMap.load_json(Path(atlas) / "hierarchy.json"),
            self.cells_data["region"].unique(),
        )
        self.cells_data["synthesis_region"] = self.cells_data["region"].apply(
            lambda region: self.region_mapper[region]
        )

        LOGGER.info("Checking TMD parameters and distributions according to cells mtypes")
        self.verify()

        LOGGER.info("Fetching atlas data from %s", atlas)
        self.assign_atlas_data(min_depth, max_depth)
        if morph_axon is not None:
            LOGGER.info("Loading axon morphologies from %s", morph_axon)
            self.axon_morph_list = load_morphology_list(morph_axon, self.task_ids)
            check_na_morphologies(
                self.axon_morph_list,
                mtypes=self.cells_data["mtype"],
                threshold=max_drop_ratio,
            )
            self.cells_data[["axon_name", "axon_scale"]] = self.axon_morph_list
            self.morph_loader = MorphLoader(base_morph_dir, file_ext="h5")
            to_compute = self._check_axon_morphology(self.cells_data)
            if to_compute is not None:  # pragma: no cover
                self.cells_data = self.cells_data.loc[to_compute]
        else:
            self.axon_morph_list = None
            self.morph_loader = None

    def set_cortical_depths(self):
        """Set cortical depths for all regions."""
        self.cortical_depths = {"default": None}
        for region in self.regions:
            if (
                region not in self.atlas.region_structure
                or self.atlas.region_structure[region]["thicknesses"] is None
            ):  # pragma: no cover
                self.cortical_depths[region] = self.cortical_depths["default"]
            else:
                self.cortical_depths[region] = np.cumsum(
                    list(self.atlas.region_structure[region]["thicknesses"].values())
                ).tolist()

    def set_default_params_and_distrs(self):
        """Set default values to all regions in tmd_parameters and tmd_distributions."""

        def set_default_values(data):
            if "default" in data:  # pragma: no cover
                for region in data:
                    if region == "default":
                        continue
                    for mtype, value in data["default"].items():
                        data[region].setdefault(mtype, value)

        set_default_values(self.tmd_parameters)
        set_default_values(self.tmd_distributions)

    def __del__(self):
        """Close the internal client when the object is deleted."""
        try:
            self._close_parallel()
        except Exception:  # pylint: disable=broad-except ; # pragma: no cover
            pass

    def _init_parallel(self, mpi_only=False):
        """Initialize MPI workers if required or get the number of available processes."""
        if self._parallel_client is not None:  # pragma: no cover
            return

        # Define a default configuration to disable some dask.distributed things
        default_dask_config = {
            "distributed": {
                "worker": {
                    "use_file_locking": False,
                    "memory": {
                        "target": False,
                        "spill": False,
                        "pause": 0.8,
                        "terminate": 0.95,
                    },
                    "profile": {
                        "enabled": False,
                        "interval": "10s",
                        "cycle": "10m",
                    },
                },
                "admin": {
                    "tick": {
                        "limit": "1h",
                    },
                },
            },
            "dataframe": {
                "convert_string": False,
            },
        }

        # Merge the default config with the existing config (keep conflicting values from defaults)
        dask_config = dask.config.merge(dask.config.config, default_dask_config)

        # Get temporary-directory from environment variables
        _TMP = os.environ.get("SHMDIR", None) or os.environ.get("TMPDIR", None)
        if _TMP is not None:
            dask_config["temporary-directory"] = _TMP

        # Merge the config with the one given as argument
        if self.dask_config is not None:
            dask_config = dask.config.merge(dask_config, self.dask_config)

        # Set the dask config
        dask.config.set(dask_config)

        if self.with_mpi:  # pragma: no cover
            # pylint: disable=import-outside-toplevel,import-error
            import dask_mpi
            from mpi4py import MPI

            dask_mpi.initialize(dashboard=False)
            comm = MPI.COMM_WORLD  # pylint: disable=c-extension-no-member
            self.nb_processes = comm.Get_size()
            client_kwargs = {}
            LOGGER.debug(
                "Initializing parallel workers using MPI (%s workers found)", self.nb_processes
            )
        elif mpi_only:
            return
        else:
            self.nb_processes = min(
                self.nb_processes if self.nb_processes is not None else os.cpu_count(),
                len(self.cells_data),
            )

            if self.with_NRN_sections:
                dask.config.set({"distributed.worker.daemon": False})
            client_kwargs = {
                "n_workers": self.nb_processes,
                "dashboard_address": None,
            }
            LOGGER.debug(
                "Initializing parallel workers using the following config: %s", client_kwargs
            )

        LOGGER.debug("Using the following dask configuration: %s", json.dumps(dask.config.config))

        # This is needed to make dask aware of the workers
        self._parallel_client = dask.distributed.Client(**client_kwargs)

    def _close_parallel(self):
        if self._parallel_client is not None:
            LOGGER.debug("Closing the Dask client")
            self._parallel_client.retire_workers()
            time.sleep(1)
            self._parallel_client.shutdown()
            self._parallel_client.close()
            self._parallel_client = None

    def assign_atlas_data(self, min_depth=25, max_depth=5000):
        """Open an Atlas and compute depths and orientations according to the given positions."""
        self.cells_data["current_depth"] = np.nan
        self.cells_data["layer_depths"] = pd.Series(
            np.nan, index=self.cells_data.index.copy(), dtype=object
        )
        for _region, regions in self.region_mapper.inverse_mapper.items():
            region_mask = self.cells_data.region.isin(regions)

            if not region_mask.any():  # pragma: no cover
                # If there is no cell in this region we can continue
                continue

            positions = self.cells.positions[region_mask]

            LOGGER.debug("Extract atlas data for %s region", _region)
            if (
                _region in self.atlas.regions
                and self.atlas.region_structure[_region].get("thicknesses", None) is not None
                and self.atlas.region_structure[_region].get("layers", None) is not None
            ):
                layers = self.atlas.layers[_region]
                thicknesses = [self.atlas.layer_thickness(layer) for layer in layers]
                depths = self.atlas.compute_region_depth(_region)
                layer_depths = self.atlas.get_layer_boundary_depths(
                    positions, thicknesses
                ).T.tolist()
                current_depths = np.clip(depths.lookup(positions), min_depth, max_depth)
            else:
                LOGGER.warning(
                    "We are not able to synthesize the region %s, we fallback to 'default' region",
                    _region,
                )
                layer_depths = None
                current_depths = None

            self.cells_data.loc[region_mask, "current_depth"] = current_depths
            self.cells_data.loc[region_mask, "layer_depths"] = pd.Series(
                data=layer_depths, index=self.cells_data.loc[region_mask].index, dtype=object
            )

            LOGGER.debug("Extract orientations data for %s region", _region)
            orientations = self.atlas.orientations.lookup(positions)
            self.cells_data.loc[region_mask, "orientation"] = pd.Series(
                data=orientations.tolist(),
                index=self.cells_data.loc[region_mask].index,
                dtype=object,
            )

    @property
    def task_ids(self):
        """Task IDs (= CellCollection IDs)."""
        return self.cells_data.index.values

    @staticmethod
    def _check_axon_morphology(cells_df) -> Optional[Morphology]:
        """Returns the name of the morphology corresponding to the given gid if found."""
        no_axon = cells_df["axon_name"].isnull()
        if no_axon.any():
            gids = no_axon.loc[no_axon].index.tolist()
            LOGGER.warning(
                "The following gids were not found in the axon morphology list: %s", gids
            )
            return no_axon.loc[~no_axon].index
        return None

    def check_context_consistency(self):
        """Check that the context_constraints entries in TMD parameters are consistent."""
        LOGGER.info("Check context consistency")
        region = "synthesis_region"
        if (
            self.cells_data.loc[0, "synthesis_region"] not in self.tmd_parameters
        ):  # pragma: no cover
            region = "region"

        has_context_constraints = self.cells_data.apply(
            lambda row: bool(
                self.tmd_parameters[row[region]][row["mtype"]].get("context_constraints", {})
            ),
            axis=1,
        ).rename("has_context_constraints")
        df = self.cells_data[["synthesis_region", "mtype"]].join(has_context_constraints)
        df["has_layers"] = df.apply(
            lambda row: row["synthesis_region"] in self.atlas.regions, axis=1
        )
        df["inconsistent_context"] = df.apply(
            lambda row: row["has_context_constraints"] and not row["has_layers"], axis=1
        )
        invalid_elements = df.loc[df["inconsistent_context"]]
        if not invalid_elements.empty:
            LOGGER.warning(
                "The morphologies with the following region/mtype couples have inconsistent "
                "context and constraints: %s",
                invalid_elements[["synthesis_region", "mtype"]].value_counts().index.tolist(),
            )

    def compute(self):
        """Run synthesis for all GIDs."""
        LOGGER.info("Prepare parameters")
        computation_parameters = ComputationParameters(
            morph_writer=self.morph_writer,
            morph_loader=self.morph_loader,
            with_NRN_sections=self.with_NRN_sections,
            retries=self.MAX_SYNTHESIS_ATTEMPTS_COUNT,
            debug_data=self.out_debug_data is not None,
        )
        self.cells_data["seed"] = (self.seed + self.cells_data.index) % (1 << 32)

        self.check_context_consistency()

        func_kwargs = {
            "computation_parameters": computation_parameters,
            "cortical_depths": self.cortical_depths,
            "rotational_jitter_std": self.rotational_jitter_std,
            "scaling_jitter_std": self.scaling_jitter_std,
            "min_hard_scale": self.min_hard_scale,
            "tmd_parameters": self.tmd_parameters,
            "tmd_distributions": self.tmd_distributions,
        }

        meta = pd.DataFrame({name: pd.Series(dtype="object") for name in self.NEW_COLUMNS})

        if self.nb_processes == 0:
            LOGGER.info("Start computation")
            computed = self.cells_data.apply(
                lambda row: _parallel_wrapper(row, **func_kwargs), axis=1
            )
        else:
            if self.chunksize is None or len(self.cells_data) <= self.chunksize:
                dd_kwargs = {"npartitions": self.nb_processes}
            else:
                dd_kwargs = {"chunksize": self.chunksize}
            LOGGER.info("Start parallel computation using %s", dd_kwargs)
            ddf = dd.from_pandas(self.cells_data, **dd_kwargs)
            future = ddf.apply(_parallel_wrapper, meta=meta, axis=1, **func_kwargs)
            future = future.persist()
            if self._progress_bar:
                dask.distributed.progress(future)
            computed = future.compute()

        LOGGER.info("Format results")
        res = self.cells_data.join(computed)
        return res

    def finalize(self, result: pd.DataFrame):
        """Finalize master work.

          - assign 'morphology' property based on workers' result
          - assign 'orientation' property to identity matrix
          - dump CellCollection to MVD3/sonata

        Args:
            result: A ``pandas.DataFrame``
        """
        LOGGER.info("Assigning CellCollection 'morphology' property...")

        assign_morphologies(self.cells, result["name"])

        LOGGER.info("Assigning CellCollection 'orientation' property...")
        # cell orientations are imbued in synthesized morphologies
        self.cells.orientations = np.broadcast_to(np.identity(3), (self.cells.size(), 3, 3))

        LOGGER.info("Export CellCollection to %s...", self.out_cells)
        self.cells.save(self.out_cells)

        def first_non_None(apical_points):
            """Returns the first non None apical coordinates."""
            for coord in apical_points:
                if coord is not None:  # pragma: no cover
                    return coord.tolist()
            return None  # pragma: no cover

        with_apicals = result.loc[~result["apical_points"].isnull()]
        if self.out_apical is not None:
            LOGGER.info("Export apical points to %s...", self.out_apical)
            with open(self.out_apical, "w", encoding="utf-8") as apical_file:
                apical_data = with_apicals[["name"]].join(
                    with_apicals["apical_points"].apply(first_non_None)
                )
                yaml.dump(apical_data.set_index("name")["apical_points"].to_dict(), apical_file)

        if self.out_apical_nrn_sections is not None:
            LOGGER.info("Export apical Neuron sections to %s...", self.out_apical_nrn_sections)
            with open(self.out_apical_nrn_sections, "w", encoding="utf-8") as apical_file:
                yaml.dump(
                    with_apicals[["name", "apical_NRN_sections"]]
                    .set_index("name")["apical_NRN_sections"]
                    .to_dict(),
                    apical_file,
                )

        if self.out_debug_data is not None:
            LOGGER.info("Export debug data to %s...", self.out_debug_data)
            result.to_pickle(self.out_debug_data)

        if self.container_path is not None:  # pragma: no cover
            # this needs at least module morpho-kit/0.3.6
            LOGGER.info("Containerizing morphologies to %s...", self.container_path)

            if which("morphokit_merge") is None:
                raise RuntimeError(
                    "The 'morphokit_merge' command is not available, please install the MorphoKit."
                )

            with subprocess.Popen(
                [
                    "morphokit_merge",
                    self.morph_writer.output_dir,
                    "--nodes",
                    self.out_cells,
                    "--output",
                    self.container_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env={"PATH": os.getenv("PATH", "")},
            ) as proc:
                LOGGER.debug(proc.communicate()[0].decode())

    def export_empty_results(self):
        """Create result DataFrame for empty population."""
        res = self.cells_data.join(
            pd.DataFrame(
                index=[],
                columns=self.NEW_COLUMNS,
                dtype=object,
            ),
        )

        LOGGER.info("Export CellCollection to %s...", self.out_cells)
        self.cells.save(self.out_cells)
        return res

    def synthesize(self):
        """Execute the complete synthesis process and export the results."""
        if self.cells_data.empty:
            LOGGER.warning("The population to synthesize is empty!")
            return self.export_empty_results()
        self._init_parallel()
        LOGGER.info("Start synthesis")
        res = self.compute()
        self.finalize(res)
        LOGGER.info("Synthesis complete")
        self._close_parallel()
        return res

    def verify(self) -> None:
        """Check that context has distributions / parameters for all given regions and mtypes."""
        with resource_stream("region_grower", "schemas/distributions.json") as distr_file:
            distributions_schema = json.load(distr_file)
        validate(self.tmd_distributions, distributions_schema)

        with resource_stream("region_grower", "schemas/parameters.json") as param_file:
            parameters_schema = json.load(param_file)
        validate(self.tmd_parameters, parameters_schema)

        for region in self.cells.properties["region"].unique():
            _region = self.region_mapper[region]
            if _region not in self.tmd_distributions:  # pragma: no cover
                _region = region

            for mtype in self.cells.properties[self.cells.properties["region"] == region][
                "mtype"
            ].unique():
                if mtype not in self.tmd_distributions[_region]:
                    error_msg = f"Missing distributions for mtype '{mtype}' in region '{_region}'"
                    raise RegionGrowerError(error_msg)
                if mtype not in self.tmd_parameters[_region]:
                    error_msg = f"Missing parameters for mtype '{mtype}' in region '{_region}'"
                    raise RegionGrowerError(error_msg)

                validate_neuron_distribs(self.tmd_distributions[_region][mtype])
                validate_neuron_params(self.tmd_parameters[_region][mtype])
                validate_model_params(self.tmd_parameters[_region][mtype]["diameter_params"])
