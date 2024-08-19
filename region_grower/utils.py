"""Utils module."""
import json
import logging
import os
import socket
import time
from collections import defaultdict

import dask
import dask.distributed
import numpy as np
import pandas as pd
from dask.distributed import LocalCluster
from voxcell.math_utils import angles_to_matrices

try:  # pragma: no cover
    import dask_mpi
    from mpi4py import MPI

    HAS_MPI = True
except ImportError:
    HAS_MPI = False

LOGGER = logging.getLogger(__name__)


class MissingMpiError(RuntimeError):
    """Exception for missing MPI libraries."""

    default_msg = (
        "The MPI libraries are not installed, please install them using the following "
        """command: 'pip install "region-grower[mpi]"'"""
    )

    def __init__(self, *args, msg=None, **kwargs):
        if msg is None:  # pragma: no cover
            msg = self.default_msg
        super().__init__(msg, *args, **kwargs)


def setup_logger(level="info", prefix="", suffix="", set_worker_prefix=False):
    """Setup application logger."""
    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    level = str(level).lower()

    if set_worker_prefix:
        try:
            worker_name = dask.distributed.get_worker().id
        except Exception:
            worker_name = ""
        if HAS_MPI:  # pragma: no cover
            comm = MPI.COMM_WORLD  # pylint: disable=c-extension-no-member
            if comm.Get_size() > 1:
                worker_name = f"#{comm.Get_rank()}"
        name = socket.gethostname()
        pid = os.getpid()
        prefix = f"{worker_name} ({pid}@{name}) - {prefix}"

    logging.basicConfig(
        format=prefix + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + suffix,
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=levels[level],
    )

    if levels[level] >= logging.INFO:  # pragma: no cover
        logging.getLogger("distributed").level = max(
            logging.getLogger("distributed").level, logging.WARNING
        )


class NumpyEncoder(json.JSONEncoder):
    """To encode numpy arrays."""

    def default(self, o):  # pylint: disable=method-hidden
        """Actual encoder."""
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        return json.JSONEncoder.default(self, o)  # pragma: no cover


def cols_from_json(df, cols):
    """Transform the given columns to Python objects from JSON strings."""
    df = df.copy(deep=False)
    for col in cols:
        null_mask = ~df[col].isnull()
        df.loc[null_mask, col] = df.loc[null_mask, col].map(json.loads)
    return df


def create_morphologies_dict(dat_file, morph_path, ext=".asc"):
    """Create dict to load the morphologies from a directory, with dat file."""
    morph_name = pd.read_csv(dat_file, sep=" ", dtype={0: object})
    name_dict = defaultdict(list)
    for morph in morph_name.values:
        name_dict[morph[2]].append(os.path.join(morph_path, str(morph[0]) + ext))
    return name_dict


def random_rotation_y(n, rng=np.random):
    """Random rotation around Y-axis.

    Args:
        n: number of rotation matrices to generate

    Returns:
        n x 3 x 3 NumPy array with rotation matrices.
    """
    # copied from `brainbuilder.cell_orientations` to avoid a heavy dependency
    # consider reusing `brainbuilder` methods if we need something more general
    # (like user-defined distributions for rotation angles)
    angles = rng.uniform(-np.pi, np.pi, size=n)
    return angles_to_matrices(angles, axis="y")


def load_morphology_list(filepath, check_gids=None):
    """Read morphology list from tab-separated file."""
    result = pd.read_csv(
        filepath, sep=r"\s+", index_col=0, dtype={"morphology": object, "scale": float}
    )
    result.loc[result["morphology"].isnull(), "morphology"] = None
    if "scale" not in result:
        result["scale"] = None
    if check_gids is not None:
        if sorted(result.index) != sorted(check_gids):
            raise RuntimeError("Morphology list GIDs mismatch")
    return result


def _failure_ratio_by_mtype(mtypes, na_mask):
    """Calculate ratio of N/A occurrences per mtype."""
    failed = mtypes.loc[na_mask].value_counts()
    overall = mtypes.value_counts()
    result = (
        pd.DataFrame(
            {
                "N/A": failed,
                "out of": overall,
            }
        )
        .dropna()
        .astype(int)
    )
    result["ratio, %"] = 100.0 * result["N/A"] / result["out of"]
    result.sort_values("ratio, %", ascending=False, inplace=True)
    return result


def check_na_morphologies(morph_list, mtypes, threshold=None):
    """Check N/A ratio per mtype."""
    na_mask = morph_list["morphology"].isnull()
    if na_mask.any():
        stats = _failure_ratio_by_mtype(mtypes, na_mask)
        LOGGER.warning("N/A morphologies for %d position(s)", np.count_nonzero(na_mask))
        LOGGER.info("N/A ratio by mtypes:\n%s", stats.to_string(float_format="%.1f"))
        if threshold is not None:
            exceeded = 0.01 * stats["ratio, %"] > threshold
            if exceeded.any():
                raise RuntimeError(
                    f"Max N/A ratio ({100.0 * threshold:.1f}%) exceeded for mtype(s): "
                    f"{', '.join(exceeded[exceeded].index)}"
                )


def assign_morphologies(cells, morphologies):
    """Assign morphologies to CellCollection.

    Args:
        cells: CellCollection to be augmented
        morphologies: dictionary {gid -> morphology_name}

    No return value; `cells` is input/output argument.
    """
    cells.properties["morphology"] = pd.Series(morphologies)
    na_mask = cells.properties["morphology"].isnull()
    if na_mask.any():
        LOGGER.info(
            "Dropping %d cells with no morphologies assigned and reindexing...",
            np.count_nonzero(na_mask),
        )
        cells.remove_unassigned_cells()

    cells.properties["morphology_producer"] = "synthesis"


def initialize_parallel_client(
    nb_processes=None, dask_config=None, no_daemon=False, with_mpi=False
):
    """Initialize dask Client.

    Use MPI workers if required or use the given number of processes.
    """
    if not with_mpi and nb_processes is None:
        return None, None

    # Define a default configuration to disable some dask.distributed things
    default_dask_config = {
        "distributed": {
            "scheduler": {
                "work-stealing-interval": "1000ms",
                "worker-saturation": 1,
            },
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
    }

    # Merge the default config with the existing config (keep conflicting values from defaults)
    new_dask_config = dask.config.merge(dask.config.config, default_dask_config)

    # Get temporary-directory from environment variables
    _TMP = os.environ.get("SHMDIR", None) or os.environ.get("TMPDIR", None)
    if _TMP is not None:
        new_dask_config["temporary-directory"] = _TMP

    # Merge the config with the one given as argument
    if dask_config is not None:
        new_dask_config = dask.config.merge(new_dask_config, dask_config)

    # Set the dask config
    dask.config.set(new_dask_config)

    if with_mpi:  # pragma: no cover
        if not HAS_MPI:
            raise MissingMpiError()
        dask_mpi.initialize()
        comm = MPI.COMM_WORLD  # pylint: disable=c-extension-no-member
        # The number of processes is computed considering that the two first processes are the main
        # process and the scheduler so they can not used as workers
        nb_processes = comm.Get_size() - 2
        LOGGER.debug("Initializing parallel workers using MPI (%s workers found)", nb_processes)
        parallel_client = dask.distributed.Client()
    else:
        nb_processes = nb_processes if nb_processes > 0 else os.cpu_count()

        if no_daemon:
            dask.config.set({"distributed.worker.daemon": False})
        LOGGER.debug("Initializing parallel workers")
        cluster = LocalCluster(n_workers=nb_processes)
        parallel_client = dask.distributed.Client(cluster)

    LOGGER.debug("Using the following dask configuration: %s", json.dumps(dask.config.config))

    # This is needed to make dask aware of the workers (this object must exist until the end of
    # the computation)

    parallel_client.run(
        setup_logger,
        level=logging.getLevelName(LOGGER.getEffectiveLevel()),
        set_worker_prefix=True,
    )
    LOGGER.info(
        "The Dask dashboard is reachable on %s:%s",
        parallel_client.run_on_scheduler(socket.gethostname),
        parallel_client.scheduler_info()["services"]["dashboard"],
    )

    LOGGER.debug(
        "Initialized Dask Client with %s workers",
        nb_processes,
    )

    return parallel_client, nb_processes


def close_parallel_client(parallel_client):
    """Close the given dask Client object."""
    if parallel_client is not None:
        LOGGER.debug("Closing the Dask client")
        try:
            parallel_client.retire_workers()
            time.sleep(1)
        except Exception:  # pragma: no cover
            pass
        try:
            parallel_client.close(timeout="10s")
            parallel_client.shutdown()
        except Exception:  # pragma: no cover
            pass
