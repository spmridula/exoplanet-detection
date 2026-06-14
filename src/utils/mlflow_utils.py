import os
import logging
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

# Experiment name — all runs go under this umbrella
EXPERIMENT_NAME = "exoplanet-detection"


def setup_mlflow(tracking_uri: str = "mlruns") -> None:
    """Initialize MLflow. Call once at the top of a training script."""
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
        logger.info(f"MLflow tracking: {tracking_uri} | experiment: {EXPERIMENT_NAME}")
    except ImportError:
        logger.warning("MLflow not installed — run: pip install mlflow")


@contextmanager
def mlflow_run(run_name: str, params: dict = None):
    """
    Context manager that wraps a training run in an MLflow run.
    Logs params on entry, handles exceptions cleanly.

    Usage
    -----
        with mlflow_run("xgboost_v1", params={"lr": 0.05, "depth": 6}) as run:
            model.fit(X_train, y_train)
            log_metrics(model.evaluate(X_val, y_val))
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("MLflow not available — running without tracking")
        yield None
        return

    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params(params)
        logger.info(f"MLflow run started: {run_name} (id={run.info.run_id[:8]})")
        try:
            yield run
        except Exception as e:
            mlflow.set_tag("error", str(e))
            raise


def log_metrics(metrics: dict) -> None:
    """Log a metrics dict to the active MLflow run."""
    try:
        import mlflow
        mlflow.log_metrics(metrics)
    except Exception:
        pass


def log_model(model: Any, artifact_name: str) -> None:
    """Log a pickled model to the active MLflow run."""
    try:
        import mlflow
        import pickle, tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(model, f)
            tmp_path = f.name
        mlflow.log_artifact(tmp_path, artifact_path=artifact_name)
        os.unlink(tmp_path)
    except Exception as e:
        logger.warning(f"Could not log model artifact: {e}")


def log_figure(fig, filename: str) -> None:
    """Log a matplotlib figure to the active MLflow run."""
    try:
        import mlflow
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fig.savefig(f.name, dpi=120, bbox_inches="tight")
            tmp_path = f.name
        mlflow.log_artifact(tmp_path, artifact_path="figures")
        os.unlink(tmp_path)
    except Exception as e:
        logger.warning(f"Could not log figure: {e}")