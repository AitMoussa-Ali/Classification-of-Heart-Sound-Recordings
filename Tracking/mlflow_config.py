import mlflow
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)

def set_project_experiment(name: str) -> None:
    experiments_dir = PROJECT_ROOT / "Experiments" / name
    experiments_dir.mkdir(parents=True, exist_ok=True)

    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        mlflow.create_experiment(
            name,
            artifact_location=experiments_dir.as_uri(),
        )

    mlflow.set_experiment(name)

    return experiments_dir