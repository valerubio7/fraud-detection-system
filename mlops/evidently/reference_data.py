from pathlib import Path

import pandas as pd


def load_reference_dataset(run_id: str, tracking_uri: str, dst_dir: str = "/tmp/evidently_ref") -> pd.DataFrame:
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)
    client.download_artifacts(run_id, "reference_dataset.parquet", str(dst_path))
    return pd.read_parquet(dst_path / "reference_dataset.parquet")
