import pyarrow as pa
import pyarrow.dataset as ds

try:
    from datasets import Dataset as _HFDataset
except ImportError:
    _HFDataset = None


def to_arrow(data) -> ds.Dataset | pa.Table:
    if isinstance(data, ds.Dataset):
        return data

    if isinstance(data, pa.Table):
        return data

    if hasattr(data, "__arrow_c_stream__"):
        return pa.table(data)

    if _HFDataset is not None and isinstance(data, _HFDataset):
        return data.data.table

    raise TypeError(f"Unsupported source type: {type(data).__name__}")


def from_parquet(path: str) -> ds.Dataset:
    return ds.dataset(path, format="parquet")
