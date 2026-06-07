import pyarrow as pa
import pyarrow.dataset as ds
import pytest
from arrowshader.data import to_arrow, from_parquet


def test_table_passthrough():
    t = pa.table({"x": [1, 2, 3]})
    assert to_arrow(t) is t


def test_dataset_passthrough():
    t = pa.table({"x": [1, 2, 3]})
    d = ds.dataset(t)
    assert to_arrow(d) is d


def test_pycapsule():
    t = pa.table({"x": [1, 2, 3]})
    reader = pa.RecordBatchReader.from_batches(t.schema, t.to_batches())
    result = to_arrow(reader)
    assert isinstance(result, pa.Table)
    assert result.num_rows == 3


def test_unsupported_type():
    with pytest.raises(TypeError, match="Unsupported"):
        to_arrow("not a table")


def test_from_parquet(tmp_path):
    path = tmp_path / "test.parquet"
    pa.parquet.write_table(pa.table({"x": [1, 2, 3]}), path)
    result = from_parquet(str(path))
    assert isinstance(result, ds.Dataset)


def test_polars():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"x": [1, 2, 3]})
    result = to_arrow(df)
    assert isinstance(result, pa.Table)
    assert result.num_rows == 3


def test_pandas():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = to_arrow(df)
    assert isinstance(result, pa.Table)
    assert result.num_rows == 3


def test_huggingface():
    datasets = pytest.importorskip("datasets")
    dset = datasets.Dataset.from_dict({"x": [1, 2, 3]})
    result = to_arrow(dset)
    assert isinstance(result, pa.Table)
    assert result.num_rows == 3
