import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from pyarrow import acero

from arrowshader.reductions import Count, Sum, Mean, Min, Max, Std, Var, First, Last, Any


_ACERO_FUNCTIONS = {
    Sum: ("hash_sum", "sum"),
    Mean: ("hash_mean", "mean"),
    Min: ("hash_min", "min"),
    Max: ("hash_max", "max"),
    Std: ("hash_stddev", "std"),
    Var: ("hash_variance", "var"),
    First: ("hash_first", "first"),
    Last: ("hash_last", "last"),
    Any: ("hash_any", "any"),
}


class Acero:
    def __init__(self, batch_readahead=4, fragment_readahead=2):
        self._scan_options = {
            "batch_readahead": batch_readahead,
            "fragment_readahead": fragment_readahead,
        }

    def aggregate(self, source, x_col, y_col, width, height, x_range, y_range, reduction):
        bin_id, mask = self._bin_id(x_col, y_col, width, height, x_range, y_range)
        columns = [x_col, y_col]
        if type(reduction) in _ACERO_FUNCTIONS:
            columns.append(reduction.column)
        plan = acero.Declaration.from_sequence(
            [
                self._source(source, columns=columns, scan_filter=mask),
                acero.Declaration("filter", acero.FilterNodeOptions(mask)),
                acero.Declaration("project", acero.ProjectNodeOptions(*self._projections(reduction, bin_id))),
                acero.Declaration(
                    "aggregate", acero.AggregateNodeOptions(self._aggregates(reduction), keys=["bin_id"])
                ),
            ]
        )
        return plan.to_table(use_threads=not isinstance(reduction, (First, Last)))

    def _bin_id(self, x_col, y_col, width, height, x_range, y_range):
        x0, x1 = x_range
        y0, y1 = y_range
        mask = (pc.field(x_col) >= x0) & (pc.field(x_col) < x1) & (pc.field(y_col) >= y0) & (pc.field(y_col) < y1)
        x_bin = pc.min_element_wise(
            ((pc.field(x_col) - x0) / (x1 - x0) * width).cast(pa.int64(), safe=False), width - 1
        )
        y_bin = pc.min_element_wise(
            ((pc.field(y_col) - y0) / (y1 - y0) * height).cast(pa.int64(), safe=False), height - 1
        )
        return y_bin * width + x_bin, mask

    def _source(self, source, columns=None, scan_filter=None):
        if isinstance(source, ds.Dataset):
            opts = {**self._scan_options}
            if columns:
                opts["columns"] = columns
            if scan_filter is not None:
                opts["filter"] = scan_filter
            return acero.Declaration("scan", acero.ScanNodeOptions(source, **opts))
        if isinstance(source, pa.Table):
            if columns:
                source = source.select(columns)
            return acero.Declaration("table_source", acero.TableSourceNodeOptions(source))
        raise TypeError(f"Unsupported normalized source type: {type(source).__name__}")

    def _projections(self, reduction, bin_id):
        exprs = [bin_id]
        names = ["bin_id"]
        if type(reduction) in _ACERO_FUNCTIONS:
            exprs.append(pc.field(reduction.column))
            names.append(reduction.column)
        return exprs, names

    def _aggregates(self, reduction):
        if isinstance(reduction, Count):
            return [("bin_id", "hash_count", None, "count")]
        if type(reduction) not in _ACERO_FUNCTIONS:
            raise TypeError(f"Unsupported reduction: {type(reduction).__name__}")
        func, name = _ACERO_FUNCTIONS[type(reduction)]
        return [(reduction.column, func, None, name)]
