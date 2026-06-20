from arrowshader.reductions import Count, Sum, Mean, Min, Max, Std, Var, First, Last, Any
import pyarrow as pa
import pyarrow.dataset as ds

_CUDF_FUNCTIONS = {
    Sum: "sum",
    Mean: "mean",
    Min: "min",
    Max: "max",
    Std: "std",
    Var: "var",
    First: "first",
    Last: "last",
    Any: "any",
}

class CuDF:
    def aggregate(self, source, x_col, y_col, width, height, x_range, y_range, reduction):
        try:
            import cudf
        except ImportError:
            raise ImportError(
                "You need cudf installed for the CuDF backend. "
                "See how to install it here: https://rapids.ai/"
            )
        if isinstance(source, ds.Dataset):
            source = source.to_table(columns=[x_col, y_col] + ([reduction.column] if type(reduction) in _CUDF_FUNCTIONS else []))
        gdf = cudf.DataFrame.from_arrow(source)
        x0, x1 = x_range
        y0, y1 = y_range
        gdf = gdf[(gdf[x_col] >= x0) & (gdf[x_col] < x1) & (gdf[y_col] >= y0) & (gdf[y_col] < y1)]
        x_bin = ((gdf[x_col] - x0) / (x1 - x0) * width).astype("int64").clip(upper=width - 1)
        y_bin = ((gdf[y_col] - y0) / (y1 - y0) * height).astype("int64").clip(upper=height - 1)
        gdf["bin_id"] = y_bin * width + x_bin
        if isinstance(reduction, Count):
            result = gdf.groupby("bin_id").size().reset_index(name="count")
        elif type(reduction) in _CUDF_FUNCTIONS:
            name = _CUDF_FUNCTIONS[type(reduction)]
            result = gdf.groupby("bin_id")[reduction.column].agg(name).reset_index(name=name)
        else:
            raise TypeError(f"Unsupported reduction: {type(reduction).__name__}")
        return result.to_arrow()