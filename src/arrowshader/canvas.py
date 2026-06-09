import numpy as np
from arrowshader.backends.acero import Acero
from arrowshader.reductions import count
from arrowshader.data import to_arrow
from arrowshader.transfer_functions import _normalize_interpolate_how

_BACKENDS = {
    "acero": Acero,
}


def _scatter(result, width, height, value_column):
    bins = result.column("bin_id").to_numpy()
    values = result.column(value_column).to_numpy()
    fill = np.nan if values.dtype.kind == "f" else 0
    grid = np.full((height, width), fill, dtype=values.dtype)
    grid[bins // width, bins % width] = values
    return grid


class Canvas:
    def __init__(self, width=600, height=600, x_range=None, y_range=None, backend="acero"):
        if x_range is None or y_range is None:
            raise ValueError("Please specify x_range and y_range")
        if backend not in _BACKENDS:
            raise ValueError(f"Unknown backend: {backend!r}. Available: {list(_BACKENDS)}")
        self.width = width
        self.height = height
        self.x_range = x_range
        self.y_range = y_range
        self._backend = _BACKENDS[backend]()

    def points(self, source, x, y, agg=None) -> np.ndarray:
        if agg is None:
            agg = count()
        result = self._backend.aggregate(
            to_arrow(source), x, y, self.width, self.height, self.x_range, self.y_range, agg
        )
        value_col = [c for c in result.column_names if c != "bin_id"][0]
        return _scatter(result, self.width, self.height, value_col)

    def plot(
        self, source, x, y, agg=None, how="eq_hist", cmap="hot", axis=False, background="black", ax=None, **kwargs
    ):
        import matplotlib.pyplot as plt

        grid = self.points(source, x, y, agg)
        mask = np.isnan(grid) if grid.dtype.kind == "f" else grid == 0
        interpolator = _normalize_interpolate_how(how)
        transformed = interpolator(grid, mask)
        if isinstance(transformed, tuple):
            transformed = transformed[0]
        if ax is None:
            ax = plt.gca()
        kwargs.setdefault("origin", "lower")
        kwargs.setdefault("interpolation", "nearest")
        kwargs.setdefault("extent", (*self.x_range, *self.y_range))
        ax.imshow(transformed, cmap=cmap, **kwargs)
        if background:
            ax.set_facecolor(background)
        if not axis:
            ax.set_axis_off()
        return ax
