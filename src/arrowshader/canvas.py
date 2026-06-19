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


def _apply_cmap(data, cmap):
    colors = np.array([tuple(int(h[i : i + 2], 16) for i in (1, 3, 5)) for h in cmap])
    span = np.linspace(0, 1, len(colors))
    r = np.nan_to_num(np.interp(data, span, colors[:, 0])).astype(np.uint8)
    g = np.nan_to_num(np.interp(data, span, colors[:, 1])).astype(np.uint8)
    b = np.nan_to_num(np.interp(data, span, colors[:, 2])).astype(np.uint8)
    a = np.where(np.isnan(data), 0, 255).astype(np.uint8)
    return np.flipud(np.dstack([r, g, b, a]))


class Canvas:
    def __init__(self, width=600, height=600, x_range=None, y_range=None, backend="acero"):
        if x_range is None or y_range is None:
            raise ValueError("Please specify x_range and y_range")
        if isinstance(backend, str):
            if backend not in _BACKENDS:
                raise ValueError(f"Unknown backend: {backend!r}. Available: {list(_BACKENDS)}")
            self._backend = _BACKENDS[backend]()
        else:
            self._backend = backend
        self.width = width
        self.height = height
        self.x_range = x_range
        self.y_range = y_range

    def points(self, source, x, y, agg=None) -> np.ndarray:
        if agg is None:
            agg = count()
        result = self._backend.aggregate(
            to_arrow(source), x, y, self.width, self.height, self.x_range, self.y_range, agg
        )
        value_col = [c for c in result.column_names if c != "bin_id"][0]
        return _scatter(result, self.width, self.height, value_col)

    def plot(self, source, x, y, agg=None, how="eq_hist", cmap=None, background="black"):
        from PIL import Image

        if cmap is None:
            import colorcet

            cmap = colorcet.fire
        grid = self.points(source, x, y, agg)
        mask = np.isnan(grid) if grid.dtype.kind == "f" else grid == 0
        transformed = _normalize_interpolate_how(how)(grid, mask)
        if isinstance(transformed, tuple):
            transformed = transformed[0]
        img = Image.fromarray(_apply_cmap(transformed, cmap))
        if background:
            bg = Image.new("RGBA", img.size, background)
            img = Image.alpha_composite(bg, img)
        return img
