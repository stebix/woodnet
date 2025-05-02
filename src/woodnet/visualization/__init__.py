from matplotlib.axes import Axes


def plot_rectangle(
    ax: Axes,
    sy: int,
    sx: int,
    dy: int,
    dx: int,
    marker: str = 'o',
    markercolor: str = 'r',
    markersize: float = 5,
    plot_lines: bool = True,
    line_color: str = 'k',
    line_alpha: float = 1.0,
) -> Axes:
    kwargs = {'marker' : marker, 'color' : markercolor, 'markersize' : markersize}
    if plot_lines:
        line_kwargs = {'color' : line_color, 'alpha' : line_alpha}
        ax.vlines(sx, ymin=sy, ymax=sy+dy, **line_kwargs)
        ax.vlines(sx+dx, ymin=sy, ymax=sy+dy, **line_kwargs)
        ax.hlines(sy, xmin=sx, xmax=sx+dx, **line_kwargs)
        ax.hlines(sy+dy, xmin=sx, xmax=sx+dx, **line_kwargs)
        
    ax.plot(sy, sx, **kwargs)
    ax.plot(sy, sx+dx, **kwargs)
    ax.plot(sy+dy, sx, **kwargs)
    ax.plot(sy+dy, sx+dx, **kwargs)
    return ax