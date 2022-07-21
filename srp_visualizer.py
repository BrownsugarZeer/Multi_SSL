import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from src.utils.cmd_parser import parsing_params


PRETTY_PRINT = True
TEMPLATE = "plotly_dark" if PRETTY_PRINT else "presentation"
PAPER_BGCOLOR = "#212946" if PRETTY_PRINT else None
PLOT_BGCOLOR = "#212946" if PRETTY_PRINT else None
GRID_COLOR = "#5063AB" if PRETTY_PRINT else None

MARKER_COLOR = "#00AFFF"
MARKER_PROJ_COLOR = "#005FFF"

MICS = np.array([
    [-0.02285, -0.02285, +0.005],
    [+0.02285, -0.02285, +0.005],
    [+0.02285, +0.02285, +0.005],
    [-0.02285, +0.02285, +0.005],
])


class PlotlySphere:
    def __init__(self, r=1):
        self.r = r
        self._fig = go.Figure()
        self._fig.layout.template = TEMPLATE
        self._mic_pos = MICS

        self._create_sphere()

    def __call__(self, fname="DOAS.html", auto_open=True):
        self._fig.write_html(fname, auto_open=auto_open)

    def _create_sphere(self):
        phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
        x = self.r*np.sin(phi)*np.cos(theta)
        y = self.r*np.sin(phi)*np.sin(theta)
        z = self.r*np.cos(phi)

        circle_r = 0.035
        circle_theta = np.linspace(0, 2*np.pi, 100)
        circle_x = circle_r * np.cos(circle_theta)
        circle_y = circle_r * np.sin(circle_theta)
        circle_z = np.zeros_like(circle_theta)

        self._fig.add_surface(
            x=x, y=y, z=z,
            opacity=0.05,
            colorscale="Plotly3",
            contours_z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="limegreen",
                project_z=True
            ),
            hoverinfo="skip",
            name="sphere",
        )

        self._fig.add_traces([
            go.Scatter3d(
                x=circle_x, y=circle_y, z=circle_z,
                mode="lines",
                line=dict(color="black", width=2),
                name="mic_array1",
            ),
            go.Scatter3d(
                x=self._mic_pos[:, 0], y=self._mic_pos[:, 1], z=self._mic_pos[:, 2],
                mode="markers",
                marker=dict(
                    color=["lightgreen", "green", "darkseagreen", "darkseagreen"],
                    size=5
                ),
                name="mic_array2",
            ),
            go.Scatter3d(
                x=[0], y=[0], z=[1],
                mode="markers",
                marker=dict(color="rgb(0,95,255)", size=5),
                visible=False,
                name="sound_sources",
            ),
            # If you want to connect between two points
            # go.Scatter3d(
            #     x=[0., 0.], y=[0., 0.], z=[0., 0.],
            #     visible=False,
            #     mode="lines",
            #     name="lines",
            # ),
        ])

        self._fig.update_layout(
            title={"text": "SRP-PHAT Visualizer", "x": 0.47, "y": 0.99},
            scene_camera=dict(eye=dict(x=1.4, y=-1.4, z=1.4)),
            paper_bgcolor=PAPER_BGCOLOR,
            plot_bgcolor=PLOT_BGCOLOR,
            scene=dict(
                xaxis=dict(
                    backgroundcolor=PAPER_BGCOLOR,
                    gridcolor=GRID_COLOR,
                ),
                yaxis=dict(
                    backgroundcolor=PAPER_BGCOLOR,
                    gridcolor=GRID_COLOR,
                ),
                zaxis=dict(
                    backgroundcolor=PAPER_BGCOLOR,
                    gridcolor=GRID_COLOR,
                ),
            ),
            showlegend=False,
            margin=go.layout.Margin(l=0, r=0, b=0, t=0, pad=0)
        )

    def update_doas(self, points, name="sound_sources"):
        self._fig.for_each_trace(
            lambda trace: trace.update(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                visible=True,
            ) if trace.name == name else (),
        )

    def add_doas(self, doas, show_xyz_proj=None):
        """Add the direction of arrivals in the Cartesian coordinates.

        doas : np.ndarray
            The doas is denoted as (azimuth, elevation) in degrees.
            doas must have the following format: (n_loc, 2)

        """
        if doas.ndim == 1:
            doas = np.expand_dims(doas, 0)

        z = np.sin(doas[:, 1] / 57.293)
        r = np.cos(doas[:, 1] / 57.293)
        x = np.cos(doas[:, 0] / 57.293) * r
        y = np.sin(doas[:, 0] / 57.293) * r
        points = np.stack((x, y, z), axis=-1)

        scatter_list = [
            [points[:, 0], points[:, 1], points[:, 2], MARKER_COLOR],                   # xyz
        ]

        xyz_plane = [
            [points[:, 0], points[:, 1], np.zeros(len(points))-1, MARKER_PROJ_COLOR],   # xy_plane
            [points[:, 0], np.zeros(len(points))+1, points[:, 2], MARKER_PROJ_COLOR],   # yz_plane
            [np.zeros(len(points))-1, points[:, 1], points[:, 2], MARKER_PROJ_COLOR],   # xz_plane
        ]

        if isinstance(show_xyz_proj, list):
            for proj, plane in zip(show_xyz_proj, xyz_plane):
                if len(proj*plane) != 0:
                    scatter_list.extend([proj*plane])
        elif show_xyz_proj:
            scatter_list.extend(xyz_plane)

        for x, y, z, c in scatter_list:
            tbox = "X: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>"
            self._fig.add_scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                marker=dict(color=c, size=5),
                hovertemplate=tbox,
                opacity=0.05,
            )


def main():
    params = parsing_params()

    if params.src is None:
        raise ValueError("the number of sources should be at least one.")
    if not Path(params.wave).is_file():
        raise FileExistsError("the file path is not correct or pass via --wave")

    fname = Path(params.wave)

    ps = PlotlySphere()
    for c in [(col*2, col*2+1) for col in range(params.src)]:
        doas = np.genfromtxt(
            fname,
            delimiter=",",
            skip_header=True,
            usecols=c,
            dtype=np.float32
        )
        ps.add_doas(doas, show_xyz_proj=[True, False, False])
    ps(fname=fname.with_suffix(".html"))


if __name__ == '__main__':
    main()
