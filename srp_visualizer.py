import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from src.utils.cmd_parser import parsing_params


# Tested wave files
# Path("data/a0e55.csv")
# Path("data/a0e45_a-90e3.csv")
# Path("data/a0e45_a270e3_a90e42.csv")


class PlotlySphere:
    def __init__(self, r=1):
        self.r = r
        self._fig = go.Figure()
        self._mic_pos = np.array([
            [-0.02285, -0.02285, +0.005],
            [+0.02285, -0.02285, +0.005],
            [+0.02285, +0.02285, +0.005],
            [-0.02285, +0.02285, +0.005],
        ])

        self._create_sphere()

    def __call__(self, f_name="DOAS.html", auto_open=True):
        self._fig.write_html(f_name, auto_open=auto_open)

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

        self._fig.add_traces([
            go.Surface(
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
            ),
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
            title_text="Ring cyclide",
            scene_camera=dict(eye=dict(x=1.4, y=-1.4, z=1.4)),
        )

    def update_doas(self, points, name="sound_sources"):
        self._fig.for_each_trace(
            lambda trace: trace.update(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                visible=True,
            ) if trace.name == name else (),
        )

    def add_doas(self, points, name="sound_sources"):
        self._fig.add_trace(
            go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode="markers",
                marker=dict(color="rgb(0,95,255)", size=5),
                name=name,
            )
        )  # xyz
        self._fig.add_trace(
            go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=np.zeros(len(points))-1,
                mode="markers",
                marker=dict(color="rgb(0,175,255)", size=5),
                name="xy plane",
            )
        )  # xy plane
        self._fig.add_trace(
            go.Scatter3d(
                x=points[:, 0], y=np.zeros(len(points))+1, z=points[:, 2],
                mode="markers",
                marker=dict(color="rgb(0,175,255)", size=5),
                name="yz plane",
            )
        )  # yz plane
        self._fig.add_trace(
            go.Scatter3d(
                x=np.zeros(len(points))-1, y=points[:, 1], z=points[:, 2],
                mode="markers",
                marker=dict(color="rgb(0,175,255)", size=5),
                name="xz plane",
            )
        )  # xz plane


def main():
    params = parsing_params()

    if params.src is None:
        raise ValueError("the number of sources should be at least one.")
    if not Path(params.wave).is_file():
        raise FileExistsError("the file path is not correct or pass via --wave")

    f_name = Path(params.wave)

    ps = PlotlySphere()
    for c in [(col*2, col*2+1) for col in range(params.src)]:
        doas = np.genfromtxt(
            f_name,
            delimiter=",",
            skip_header=True,
            usecols=c,
            dtype=np.float32
        )
        z = np.sin(doas[:, 1] / 57.293)
        r = np.cos(doas[:, 1] / 57.293)
        x = np.cos(doas[:, 0] / 57.293) * r
        y = np.sin(doas[:, 0] / 57.293) * r
        doas = np.stack((x, y, z), axis=-1)

        ps.add_doas(doas)
    ps(f_name=f_name.with_suffix(".html"))


if __name__ == '__main__':
    main()
