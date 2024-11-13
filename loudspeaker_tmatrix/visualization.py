import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from loudspeaker_tmatrix.utils import to_db


def plot_loudspeaker_response(
    response_array: np.ndarray,
    freq_array: np.ndarray,
    title: str,
    magnitude_in_db: bool,
    magnitude_units: str,
    shift_phase: bool,
):

    if magnitude_in_db:
        magnitude = to_db(response_array)
    else:
        magnitude = np.abs(response_array)

    if shift_phase:
        phase = np.angle(-response_array, deg=True)
    else:
        phase = np.angle(response_array, deg=True)

    fig, ax1 = plt.subplots(figsize=(12, 3))
    fig.suptitle(title)

    ax1.semilogx(freq_array, magnitude, label="Magnitude", color="C0")
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel(f"Magnitude [{magnitude_units}]", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")

    ax2 = ax1.twinx()
    ax2.semilogx(freq_array, phase, color="r", label="Phase")
    x_ticks = np.sort(np.array([16, 31, 63, 125, 250, 500, 1000, 2000, 4000]))
    ax2.set_xticks(ticks=x_ticks, labels=x_ticks.tolist(), rotation=45)
    ax2.set_ylabel("Phase [º]", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    y_label1 = [r"$-180º$", r"$-90º$", r"$0º$", r"$90º$", r"$180º$"]
    ax2.set_yticks(np.array([-180, -90, 0, 90, 180]), y_label1)

    ax1.set_xlim(10, 4000)
    ax2.set_ylim(-180, 180)
    ax1.grid(axis="x")
    ax2.grid(axis="y")
    return fig


def plotly_full_figure(
    freq_array,
    loudspeaker_responses_1,
    loudspeaker_responses_2=None,
    single_speaker=False,
):
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.1,
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": True}],
            [{"secondary_y": True}],
            [{"secondary_y": True}],
        ],  # Enable secondary y-axis for both rows
    )

    ###### Speaker 1
    ### Acoustical Pressure
    fig.add_trace(
        go.Scatter(
            x=freq_array,
            y=to_db(loudspeaker_responses_1["acoustical_pressure"]),
            mode="lines",
            name="Speaker 1",
            line=dict(color="blue"),
            hovertemplate="%{x:.1f} Hz<br>%{y:.1f} " + "dB",
            legendgroup="Speaker 1 (Magnitude)",
            showlegend=True,
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=freq_array,
            y=np.angle(loudspeaker_responses_1["acoustical_pressure"], deg=True),
            mode="lines",
            name="Speaker 1",
            line=dict(color="red"),
            hovertemplate="%{x:.1f} Hz<br>%{y:.1f} º",
            legendgroup="Speaker 1 (Phase)",
            showlegend=True,
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    ### Electrical Impedance
    fig.add_trace(
        go.Scatter(
            x=freq_array,
            y=np.abs(loudspeaker_responses_1["electrical_impedance"]),
            mode="lines",
            name="Speaker 1",
            line=dict(color="blue"),
            hovertemplate="%{x:.1f} Hz<br>%{y:.1f} " + "Ohm",
            legendgroup="Speaker 1 (Magnitude)",
            showlegend=False,
        ),
        row=2,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=freq_array,
            y=np.angle(loudspeaker_responses_1["electrical_impedance"], deg=True),
            mode="lines",
            name="Speaker 1",
            line=dict(color="red"),
            hovertemplate="%{x:.1f} Hz<br>%{y:.1f} º",
            legendgroup="Speaker 1 (Phase)",
            showlegend=False,
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    ### Mechanical Velocity
    fig.add_trace(
        go.Scatter(
            x=freq_array,
            y=np.abs(loudspeaker_responses_1["mechanical_velocity"]),
            mode="lines",
            name="Speaker 1",
            line=dict(color="blue"),
            hovertemplate="%{x:.2f} Hz<br>%{y:.2f} " + "m/s",
            legendgroup="Speaker 1 (Magnitude)",
            showlegend=False,
        ),
        row=3,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=freq_array,
            y=np.angle(-loudspeaker_responses_1["mechanical_velocity"], deg=True),
            mode="lines",
            name="Speaker 1",
            line=dict(color="red"),
            hovertemplate="%{x:.2f} Hz<br>%{y:.2f} º",
            legendgroup="Speaker 1 (Phase)",
            showlegend=False,
        ),
        row=3,
        col=1,
        secondary_y=True,
    )
    ### Mechanical displacement
    fig.add_trace(
        go.Scatter(
            x=freq_array,
            y=np.abs(loudspeaker_responses_1["mechanical_displacement"]),
            mode="lines",
            name="Speaker 1",
            line=dict(color="blue"),
            hovertemplate="%{x:.2f} Hz<br>%{y:.2f} " + "mm",
            legendgroup="Speaker 1 (Magnitude)",
            showlegend=False,
        ),
        row=4,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=freq_array,
            y=np.angle(loudspeaker_responses_1["mechanical_displacement"], deg=True),
            mode="lines",
            name="Speaker 1",
            line=dict(color="red"),
            hovertemplate="%{x:.2f} Hz<br>%{y:.2f} º",
            legendgroup="Speaker 1 (Phase)",
            showlegend=False,
        ),
        row=4,
        col=1,
        secondary_y=True,
    )

    ###### Speaker 2
    if not single_speaker:
        ### Acoustical Pressure
        fig.add_trace(
            go.Scatter(
                x=freq_array,
                y=to_db(loudspeaker_responses_2["acoustical_pressure"]),
                mode="lines",
                name="Speaker 2",
                line=dict(color="blue", dash="dot"),
                hovertemplate="%{x:.1f} Hz<br>%{y:.1f} " + "dB",
                legendgroup="Speaker 2 (Magnitude)",
                showlegend=True,
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=freq_array,
                y=np.angle(loudspeaker_responses_2["acoustical_pressure"], deg=True),
                mode="lines",
                name="Speaker 2",
                line=dict(color="red", dash="dot"),
                hovertemplate="%{x:.1f} Hz<br>%{y:.1f} º",
                legendgroup="Speaker 2 (Phase)",
                showlegend=True,
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

        ### Electrical Impedance
        fig.add_trace(
            go.Scatter(
                x=freq_array,
                y=np.abs(loudspeaker_responses_2["electrical_impedance"]),
                mode="lines",
                name="Speaker 2",
                line=dict(color="blue", dash="dot"),
                hovertemplate="%{x:.1f} Hz<br>%{y:.1f} " + "Ohm",
                legendgroup="Speaker 2 (Magnitude)",
                showlegend=False,
            ),
            row=2,
            col=1,
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=freq_array,
                y=np.angle(loudspeaker_responses_2["electrical_impedance"], deg=True),
                mode="lines",
                name="Speaker 2",
                line=dict(color="red", dash="dot"),
                hovertemplate="%{x:.1f} Hz<br>%{y:.1f} º",
                legendgroup="Speaker 2 (Phase)",
                showlegend=False,
            ),
            row=2,
            col=1,
            secondary_y=True,
        )

        ### Mechanical Velocity
        fig.add_trace(
            go.Scatter(
                x=freq_array,
                y=np.abs(loudspeaker_responses_2["mechanical_velocity"]),
                mode="lines",
                name="Speaker 2",
                line=dict(color="blue", dash="dot"),
                hovertemplate="%{x:.1f} Hz<br>%{y:.1f} " + "m/s",
                legendgroup="Speaker 2 (Magnitude)",
                showlegend=False,
            ),
            row=3,
            col=1,
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=freq_array,
                y=np.angle(-loudspeaker_responses_2["mechanical_velocity"], deg=True),
                mode="lines",
                name="Speaker 2",
                line=dict(color="red", dash="dot"),
                hovertemplate="%{x:.1f} Hz<br>%{y:.1f} º",
                legendgroup="Speaker 2 (Phase)",
                showlegend=False,
            ),
            row=3,
            col=1,
            secondary_y=True,
        )
        ### Mechanical displacement
        fig.add_trace(
            go.Scatter(
                x=freq_array,
                y=np.abs(loudspeaker_responses_2["mechanical_displacement"]),
                mode="lines",
                name="Speaker 2",
                line=dict(color="blue", dash="dot"),
                hovertemplate="%{x:.1f} Hz<br>%{y:.1f} " + "mm",
                legendgroup="Speaker 2 (Magnitude)",
                showlegend=False,
            ),
            row=4,
            col=1,
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=freq_array,
                y=np.angle(
                    loudspeaker_responses_2["mechanical_displacement"], deg=True
                ),
                mode="lines",
                name="Speaker 2",
                line=dict(color="red", dash="dot"),
                hovertemplate="%{x:.1f} Hz<br>%{y:.1f} º",
                legendgroup="Speaker 2 (Phase)",
                showlegend=False,
            ),
            row=4,
            col=1,
            secondary_y=True,
        )

    ### Update layout
    fig.update_layout(
        template="ggplot2",
        xaxis=dict(
            type="log",
            tickvals=[16, 31, 63, 125, 250, 500, 1000, 2000, 4000],
            range=[np.log10(10), np.log10(4000)],
        ),
        xaxis2=dict(
            type="log",
            tickvals=[16, 31, 63, 125, 250, 500, 1000, 2000, 4000],
            range=[np.log10(10), np.log10(4000)],
        ),
        xaxis3=dict(
            type="log",
            tickvals=[16, 31, 63, 125, 250, 500, 1000, 2000, 4000],
            range=[np.log10(10), np.log10(4000)],
        ),
        xaxis4=dict(
            title="Frequency [Hz]",
            titlefont=dict(size=14),
            tickfont=dict(size=12),
            type="log",
            tickvals=[16, 31, 63, 125, 250, 500, 1000, 2000, 4000],
            range=[np.log10(10), np.log10(4000)],
        ),
        yaxis=dict(title="Pressure level [dB]", color="blue"),
        yaxis2=dict(
            title="Phase [º]",
            color="red",
            tickvals=[-180, -90, 0, 90, 180],
            ticktext=["-180º", "-90º", "0º", "90º", "180º"],
            range=[-180, 180],
        ),
        yaxis3=dict(title="Impedance [Ohm]", color="blue"),
        yaxis4=dict(
            title="Phase [º]",
            color="red",
            tickvals=[-180, -90, 0, 90, 180],
            ticktext=["-180º", "-90º", "0º", "90º", "180º"],
            range=[-180, 180],
        ),
        yaxis5=dict(title="Velocity [m/s]", color="blue"),
        yaxis6=dict(
            title="Phase [º]",
            color="red",
            tickvals=[-180, -90, 0, 90, 180],
            ticktext=["-180º", "-90º", "0º", "90º", "180º"],
            range=[-180, 180],
        ),
        yaxis7=dict(title="Displacement [mm]", color="blue"),
        yaxis8=dict(
            title="Phase [º]",
            color="red",
            tickvals=[-180, -90, 0, 90, 180],
            ticktext=["-180º", "-90º", "0º", "90º", "180º"],
            range=[-180, 180],
        ),
        legend=dict(
            font=dict(size=10),
            x=0.245,
            y=1.06,
            orientation="h",
            itemwidth=30,
        ),
        showlegend=not single_speaker,
        width=1200,
        height=1000,
        margin=dict(l=20, r=20, t=30, b=20),
        annotations=[
            dict(
                text="<b>Acoustical Pressure</b>",
                x=0.47,
                y=1.03,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=18),
            ),
            dict(
                text="<b>Electrical Impedance</b>",
                x=0.47,
                y=0.755,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=18),
            ),
            dict(
                text="<b>Mechanical Velocity</b>",
                x=0.47,
                y=0.47,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=18),
            ),
            dict(
                text="<b>Mechanical Displacement</b>",
                x=0.47,
                y=0.175,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=18),
            ),
        ],
    )
    return fig
