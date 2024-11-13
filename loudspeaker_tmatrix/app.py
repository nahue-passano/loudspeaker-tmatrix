import streamlit as st
import numpy as np
import pandas as pd

from loudspeaker_tmatrix.core import simulate_loudspeaker
from loudspeaker_tmatrix.visualization import plotly_full_figure
from loudspeaker_tmatrix.config import (
    load_config,
    load_loudspeaker_config,
)

st.set_page_config(layout="wide")

config_path = "configs/config.yaml"
default_loudspeaker_path = "configs/default_loudspeaker.yaml"

cfg = load_config(config_path)
loudspeaker_cfg = load_loudspeaker_config(default_loudspeaker_path)

freq_array = np.logspace(
    np.log10(cfg.frequency.min), np.log10(cfg.frequency.max), num=cfg.frequency.n_bins
)

angular_freq_array = 2 * np.pi * freq_array

left, right = st.columns([0.4, 0.6])
left.title("Loudspeaker simulation")
right.image("images/loudspeaker.png")

single_speaker, two_speakers = st.tabs(["Single speaker", "Two speakers comparison"])

with single_speaker:
    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        st.title("Thielle small parameters")
        ### Sliders
        slider_spk1_Re = st.slider(
            "Electrical Coil Resistance (Re) [Ohm] .",
            1.0,
            20.0,
            loudspeaker_cfg.electrical.coil_resistance,
        )
        slider_spk1_Le = st.slider(
            "Electrical Coil Inductance (Le) [mH] .",
            0.0,
            5.0,
            loudspeaker_cfg.electrical.coil_inductance * 1e3,
        )
        slider_spk1_Le /= 1e3
        slider_spk1_Bl = st.slider(
            "Electromechanical factor (Bl) [T*m] .",
            1.0,
            20.0,
            loudspeaker_cfg.electromechanical_factor,
        )
        slider_spk1_Rm = st.slider(
            "Mechanical Resistance (Rms) [kg/s] .",
            0.0,
            10.0,
            loudspeaker_cfg.mechanical.resistance,
        )
        slider_spk1_Mm = st.slider(
            "Mechanical Mass (Mms) [g] .",
            1.0,
            100.0,
            loudspeaker_cfg.mechanical.mass * 1e3,
        )
        slider_spk1_Mm /= 1e3
        slider_spk1_Cm = st.slider(
            "Mechanical Compliance (Cms) [mm/N] .",
            0.0,
            5.0,
            loudspeaker_cfg.mechanical.compliance * 1e3,
        )
        slider_spk1_Cm /= 1e3
        slider_spk1_diam = st.slider(
            "Effective diameter of radiation [cm] .",
            1.0,
            50.0,
            loudspeaker_cfg.acoustical.effective_diameter * 1e2,
        )
        slider_spk1_diam /= 1e2

    thiele_small_params_1 = {
        "Re": slider_spk1_Re,
        "Le": slider_spk1_Le,
        "Bl": slider_spk1_Bl,
        "Mm": slider_spk1_Mm,
        "Cm": slider_spk1_Cm,
        "Rm": slider_spk1_Rm,
        "effective_diameter": slider_spk1_diam,
    }

    loudspeaker_responses_1 = simulate_loudspeaker(
        thiele_small_params_1, angular_freq_array, cfg.acoustical_constants
    )

    plotly_fig = plotly_full_figure(
        freq_array, loudspeaker_responses_1, single_speaker=True
    )
    with col1:
        selectivity_factors = loudspeaker_responses_1["selectivity_params"]

        st.dataframe(
            pd.DataFrame([selectivity_factors]),
            hide_index=True,
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(plotly_fig, use_container_width=True, theme=None)


with two_speakers:
    spk1, plots, spk2 = st.columns([0.17, 0.66, 0.17])

    with spk1:
        st.title("Speaker 1")
        ### Sliders
        slider_spk1_Re = st.slider(
            "Electrical Coil Resistance (Re) [Ohm]",
            1.0,
            20.0,
            loudspeaker_cfg.electrical.coil_resistance,
        )
        slider_spk1_Le = st.slider(
            "Electrical Coil Inductance (Le) [mH]",
            0.0,
            5.0,
            loudspeaker_cfg.electrical.coil_inductance * 1e3,
        )
        slider_spk1_Le /= 1e3
        slider_spk1_Bl = st.slider(
            "Electromechanical factor (Bl) [T*m]",
            1.0,
            20.0,
            loudspeaker_cfg.electromechanical_factor,
        )
        slider_spk1_Rm = st.slider(
            "Mechanical Resistance (Rms) [kg/s]",
            0.0,
            10.0,
            loudspeaker_cfg.mechanical.resistance,
        )
        slider_spk1_Mm = st.slider(
            "Mechanical Mass (Mms) [g]",
            1.0,
            100.0,
            loudspeaker_cfg.mechanical.mass * 1e3,
        )
        slider_spk1_Mm /= 1e3
        slider_spk1_Cm = st.slider(
            "Mechanical Compliance (Cms) [mm/N]",
            0.0,
            5.0,
            loudspeaker_cfg.mechanical.compliance * 1e3,
        )
        slider_spk1_Cm /= 1e3
        slider_spk1_diam = st.slider(
            "Effective diameter of radiation [cm]",
            1.0,
            50.0,
            loudspeaker_cfg.acoustical.effective_diameter * 1e2,
        )
        slider_spk1_diam /= 1e2

    with spk2:
        st.title("Speaker 2")
        ### Sliders
        slider_spk2_Re = st.slider(
            "Electrical Coil Resistance (Re) [Ohm].",
            1.0,
            20.0,
            loudspeaker_cfg.electrical.coil_resistance,
        )
        slider_spk2_Le = st.slider(
            "Electrical Coil Inductance (Le) [mH].",
            0.0,
            5.0,
            loudspeaker_cfg.electrical.coil_inductance * 1e3,
        )
        slider_spk2_Le /= 1e3
        slider_spk2_Bl = st.slider(
            "Electromechanical factor (Bl) [T*m].",
            1.0,
            20.0,
            loudspeaker_cfg.electromechanical_factor,
        )
        slider_spk2_Rm = st.slider(
            "Mechanical Resistance (Rms) [kg/s].",
            0.0,
            10.0,
            loudspeaker_cfg.mechanical.resistance,
        )
        slider_spk2_Mm = st.slider(
            "Mechanical Mass (Mms) [g].",
            1.0,
            100.0,
            loudspeaker_cfg.mechanical.mass * 1e3,
        )
        slider_spk2_Mm /= 1e3
        slider_spk2_Cm = st.slider(
            "Mechanical Compliance (Cms) [mm/N].",
            0.0,
            5.0,
            loudspeaker_cfg.mechanical.compliance * 1e3,
        )
        slider_spk2_Cm /= 1e3
        slider_spk2_diam = st.slider(
            "Effective diameter of radiation [cm].",
            1.0,
            50.0,
            loudspeaker_cfg.acoustical.effective_diameter * 1e2,
        )
        slider_spk2_diam /= 1e2

    # Speaker 1 simulation
    thiele_small_params_1 = {
        "Re": slider_spk1_Re,
        "Le": slider_spk1_Le,
        "Bl": slider_spk1_Bl,
        "Mm": slider_spk1_Mm,
        "Cm": slider_spk1_Cm,
        "Rm": slider_spk1_Rm,
        "effective_diameter": slider_spk1_diam,
    }

    loudspeaker_responses_1 = simulate_loudspeaker(
        thiele_small_params_1, angular_freq_array, cfg.acoustical_constants
    )

    # Speaker 2 simulation
    thiele_small_params_2 = {
        "Re": slider_spk2_Re,
        "Le": slider_spk2_Le,
        "Bl": slider_spk2_Bl,
        "Mm": slider_spk2_Mm,
        "Cm": slider_spk2_Cm,
        "Rm": slider_spk2_Rm,
        "effective_diameter": slider_spk2_diam,
    }

    loudspeaker_responses_2 = simulate_loudspeaker(
        thiele_small_params_2, angular_freq_array, cfg.acoustical_constants
    )

    with spk1:
        selectivity_factors = loudspeaker_responses_1["selectivity_params"]

        st.dataframe(
            pd.DataFrame([selectivity_factors]),
            hide_index=True,
            use_container_width=True,
        )

    with spk2:
        selectivity_factors = loudspeaker_responses_2["selectivity_params"]

        st.dataframe(
            pd.DataFrame([selectivity_factors]),
            hide_index=True,
            use_container_width=True,
        )

    with plots:
        plotly_fig = plotly_full_figure(
            freq_array, loudspeaker_responses_1, loudspeaker_responses_2
        )
        st.plotly_chart(plotly_fig)
