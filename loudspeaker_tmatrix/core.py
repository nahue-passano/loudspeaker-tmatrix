import numpy as np

from loudspeaker_tmatrix import tmatrix
from loudspeaker_tmatrix.utils import layer_wise_dot_product, struve, bessel
from loudspeaker_tmatrix.config import AcousticalConstantsConfig


def simulate_loudspeaker(
    thiele_small_params: dict,
    angular_freq_array: np.ndarray,
    acoustical_constants: AcousticalConstantsConfig,
):

    n_bins = len(angular_freq_array)
    Re_tmatrix = tmatrix.resistance_series(thiele_small_params["Re"], n_bins)
    Z_Le_tmatrix = tmatrix.inductance_series(
        thiele_small_params["Le"], angular_freq_array
    )
    Bl_tmatrix = tmatrix.gyrator(thiele_small_params["Bl"], n_bins)
    Z_Mm_tmatrix = tmatrix.inductance_series(
        thiele_small_params["Mm"], angular_freq_array
    )
    Z_Cm_tmatrix = tmatrix.capacitance_series(
        thiele_small_params["Cm"], angular_freq_array
    )
    Rm_tmatrix = tmatrix.resistance_series(thiele_small_params["Rm"], n_bins)

    electro_mechanical_tmatrix = layer_wise_dot_product(
        Re_tmatrix,
        Z_Le_tmatrix,
        Bl_tmatrix,
        Z_Mm_tmatrix,
        Z_Cm_tmatrix,
        Rm_tmatrix,
    )

    t_11 = electro_mechanical_tmatrix[0, 0]
    t_12 = electro_mechanical_tmatrix[0, 1]
    t_21 = electro_mechanical_tmatrix[1, 0]
    t_22 = electro_mechanical_tmatrix[1, 1]

    # Electrical response
    electrical_impedance_shorted_output = t_12 / t_22

    # Mechanical response
    electrical_input_voltage = np.ones(n_bins)
    electrical_input_current = (
        electrical_input_voltage / electrical_impedance_shorted_output
    )

    electro_mechanical_tmatrix_det = np.abs(t_11 * t_22 - t_12 * t_21)

    electro_mechanical_tmatrix_inv = np.array(
        [
            [
                t_22 / electro_mechanical_tmatrix_det,
                -t_12 / electro_mechanical_tmatrix_det,
            ],
            [
                -t_21 / electro_mechanical_tmatrix_det,
                t_11 / electro_mechanical_tmatrix_det,
            ],
        ]
    )

    mechanical_force, mechanical_velocity = layer_wise_dot_product(
        electro_mechanical_tmatrix_inv,
        np.array(
            [
                [electrical_input_voltage, electrical_input_voltage],
                [electrical_input_current, electrical_input_current],
            ]
        ),
    )[:, 0]

    mechanical_displacement = mechanical_velocity / (
        1j * angular_freq_array
    )  # [(m/s) / (rad/s)] = [m/rad]
    mechanical_displacement = mechanical_displacement * 2 * np.pi * 1000  # [mm]

    # Acoustical response
    air_impedance = acoustical_constants.air_density * acoustical_constants.sound_speed
    wave_number_array = angular_freq_array / acoustical_constants.sound_speed

    effective_radiation_radius = thiele_small_params["effective_diameter"] / 2

    ka_array = wave_number_array * effective_radiation_radius
    Sd_value = np.pi * effective_radiation_radius**2
    Sd_tmatrix = tmatrix.transformer(Sd_value, n_bins)

    ### Mechanical impedance of radiation
    ZM_rad_real_array = Sd_value * air_impedance * (1 - bessel(2 * ka_array) / ka_array)
    ZM_rad_imag_array = (
        Sd_value * air_impedance * (1j * (struve(2 * ka_array) / ka_array))
    )
    ZM_rad_array = (ZM_rad_real_array + 1j * ZM_rad_imag_array).astype(np.complex128)
    ZM_rad_tmatrix = np.array(
        [[np.ones(n_bins), ZM_rad_array], [np.zeros(n_bins), np.ones(n_bins)]]
    )

    # Specific acoustic impedance
    ZA_rad = (
        1j
        * angular_freq_array
        * acoustical_constants.air_density
        * acoustical_constants.directivity_factor
    ) / (
        4
        * np.pi
        * acoustical_constants.measurement_distance
        * np.exp(1j * wave_number_array * acoustical_constants.measurement_distance)
    )
    Z_delay = np.exp(
        -1j * wave_number_array * acoustical_constants.measurement_distance
    )  # Phase rotation due air propagation time

    electro_mechanical_acoustic_tmatrix = layer_wise_dot_product(
        electro_mechanical_tmatrix, ZM_rad_tmatrix, Sd_tmatrix
    )

    electrical_input_voltage = 2.83

    t_11 = electro_mechanical_acoustic_tmatrix[0, 0]
    t_12 = electro_mechanical_acoustic_tmatrix[0, 1]
    t_21 = electro_mechanical_acoustic_tmatrix[1, 0]
    t_22 = electro_mechanical_acoustic_tmatrix[1, 1]

    voltage_pressure_transfer_function = (ZA_rad) / (t_11 * ZA_rad + t_12)

    acoustical_pressure = (
        electrical_input_voltage
        * (voltage_pressure_transfer_function / Z_delay)
        / acoustical_constants.reference_pressure
    )

    # fmt: off
    mechanical_fs = 1 / (2*np.pi*(thiele_small_params["Mm"]*thiele_small_params["Cm"])**(1/2))
    Qm = 2*np.pi*mechanical_fs*(thiele_small_params["Mm"]+0.00092)/thiele_small_params["Rm"]
    Qe = 2*np.pi*mechanical_fs*(thiele_small_params["Mm"]+0.00092)/(thiele_small_params["Bl"]**2/thiele_small_params["Re"])
    Qt = (Qm*Qe) / (Qm+Qe)                                          
    # fmt: on

    loudspeaker_responses = {
        "electrical_impedance": electrical_impedance_shorted_output,
        "mechanical_force": mechanical_force,
        "mechanical_velocity": mechanical_velocity,
        "mechanical_displacement": mechanical_displacement,
        "acoustical_pressure": acoustical_pressure,
        "selectivity_params": {
            "fs": np.round(mechanical_fs, 2),
            "Qm": np.round(Qm, 2),
            "Qe": np.round(Qe, 2),
            "Qt": np.round(Qt, 2),
        },
    }

    return loudspeaker_responses
