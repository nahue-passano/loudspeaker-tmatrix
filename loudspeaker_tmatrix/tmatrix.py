import numpy as np


def series_impedance(impedance_array):
    n_bins = len(impedance_array)
    tmatrix = np.array(
        [
            [np.ones(n_bins), impedance_array],
            [np.zeros(n_bins), np.ones(n_bins)],
        ]
    )
    return tmatrix


def inductance_series(inductance_value, angular_freq_array):
    impedance_array = 1j * angular_freq_array * inductance_value
    return series_impedance(impedance_array)


def capacitance_series(capacitance_value, angular_freq_array):
    impedance_array = 1 / (1j * angular_freq_array * capacitance_value)
    return series_impedance(impedance_array)


def resistance_series(resistance_value, n_bins):
    impedance_array = np.ones(n_bins) * resistance_value
    return series_impedance(impedance_array)


def transformer(transformer_value, n_bins):
    transformer_array = np.ones(n_bins) * transformer_value
    transformer = np.array(
        [
            [transformer_array, np.zeros(n_bins)],
            [np.zeros(n_bins), 1 / transformer_array],
        ]
    )
    return transformer


def gyrator(gyrator_value, n_bins):
    gyrator_array = np.ones(n_bins) * gyrator_value
    gyrator = np.array(
        [
            [np.zeros(n_bins), gyrator_array],
            [1 / gyrator_array, np.zeros(n_bins)],
        ]
    )
    return gyrator
