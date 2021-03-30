from scipy.signal import butter, lfilter

"""
Calculate Butterworth coefficients for constructing a Butterworth filter with frequency response as low
as possible in the passband.

Parameters
----------
:param low_cut: (int) The lower bound to the passband of the frequency spectrum.
:param high_cut: (int) The upper bound to the passband of the frequency spectrum.
:param sampling_frequency: (int) Sampling frequency in Hertz.
:param order: (int) The order of the filter.
"""


def butter_bandpass(low_cut, high_cut, sampling_frequency, order):
    nyquist_frequency = 0.5 * sampling_frequency
    low = low_cut / nyquist_frequency
    high = high_cut / nyquist_frequency
    # IIR filter constants
    b, a = butter(order, [low, high], btype='band')
    return b, a


"""
Applies a Butterworth bandpass filter to the given data.

Parameters
----------
:param data: (file) The data to be processed.
:param low_cut: (int) see above.
:param high_cut: (int) see above.
:param order: (int) see above.
"""


def butter_bandpass_filter(data, low_cut, high_cut, fs, order=5):
    b, a = butter_bandpass(low_cut, high_cut, fs, order=order)
    y = lfilter(b, a, data)
    return y
