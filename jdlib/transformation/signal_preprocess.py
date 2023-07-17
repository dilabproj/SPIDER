from typing import List, Tuple
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from biosppy.signals import ecg


def cut_signal(signals: List[float], length: int) -> List[float]:
    """Drop ecg last few signal

    Arguments:
        signals {List[float]} -- Signal to cut
        length {int} -- Cutting point of the raw signal

    Returns:
        List[float] -- Cutted signal
    """

    assert len(signals) > length, "length is too long!"
    return signals[0:length]


def _median_filter(signals: List[float], window_len: Tuple[int, int]) -> Tuple[List[float], List[float]]:
    """High pass median filter

    Arguments:
        signals {List[float]} -- Signal to filter
        window_len {List[int]} -- Two kernel sizes

    Returns:
        Tuple[List[float], List[float]] -- Filtered signals
    """

    baseline = signal.medfilt(signals, window_len[0])
    baseline = signal.medfilt(baseline, window_len[1])
    for i, sig in enumerate(signals):
        signals[i] = sig - baseline[i]
    return signals, baseline


def remove_baseline_wander(signals: List[float],
                           types: str,
                           lead_sampling_rate: int, 
                           window_len: Tuple[int, int] = (101, 301),
                           return_baseline: bool = False):  # Remove Return Type Annotation to Pass Mypy
    """Baseline wander removal

    Note:
        We consider sampling rate of 500Hz, so the window length of 200 and 600 should be 100 and 300.

    Arguments:
        signals {List[float]} -- Signal to filter
        types {str} -- Filter type
        lead_sampling_rate {int} -- the sampling rate of the signal

    Keyword Arguments:
        window_len {Tuple[int, int]} -- Kernel size for median filter (default: {(101, 301)})

    Raises:
        Exception -- Types empty or mismatch

    Returns:
        List[float] -- Filtered signal
    """

    if types == 'butterworth_highpass':
        assert ~return_baseline, "butterworth does not contain baseline!"
        filtered_signal, _, _ = ecg.st.filter_signal(signals, 'butter', 'highpass', 2, 1, lead_sampling_rate)
    elif types == 'median_filter1D':
        filtered_signal, baseline = _median_filter(signals, window_len)
    else:
        raise Exception("No this type of baseline wander removal!")
    return (filtered_signal, baseline) if return_baseline else filtered_signal


def remove_highfreq_noise(signals: List[float], types: str, lead_sampling_rate: int) -> List[float]:
    """High frequency noise removal

    Arguments:
        signals {List[float]} -- Signal to filter
        types {str} -- Filter type
        lead_sampling_rate {int} -- the sampling rate of the signal

    Raises:
        Exception -- Types empty or mismatch

    Returns:
        List[float] -- Filtered signal
    """

    if types == 'butterworth_lowpass':
        filtered_signal, _, _ = ecg.st.filter_signal(signals, 'butter', 'lowpass', 12, 35, lead_sampling_rate)
    elif types == 'fir':
        filtered_signal, _, _ = ecg.st.filter_signal(signals, 'FIR', 'lowpass', 12, 35, lead_sampling_rate)
    else:
        raise Exception("No this type of high frequency removal!")
    return filtered_signal


def get_oneheartbeat(signals: List[float], sample_rate: float, method: str = "first") -> List[float]:
    """Get one heart beat by mean or median in a lead signal

    Arguments:
        signals {List[float]} -- One lead signal
        sample_rate {float} -- Sample rate

    Keyword Arguments:
        method {str} -- Method to aggregate all heart beat (default: {"mean"})

    Raises:
        Exception: Method which is not included

    Returns:
        List[float] -- Aggregated one heart beat
    """

    # Get R peaks
    rpeaks = ecg.hamilton_segmenter(signals, sample_rate)[0]

    template, _ = ecg.extract_heartbeats(signals, rpeaks, sample_rate)

    if method == "first":
        return template[0]
    elif method == "mean":
        return list(np.mean(template, axis=0))
    elif method == "median":
        return list(np.median(template, axis=0))
    else:
        raise Exception("Not contain this method!")


def preprocess_lead(lead: List[float], lead_sampling_rate: int) -> List[float]:
    """Preprocess lead data to remove baseline wander and high freq noise

    Arguments:
        lead {List[float]} -- original lead data
        lead_sampling_rate {int} -- the sampling rate of the signal

    Returns:
        List[float] -- corrected_signal
    """
    corrected_signal = remove_baseline_wander(lead, 'butterworth_highpass', lead_sampling_rate)
    corrected_signal = remove_highfreq_noise(corrected_signal, 'butterworth_lowpass', lead_sampling_rate)
    return corrected_signal


def demo_plot(signals: List[float], filename: str):
    """Plot signals

    Arguments:
        signals {List[float]} -- Signal to plot
        filename {str} -- Filename
    """

    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(8, 4), dpi=200)
    plt.tight_layout()
    plt.plot(signals)
    plt.savefig(f"./Image/{filename}.png")
    plt.close()
    # plt.show()