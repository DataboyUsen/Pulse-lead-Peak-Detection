# -*- coding: utf-8 -*-
"""
An adjusted version of ppg_simulate( ) function I wrote for adding noise, 
adjusted version is very similar to original one from neurokit2 package except 
for an extra parameter controlling the degree of noise. 

And adjusted version outputs 2 signals, one is normal ppg signal and another is noise-added ppg signal.

For running this project, one may just open the souce code of neurokit2 and replace the original function 
by flowwing one 
"""


def ppg_simulate(
    duration=120,
    sampling_rate=1000,
    heart_rate=70,
    frequency_modulation=0.2,
    ibi_randomness=0.1,
    drift=0,
    motion_amplitude=0.1,
    powerline_amplitude=0.01,
    burst_number=0,
    burst_amplitude=1,
    random_state=None,
    random_state_distort="spawn",
    show=False,
    motion_freq=2
):
    """**Simulate a photoplethysmogram (PPG) signal**

    Phenomenological approximation of PPG. The PPG wave is described with four landmarks: wave
    onset, location of the systolic peak, location of the dicrotic notch and location of the
    diastolic peaks. These landmarks are defined as x and y coordinates (in a time series). These
    coordinates are then interpolated at the desired sampling rate to obtain the PPG signal.

    Parameters
    ----------
    duration : int
        Desired recording length in seconds. The default is 120.
    sampling_rate : int
        The desired sampling rate (in Hz, i.e., samples/second). The default is 1000.
    heart_rate : int
        Desired simulated heart rate (in beats per minute). The default is 70. Note that for the
        ECGSYN method, random fluctuations are to be expected to mimic a real heart rate. These
        fluctuations can cause some slight discrepancies between the requested heart rate and the
        empirical heart rate, especially for shorter signals.
    frequency_modulation : float
        Float between 0 and 1. Determines how pronounced respiratory sinus arrythmia (RSA) is
        (0 corresponds to absence of RSA). The default is 0.3.
    ibi_randomness : float
        Float between 0 and 1. Determines how much random noise there is in the duration of each
        PPG wave (0 corresponds to absence of variation). The default is 0.1.
    drift : float
        Float between 0 and 1. Determines how pronounced the baseline drift (.05 Hz) is
        (0 corresponds to absence of baseline drift). The default is 1.
    motion_amplitude : float
        Float between 0 and 1. Determines how pronounced the motion artifact (0.5 Hz) is
        (0 corresponds to absence of motion artifact). The default is 0.1.
    powerline_amplitude : float
        Float between 0 and 1. Determines how pronounced the powerline artifact (50 Hz) is
        (0 corresponds to absence of powerline artifact). Note that powerline_amplitude > 0 is only
        possible if ``sampling_rate`` is >= 500. The default is 0.1.
    burst_amplitude : float
        Float between 0 and 1. Determines how pronounced high frequency burst artifacts are
        (0 corresponds to absence of bursts). The default is 1.
    burst_number : int
        Determines how many high frequency burst artifacts occur. The default is 0.
    show : bool
        If ``True``, returns a plot of the landmarks and interpolated PPG. Useful for debugging.
    random_state : None, int, numpy.random.RandomState or numpy.random.Generator
        Seed for the random number generator. See for ``misc.check_random_state`` for further information.
    random_state_distort : {'legacy', 'spawn'}, None, int, numpy.random.RandomState or numpy.random.Generator
        Random state to be used to distort the signal. If ``"legacy"``, use the same random state used to
        generate the signal (discouraged as it creates dependent random streams). If ``"spawn"``, spawn
        independent children random number generators from the random_state argument. If any of the other types,
        generate independent children random number generators from the random_state_distort provided (this
        allows generating multiple version of the same signal distorted by different random noise realizations).

    Returns
    -------
    ppg : array
        A vector containing the PPG.

    See Also
    --------
    ecg_simulate, rsp_simulate, eda_simulate, emg_simulate

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      ppg = nk.ppg_simulate(duration=40, sampling_rate=500, heart_rate=75, random_state=42)

    """
    # Seed the random generator for reproducible results
    rng = check_random_state(random_state)
    random_state_distort = check_random_state_children(random_state, random_state_distort, n_children=4)

    # At the requested sampling rate, how long is a period at the requested
    # heart-rate and how often does that period fit into the requested
    # duration?
    period = 60 / heart_rate  # in seconds
    n_period = int(np.floor(duration / period))
    periods = np.ones(n_period) * period

    # Seconds at which waves begin.
    x_onset = np.cumsum(periods)
    x_onset -= x_onset[0]  # make sure seconds start at zero
    # Add respiratory sinus arrythmia (frequency modulation).
    periods, x_onset = _frequency_modulation(
        periods,
        x_onset,
        modulation_frequency=0.05,
        modulation_strength=frequency_modulation,
    )
    # Randomly modulate duration of waves by subracting a random value between
    # 0 and ibi_randomness% of the wave duration (see function definition).
    x_onset = _random_x_offset(x_onset, ibi_randomness, rng)
    # Corresponding signal amplitudes.
    y_onset = rng.normal(0, 0.1, n_period)

    # Seconds at which the systolic peaks occur within the waves.
    x_sys = x_onset + rng.normal(0.175, 0.01, n_period) * periods
    # Corresponding signal amplitudes.
    y_sys = y_onset + rng.normal(1.5, 0.15, n_period)

    # Seconds at which the dicrotic notches occur within the waves.
    x_notch = x_onset + rng.normal(0.4, 0.001, n_period) * periods
    # Corresponding signal amplitudes (percentage of systolic peak height).
    y_notch = y_sys * rng.normal(0.49, 0.01, n_period)

    # Seconds at which the diastolic peaks occur within the waves.
    x_dia = x_onset + rng.normal(0.45, 0.001, n_period) * periods
    # Corresponding signal amplitudes (percentage of systolic peak height).
    y_dia = y_sys * rng.normal(0.51, 0.01, n_period)

    x_all = np.concatenate((x_onset, x_sys, x_notch, x_dia))
    x_all.sort(kind="mergesort")
    x_all = np.ceil(x_all * sampling_rate).astype(int)  # convert seconds to samples

    y_all = np.zeros(n_period * 4)
    y_all[0::4] = y_onset
    y_all[1::4] = y_sys
    y_all[2::4] = y_notch
    y_all[3::4] = y_dia

    if show:
        __, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax0.scatter(x_all, y_all, c="r")

    # Interpolate a continuous signal between the landmarks (i.e., Cartesian
    # coordinates).
    samples = np.arange(int(np.ceil(duration * sampling_rate)))
    ppg = signal_interpolate(x_values=x_all, y_values=y_all, x_new=samples, method="akima")
    # Remove NAN (values outside interpolation range, i.e., after last sample).
    ppg[np.isnan(ppg)] = np.nanmean(ppg)
    ppg_raw = ppg.copy()

    if show:
        ax0.plot(ppg)

    # Add baseline drift.
    if drift > 0:
        drift_freq = 0.05
        if drift_freq < (1 / duration) * 2:
            drift_freq = (1 / duration) * 2
        ppg = signal_distort(
            ppg,
            sampling_rate=sampling_rate,
            noise_amplitude=drift,
            noise_frequency=drift_freq,
            random_state=random_state_distort[0],
            silent=True,
        )
    # Add motion artifacts.
    if motion_amplitude > 0:
        ppg = signal_distort(
            ppg,
            sampling_rate=sampling_rate,
            noise_amplitude=motion_amplitude,
            noise_frequency=motion_freq,
            random_state=random_state_distort[1],
            silent=True,
        )
    # Add high frequency bursts.
    if burst_amplitude > 0:
        ppg = signal_distort(
            ppg,
            sampling_rate=sampling_rate,
            artifacts_amplitude=burst_amplitude,
            artifacts_frequency=100,
            artifacts_number=burst_number,
            random_state=random_state_distort[2],
            silent=True,
        )
    # Add powerline noise.
    if powerline_amplitude > 0:
        ppg = signal_distort(
            ppg,
            sampling_rate=sampling_rate,
            powerline_amplitude=powerline_amplitude,
            powerline_frequency=50,
            random_state=random_state_distort[3],
            silent=True,
        )

    if show:
        ax1.plot(ppg)

    return ppg,ppg_raw