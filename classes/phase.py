import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import c

c = c.value


class PHASE:

    def __init__(self, phase_data, frequency_data, traceback=True):
        """
        Parameters
        ----------
        phase_data : ndarray
            Phase data with shape (time_index, frequency_index)
            Units: degrees

        frequency_data : ndarray
            Frequency array corresponding to frequency axis
            Units: MHz

        traceback : bool
            If True, prints progress information.
        """

        self.phase_data = phase_data
        self.frequency_data = frequency_data
        self.traceback = traceback

        if self.traceback:
            print("PHASE class initialized")
            print("Phase data shape:", phase_data.shape)
            print("Frequency array length:", len(frequency_data))

    # -----------------------------------------------------
    # Phase unwrap function
    # -----------------------------------------------------

    def unwrap(self, phase_array):
        """
        Unwrap phase in degrees.
        """

        unwrapped = np.unwrap(np.radians(phase_array))
        unwrapped = np.degrees(unwrapped)

        adjustment = phase_array[-1] - unwrapped[-1]
        unwrapped += adjustment

        return unwrapped

    # -----------------------------------------------------
    # Diagnostic slope fit for a single timestamp
    # -----------------------------------------------------

    def diagnostic_fit(self,
                       time_index=0,
                       start_bw_ind=250,
                       stop_bw_ind=325,
                       start_index=0,
                       end_index=-1):

        """
        Perform slope fitting for a single timestamp
        and produce diagnostic plots.
        """

        phase_slice = self.phase_data[time_index, start_bw_ind:stop_bw_ind]
        freq_slice = self.frequency_data[start_bw_ind:stop_bw_ind]

        if end_index == -1:
            end_index = len(phase_slice)

        phase_slice = phase_slice[start_index:end_index]
        freq_slice = freq_slice[start_index:end_index]

        # Convert MHz → Hz
        freq_hz = freq_slice * 1e6

        # Unwrap phase
        unwrapped = self.unwrap(phase_slice)

        # Plot wrapped vs unwrapped
        plt.figure(figsize=(10,6))
        plt.plot(freq_hz, phase_slice, marker='o', label='Wrapped')
        plt.plot(freq_hz, unwrapped, marker='o', label='Unwrapped')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (degrees)')
        plt.title(f'Phase Diagnostic (time index {time_index})')
        plt.legend()
        plt.show()

        # Fit slope
        m, _ = np.polyfit(freq_hz, np.radians(unwrapped), 1)

        # Compute intercept for plotting
        b = np.mean(np.radians(unwrapped)) - m*np.mean(freq_hz)

        # Compute delay
        tau = m / (2*np.pi)
        distance = tau * c

        print("Estimated tau:", tau, "sec")
        print("Estimated distance:", distance, "m")

        # Plot fit
        plt.figure(figsize=(10,6))
        plt.plot(freq_hz, (m*freq_hz)+b, marker='o', label='Linear Fit')
        plt.plot(freq_hz, np.radians(unwrapped), marker='o', label='Unwrapped')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (radians)')
        plt.title('Linear Fit Diagnostic')
        plt.legend()
        plt.show()

    # -----------------------------------------------------
    # Main slope fitting over all timestamps
    # -----------------------------------------------------
    def slope_fit_all_times(self,
                            start_bw_ind=250,
                            stop_bw_ind=325,
                            start_index=0,
                            end_index=-1):

        """
        Fit phase slope vs frequency for every timestamp.

        Saves results internally as:
            self.tau
            self.distance
            self.slopes
        """

        n_times = self.phase_data.shape[0]

        tau_list = []
        distance_list = []
        slopes = []

        for i in range(n_times):

            if self.traceback:
                print(f"\rProcessing {i+1}/{n_times} ({100*(i+1)/n_times:.1f}%)", end="")

            phase_slice = self.phase_data[i, start_bw_ind:stop_bw_ind]
            freq_slice = self.frequency_data[start_bw_ind:stop_bw_ind]

            if end_index == -1:
                end_index = len(phase_slice)

            phase_slice = phase_slice[start_index:end_index]
            freq_slice = freq_slice[start_index:end_index]

            # Convert MHz → Hz
            freq_hz = freq_slice * 1e6

            # Unwrap phase
            unwrapped = self.unwrap(phase_slice)

            # Linear fit
            m, _ = np.polyfit(freq_hz, np.radians(unwrapped), 1)

            # Time delay
            tau = m / (2*np.pi)

            # Distance
            distance = tau * c

            tau_list.append(tau)
            distance_list.append(distance)
            slopes.append(m)

        # Move to new line after progress counter finishes
        if self.traceback:
            print()

        self.tau = np.array(tau_list)
        self.distance = np.array(distance_list)
        self.slopes = np.array(slopes)

        if self.traceback:
            print("Slope fitting complete.")

    # -----------------------------------------------------
    # Plot results
    # -----------------------------------------------------

    def plot_results(self):

        if not hasattr(self, "distance"):
            print("Run slope_fit_all_times() first")
            return

        plt.figure(figsize=(10,6))
        plt.plot(self.distance, marker='o')
        plt.xlabel("Time Index")
        plt.ylabel("Distance (m)")
        plt.title("Estimated Distance vs Time")
        plt.show()