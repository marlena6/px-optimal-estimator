# Description: Functions for the optimal estimator calculation.

# Global imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import time

def bin_spectra(spectra, x_spectra, bin_size):
    """
    Bin spectra over the Np axis.

    Parameters:
    spectra (np.ndarray): The input 2D array of shape (Ns, Np).
    x_spectra (np.ndarray): The input 1D array of shape (Np).
    bin_size (int): The number of adjacent pixels to combine for binning.

    Returns:
    np.ndarray: The binned spectra array.
    np.ndarray: The binned x_spectra array.
    """
    Ns, Np = spectra.shape
    # Calculate the number of bins
    num_bins = Np / bin_size
    if num_bins % 1 != 0:
        raise ValueError("bin_size must evenly divide Np.")
        # # Truncate the spectra to make it evenly divisible by bin_size
        # spectra = spectra[:, :num_bins * bin_size]
    else:
        num_bins = int(num_bins)
        print("Number of bins:", num_bins)
    # Reshape and sum/average over the new axis
    binned_spectra = spectra.reshape(Ns, num_bins, bin_size).mean(axis=2)
    binned_x_spectra = x_spectra.reshape(num_bins, bin_size).mean(axis=1)

    return binned_spectra, binned_x_spectra


def cij_alpha(Np, kbin, delta_x_matrix, pix_width, plot=False):
    """
    Calculate the Cij_alpha matrix for a given kbin.

    Parameters:
    Np (int): The number of pixels.
    kbin (list): The kbin range in form [kmin, kmax].
    delta_x_matrix (np.ndarray): The matrix of pixel separations, shape (Np, Np).
    pix_width (float): The pixel width in Mpc.
    plot (bool): Whether to plot the Cij_alpha matrix.

    Returns:
    np.ndarray: The Cij_alpha matrix.
    """

    # Naim's Eq 14 with an exponential factor

    # normally would define the interpolation kernel here
    # not going to bother with this, do this at one redshift only for now

    # define the matrix
    kbin_spacing = kbin[1] - kbin[0]
    Cij_alpha = np.zeros((Np, Np))
    # set up the FFT with fine spacing and large range
    Nk_full = 100000
    kmax = 200
    kpar_range_full = np.linspace(0, kmax, Nk_full)
    kspacing_full = kmax / Nk_full
    f_k = w_k(kpar_range_full, pix_width) ** 2
    # set the function to zero outside the kbin
    f_k[kpar_range_full > kbin[1]] = 0
    f_k[kpar_range_full < kbin[0]] = 0
    if plot:
        plt.plot(kpar_range_full, f_k)
        plt.xlabel("kpar")
        plt.ylabel("$W_k^2$")
        plt.axvline(kbin[0], color="black", linestyle="--")
        plt.axvline(kbin[1], color="black", linestyle="--")
        plt.xlim([0, 10])
        plt.show()
        plt.clf()
    f_x_full = (
        np.fft.irfft(f_k) * Nk_full * kspacing_full / (np.pi) 
    )  # into units of Mpc-1 ?
    delta_x_spacing = np.pi / (kpar_range_full[-1] - kpar_range_full[0]) # something wrong with this line?
    # print(delta_x_spacing, "compared to Nyquist frequency", np.pi/kmax)
    Nx = 2 * (Nk_full - 1)
    x = np.arange(Nx) * delta_x_spacing
    # print(x[1]-x[0], "is the spacing between x values")

    # make f_x into a function
    f_x = CubicSpline(x, f_x_full)
    Cij_alpha = f_x(delta_x_matrix)

    if plot:
        # get the args for first unique values of delta_x
        unique_delta_x, is_unique = np.unique(delta_x_matrix.flatten().round(decimals=4), return_index=True)
        cij_alpha_unique = Cij_alpha.flatten()[is_unique]

        plt.plot(unique_delta_x, cij_alpha_unique, "*", alpha=1, color="k")

        # uncomment the following to check against brute-force integration
        kpar_range = np.linspace(kbin[0], kbin[1], 200)
        w_kpar = w_k(kpar_range, pix_width)
        for delx in unique_delta_x:
            y = (
                1.0 / (np.pi) * np.cos(kpar_range * delx) * w_kpar**2
            )  # taking away the factor of 2 in denom, since we're representing with cos
            Q_ij = np.trapz(y, x=kpar_range) # needs units of Mpc^-1
            plt.plot(delx, Q_ij, "o", alpha=0.2, color="magenta")
        plt.xlim([0, np.amax(delta_x_matrix)])
        plt.plot(x, f_x_full, label="FFT", color="black")
        plt.xlabel(r"$\Delta$ x [Mpc]")
        plt.ylabel("Cij_alpha")
        plt.legend()
        plt.title("kbin = [{:.3f}, {:.3f}]".format(kbin[0], kbin[1]))
    # L = Np * pix_width
    # proposed normalization to account for differing k bin widths didn't really work: *(2*np.pi/L/kbin_spacing)
    return Cij_alpha 


def cij_alpha_approx(Np, kbin, delta_x_matrix, pix_width, plot=False):
    """
    Calculate the Cij_alpha matrix approximately for a given kbin.

    Parameters:
    Np (int): The number of pixels.
    kbin (list): The kbin range in form [kmin, kmax].
    delta_x_matrix (np.ndarray): The matrix of pixel separations, shape (Np, Np).
    pix_width (float): The pixel width in Mpc.
    plot (bool): Whether to plot the Cij_alpha matrix.

    Returns:
    np.ndarray: The Cij_alpha matrix.
    """

    # define the matrix
    kpar_central = (kbin[0] + kbin[1]) / 2.0
    delta_kpar = kbin[1] - kbin[0]
    # for now, set all the relevant matrix elements to a single value
    cij_alpha = (
        delta_kpar
        / (np.pi)
        * np.cos(kpar_central * delta_x_matrix)
        * w_k(kpar_central, pix_width) ** 2
    )
    return cij_alpha


def c0(xi_fid, delta_x_matrix):
    # xi_fid should be a function
    c0 = xi_fid(delta_x_matrix)
    return c0


# def make_delta_x_matrix(box_length, num_pix, xpar):
#     delta_x_matrix = np.zeros((num_pix, num_pix))
#     for i in range(num_pix):
#         for j in range(num_pix):
#             delta_x_raw = abs(xpar[i] - xpar[j])
#             if delta_x_raw <= (box_length / 2.0):
#                 delta_x_matrix[i, j] = abs(xpar[i] - xpar[j])
#             else:
#                 delta_x_matrix[i, j] = box_length % abs(
#                     xpar[i] - xpar[j]
#                 )  # account for periodic bdry conds
#     return delta_x_matrix

def make_delta_x_matrix(box_length, num_pix, xpar):
    delta_x_matrix = np.zeros((num_pix, num_pix))
    for i in range(num_pix):
        for j in range(num_pix):
            # Calculate the raw distance
            delta_x_raw = abs(xpar[i] - xpar[j])
            # Apply periodic boundary conditions
            delta_x_matrix[i, j] = min(delta_x_raw, box_length - delta_x_raw)
    return delta_x_matrix


def s_fid(arinyo, z, arinyo_params, delta_x_matrix, pix_width, plot=False):
    # s_fid = np.zeros((len(xpar), len(xpar)))
    # Np_fine=20000
    # xfine=np.linspace(0.0,L,Np_fine)
    # spacing_fine=L/Np_fine
    # kfine = np.fft.rfftfreq(Np_fine, spacing_fine)*2*np.pi
    Nk_full = 100000
    kmax = 200
    kfine = np.linspace(0, kmax, Nk_full)
    pixwidth_arr = np.full(len(kfine), pix_width)
    p1d_fid = arinyo.P1D_Mpc(z, kfine, parameters=arinyo_params)
    windowed_pk = p1d_fid*w_k(kfine, pixwidth_arr)**2
    # FFT to get the windowed correlation function
    kpar_range_full = np.linspace(0, kmax, Nk_full)
    kspacing_full = kmax / Nk_full
    windowed_xi = np.fft.irfft(windowed_pk) * Nk_full * kspacing_full / np.pi
    delta_x_spacing = np.pi / (kpar_range_full[-1] - kpar_range_full[0])
    Nx = 2 * (Nk_full - 1)
    x = np.arange(Nx) * delta_x_spacing
    if plot:
        plt.plot(x, windowed_xi, label="FFT")
        plt.xlabel("x [Mpc]")
        plt.ylabel("xi")
        plt.show()
        plt.clf()

    # make xi into a function
    xi_fid = CubicSpline(x, windowed_xi)
    # calculate the matrix
    s_fid = xi_fid(delta_x_matrix)
    # for i in range(len(xpar)):
    #     for j in range(len(xpar)):
    #         delx_ij = np.abs(xpar[j] - xpar[i])
    #         delx_ij_arr = np.full(len(kfine), delx_ij)
    #         # integrate over kpar
    #         integrand = (
    #             np.cos(kfine * delx_ij_arr)
    #             * w_k(kfine, pixwidth_arr) ** 2
    #             * p1d_fid
    #         )
    #         int = np.trapz(integrand, kfine) / np.pi
    #         # note: could maybe be doing this with FFTs instead, wasn't sure if that would reduce precision
    #         s_fid[i, j] = int
    # # cos_transform = dct(windowed_pk, type=2, norm='ortho')
    return s_fid


def w_k(k, pix_width):
    """
    Define the window function in k-space.

    Parameters:
    k (np.ndarray): The array of k values.
    pix_width (float): The pixel width in Mpc.

    Returns:
    np.ndarray: The window function.
    """

    # define the window function
    # R = 71600
    # delta_lambda = c/R # in velocity units
    # not that this is defined differently in Naim and Solene's papers. It should be:
    # W_k = np.exp(-kpar**2*delta_lambda**2/2)*np.sinc(kpar*pix_width/2)

    # for now, no resolution part
    # all spectra have the pixel width so we can just define this once
    return np.sinc(k * pix_width / 2.0 / np.pi)  # *np.exp(-k**2*(1/R)**2/2.)


# the working version
def get_P1D_est(Np, nsub, delta_x_matrix, pix_spacing, delta_flux, kbin_est, S_fiducial, C_0_invmat):
    nside = delta_flux.shape[0]
    print("Starting P1D.")

    kbin_est_centers = [(k[0]+k[1])/2. for k in kbin_est]
    print("center of kbins: ", kbin_est_centers)
    # set up the first derivative matrix
    # Lalpha = np.zeros(len(kbin_est_centers))
    Lalpha = np.zeros(len(kbin_est))
    F_alpha_beta  = np.zeros((len(kbin_est),len(kbin_est)))

    # get the Q_alphas:
    Cij_alpha_k = []
    print("Getting derivative matrices.")
    for k in range(len(kbin_est)):
        Cij_k = cij_alpha(Np, [kbin_est[k][0], kbin_est[k][1]], delta_x_matrix, pix_spacing)
        Cij_alpha_k.append(Cij_k)
    print("Starting loop through data.")
    for k in range(len(kbin_est)):
        # this loop we will find the P1D for k within [kbin_est[k], kbin_est[k+1]]
        start = time.time()
        print(f"Running (alpha)=k={[kbin_est[k][0], kbin_est[k][1]]}")
        counter = 0
        # get the derivative matrix from function Cij_alpha
        # won't always be the case, but for noiseless data, this is the same for any I=J pair
        Q_alpha = Cij_alpha_k[k]
        plt.imshow(Q_alpha)
        plt.colorbar()
        plt.show()
        plt.clf()
        for m in range(nside):
            # if m%50==0:
            #     print(f"m={m}")
            for n in range(nside):
                # for P1D, only nonzero when the two skewers are the same
                delta_I = delta_J = delta_flux[m,n][:, np.newaxis]
                if not np.isnan(delta_I).any():
                    # print(f"m={m}, n={n}")
                    counter+=1
                    if counter%20==0:
                        print(f"percent done: {counter/nsub*100}")
                    # print(f"Skewer I is at [{m,n}]") # and value {delta_I}
                    y_I = y_J = np.matmul(C_0_invmat,delta_I)
                    d_alpha = np.matmul(np.matmul(y_I.T, Q_alpha), y_J)
                    CQC = np.matmul(np.matmul(C_0_invmat, Q_alpha), C_0_invmat)
                    t_alpha = np.trace(np.matmul(CQC, S_fiducial))
                    # for kprime in range(1,5): # our beta
                    for kprime in range(len(kbin_est)):
                        # compute the second derivative
                        Q_beta = Cij_alpha_k[kprime]
                        # Q_beta = Cij_alpha(Np_b, kpar_matrix, [kbin_est[kprime], kbin_est[kprime+1]], delta_x_matrix, pix_spacing_b)
                        F_alpha_beta[k,kprime] += 0.5*np.trace(np.matmul(CQC, Q_beta))
                    # for this k, add the terms from this (m,n) skewer to the total derivative
                    # print(d_alpha, t_alpha)
                    Lalpha[k] += (d_alpha) #  - t_alpha
                    
                    
        end = time.time()
        print(f"This k took {end-start} seconds")
    F_inv = np.linalg.inv(F_alpha_beta)
    theta_est = 1/2*np.matmul(F_inv, Lalpha)
    return kbin_est_centers, theta_est, F_alpha_beta, Lalpha

# the cleaner version
def estimate_p1d(Np, nsub, delta_x_matrix, pix_spacing, delta_flux, kbin_est, S_fiducial, C_0_invmat):
    print("Starting P1D.")
    nside = delta_flux.shape[0]
    kbin_est_centers = [(k[0]+k[1])/2. for k in kbin_est]
    print("center of kbins: ", kbin_est_centers)
    # set up the first derivative matrix
    Lbeta = np.zeros(len(kbin_est))
    F_alpha_beta  = np.zeros((len(kbin_est),len(kbin_est)))

    # get the Q_alphas:
    Cij_alpha_k = []
    print("Getting derivative matrices.")
    for k in range(len(kbin_est)):
        Cij_k = cij_alpha(Np, [kbin_est[k][0], kbin_est[k][1]], delta_x_matrix, pix_spacing)
        Cij_alpha_k.append(Cij_k)
    print("Starting loop through data.")

    for kalpha in range(len(kbin_est)):
        start = time.time()
        Q_alpha = Cij_alpha_k[kalpha]
        CQC_alpha = np.matmul(np.matmul(C_0_invmat, Q_alpha), C_0_invmat)

        # get the data to do the first derivative. Only need to do this once for every k, so it shouldn't be in the kbeta loop
        for m in range(nside):
            for n in range(nside):
                delta_I = delta_J = delta_flux[m,n][:, np.newaxis]
                if not np.isnan(delta_I).any():
                    y_I = y_J = np.matmul(C_0_invmat,delta_I)
                    d_beta = np.matmul(np.matmul(y_I.T, Q_alpha), y_J)
                    t_beta = np.trace(np.matmul(CQC_alpha, S_fiducial))
                    Lbeta[kalpha] += (d_beta-t_beta)
                    for kbeta in range(len(kbin_est)): # run through all the k bins
                        Q_beta = Cij_alpha_k[kbeta]
                        F_alpha_beta[kalpha,kbeta] += 0.5*np.trace(np.matmul(CQC_alpha, Q_beta))
        end = time.time()
        print(f"This k took {end-start} seconds")
    F_inv = np.linalg.inv(F_alpha_beta)
    theta_est = 1/2*np.matmul(F_inv, Lbeta.T)
    return kbin_est_centers, theta_est, F_alpha_beta, Lbeta
