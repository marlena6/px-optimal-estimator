# masking calculations for 1D FFTs
import numpy as np




def calculate_mask_mn(mask_array_fft, m, n, Np, Nq):
    # calculate the mask matrix
    # mask_array_fft is the FFT of the mask array. Nq x Nk
    # m is the index of k bins we want to compute
    # initialize mask matrix


    l = m-n
    sum_over_quasars = 0
    for q in range(mask_array_fft.shape[0]):
        if l >= 0:
            w_lq = mask_array_fft[q, l]
        if l < 0:
            w_lq = np.conjugate(mask_array_fft[q, np.abs(l)])
        sum_over_quasars += (w_lq.__abs__())**2 / Np
    return sum_over_quasars/Nq


def calculate_masked_power(m, mask_array_fft, theory_power):
    # theory_power is a length Np vector
    Nq = mask_array_fft.shape[0]
    Nk = mask_array_fft.shape[1]
    masked_power = 0
    for n in range(Nk):
        masked_power += theory_power[n] * calculate_mask_mn(mask_array_fft, m, n, Nk, Nq)
    return masked_power
        
    # return matrix of n x q, where q is quasar index and n is number of pixels