# masking calculations for 1D FFTs
import numpy as np
import numpy.fft as fft




def calculate_mask_mn(mask_array_fft, m, n, N, Nq):
    # calculate the mask matrix
    # mask_array_fft is the rFFT of the mask array. Nq x Nk
    # m is the index of k bins we want to compute
    # initialize mask matrix


    l = m-n
    sum_over_quasars = 0
    for q in range(Nq):
        if (l >= 0) & (l<=N//2):
            w_lq = mask_array_fft[q, l]
        elif (l >= 0) & (l>N//2):
            w_lq = np.conjugate(mask_array_fft[q, [l-(N//2)]])
        elif (l < 0)&(np.abs(l)<=N//2):
            w_lq = np.conjugate(mask_array_fft[q, np.abs(l)])
        elif (l < 0)&(np.abs(l)>N//2):
            w_lq = mask_array_fft[q, [abs(l)-(N//2)]]
        
        sum_over_quasars += (w_lq.__abs__())**2 / N
    return sum_over_quasars/Nq

def calculate_mask_mn_fft(mask_array_fft, m, n, N, Nq):
    # calculate the mask matrix
    # mask_array_fft is the FFT of the mask array. Nq x Nk
    # m is the index of k bins we want to compute
    l = m-n
    sum_over_quasars = 0
    for q in range(Nq):
        if l >= 0:
            w_lq = mask_array_fft[q, l]
        if l < 0:
            # w_lq_test = np.conjugate(mask_array_fft[q, np.abs(l)])
            # print(N-l, "is the index we are looking for")
            w_lq = mask_array_fft[q, N-abs(l)]
            # print(w_lq_test, w_lq, "are they the same?")
        sum_over_quasars += (w_lq.__abs__())**2
    return sum_over_quasars/Nq/N**2



def calculate_masked_power(m, mask_array_fft, theory_power):
    # theory_power is a length Np vector
    Nq = mask_array_fft.shape[0]
    N = mask_array_fft.shape[1]*2-1
    masked_power = 0
    for n in range(N):
        if n>N//2:
            # print("in special case. n=",n, "m-n=",m-n, "getting this element of the theory power", n-(N//2))
            masked_power += np.conjugate(theory_power[n-(N//2)]) * calculate_mask_mn(mask_array_fft, m, n, N, Nq)
        else:
            masked_power += theory_power[n] * calculate_mask_mn(mask_array_fft, m, n, N, Nq)
            # print("n=",n, "m-n=",m-n, "getting this element of the theory power", n)
        
    return masked_power
        
    # return matrix of n x q, where q is quasar index and n is number of pixels

def calculate_masked_power_fft(m, mask_array_fft, theory_power):
    # theory_power is a length Np vector
    Nq = mask_array_fft.shape[0]
    N = mask_array_fft.shape[1]
    masked_power = 0
    for n in range(N-1):
        masked_power += theory_power[n] * calculate_mask_mn_fft(mask_array_fft, m, n, N, Nq)
    return masked_power
        

# modify all skewers such that they begin at random locations along the line-of-sight, and wrap around. All should be 1/2 length of box.
def randomize_skewer_start(skewer_grid, rng):
    nside = skewer_grid.shape[0]
    Np = skewer_grid.shape[2]
    new_skewer_length = Np//2
    new_skewer_grid = np.zeros((nside, nside, new_skewer_length))
    for i in range(nside):
        for j in range(nside):
            start = rng.integers(Np)
            if start+new_skewer_length > Np:
                remainder = new_skewer_length - (Np-start)
                new_skewer_grid[i,j,:] = np.concatenate((skewer_grid[i,j,start:], skewer_grid[i,j,:remainder]))
            else:
                new_skewer_grid[i,j,:] = skewer_grid[i,j,start:start+new_skewer_length]
    return new_skewer_grid, new_skewer_length

# modify all skewers such that they are all only 1/2 length of box, start at random locations, and have zero padding elsewhere
def skewer_offset_and_pad(skewer_grid, rng, return_mask=True):
    mask = np.ones(skewer_grid.shape)
    new_skewer_grid = np.ma.MaskedArray.copy(skewer_grid)
    nside = skewer_grid.shape[0]
    Np = skewer_grid.shape[2]
    pad_length = Np//3
    for i in range(nside):
        for j in range(nside):
            pad_start = rng.integers(Np)
            if pad_start + pad_length > Np:
                remainder = pad_length - (Np-pad_start)
                # new_skewer_grid[i,j,pad_start:] = 0
                # new_skewer_grid[i,j,:remainder] = 0
                mask[i,j,pad_start:] = 0
                mask[i,j,:remainder] = 0
            else:
                # new_skewer_grid[i,j,pad_start:pad_start+pad_length] = 0
                mask[i,j,pad_start:pad_start+pad_length] = 0
    mask = np.asarray(mask)
    if return_mask:
        return new_skewer_grid*mask, mask
    else:
        return new_skewer_grid*mask