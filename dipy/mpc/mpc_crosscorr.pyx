import numpy as np
cimport cython
cimport numpy as cnp
ctypedef cython.floating floating
ctypedef cython.numeric number
from libc.stdlib cimport rand
from libc.math cimport pow
cimport dipy.mpc.additive_share
from dipy.mpc.additive_share cimport *
from libc.stdio cimport printf
cdef int BITS = 64
cdef int SHARE_COMM = BITS * 2
cdef int MUL_COMM = BITS * 2
cdef int RECON_COMM = BITS * 2
cdef: 
    Share share_tmp
    ShareData = np.asarray(<Share[:1]>(&share_tmp)).dtype

cdef inline int _int_max(int a, int b) nogil:
    r"""
    Returns the maximum of a and b
    """
    return a if a >= b else b


cdef inline int _int_min(int a, int b) nogil:
    r"""
    Returns the minimum of a and b
    """
    return a if a <= b else b

cdef enum:
    SI = 0
    SI2 = 1
    SJ = 2
    SJ2 = 3
    SIJ = 4
    CNT = 5
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def precompute_cc_factors_3d(floating[:, :, :] static,
                                  floating[:, :, :] moving, int radius):
    r"""Precomputations to quickly compute the gradient of the CC Metric

    This version of precompute_cc_factors_3d is for testing purposes, it
    directly computes the local cross-correlation factors without any
    optimization, so it is less error-prone than the accelerated version.
    """
    cdef:
        cnp.npy_intp ns = static.shape[0]
        cnp.npy_intp nr = static.shape[1]
        cnp.npy_intp nc = static.shape[2]
        cnp.npy_intp s, r, c, k, i, j, t, num_bits
        cnp.npy_intp firstc, lastc, firstr, lastr, firsts, lasts
        double Imean, Jmean
        Share sum_share, share_static, share_moving, share_a, share_b, share_c, share_static_mean, share_moving_mean, share_forward, share_backward, share_shared
        BeaversTriple triple
        floating[:, :, :, :] factors = np.zeros((ns, nr, nc, 5),
                                                dtype=np.asarray(static).dtype)
      
        double[:] sums = np.zeros((6,), dtype=np.float64)

        Share[:, :, :] share_static_3d = np.zeros((ns, nr, nc), dtype=ShareData)
        Share[:, :, :] share_moving_3d = np.zeros((ns, nr, nc), dtype=ShareData)
        Share[:, :, :] share_factor_2 = np.zeros((ns, nr, nc), dtype=ShareData)

    with nogil:
        triple = generate_mul_triple()
        num_bits = 0
        for s in range(ns):
            for r in range(nr):
                for c in range(nc):
                    share_static_3d[s,r,c] = share(encode(static[s, r, c]))
                    num_bits+= SHARE_COMM
                    share_moving_3d[s,r,c] = share(encode(moving[s, r, c]))
                    num_bits+= SHARE_COMM
        for s in range(ns):
            firsts = _int_max(0, s - radius)
            lasts = _int_min(ns - 1, s + radius)
            for r in range(nr):
                firstr = _int_max(0, r - radius)
                lastr = _int_min(nr - 1, r + radius)
                for c in range(nc):
                    firstc = _int_max(0, c - radius)
                    lastc = _int_min(nc - 1, c + radius)
                    share_sum_sfm = share(encode(0.0))
                    num_bits+= SHARE_COMM
                    for t in range(6):
                        sums[t] = 0
                    for k in range(firsts, 1 + lasts):
                        for i in range(firstr, 1 + lastr):
                            for j in range(firstc, 1 + lastc):
                                sums[SI] += static[k, i, j]
                                sums[SI2] += static[k, i, j]**2
                                sums[SJ] += moving[k, i, j]
                                sums[SJ2] += moving[k, i, j]**2
                                sums[SIJ] += static[k, i, j]*moving[k, i, j]
                                share_static = share_static_3d[k, i, j]
                                share_moving = share_moving_3d[k, i, j]
                                share_sum_sfm = add(share_sum_sfm, truncate(mul(share_static,  share_moving, triple)))
                                num_bits+= MUL_COMM
                                sums[CNT] += 1
                    Imean = sums[SI] / sums[CNT]
                    Jmean = sums[SJ] / sums[CNT]
                    factors[s, r, c, 0] = static[s, r, c] - Imean
                    factors[s, r, c, 1] = moving[s, r, c] - Jmean
                    factors[s, r, c, 2] = (sums[SIJ] - Jmean * sums[SI] -
                                           Imean * sums[SJ] +
                                           sums[CNT] * Jmean * Imean)
                    factors[s, r, c, 3] = (sums[SI2] - Imean * sums[SI] -
                                           Imean * sums[SI] +
                                           sums[CNT] * Imean * Imean)
                    factors[s, r, c, 4] = (sums[SJ2] - Jmean * sums[SJ] -
                                           Jmean * sums[SJ] +
                                           sums[CNT] * Jmean * Jmean)

                    share_Imean = share(encode(Imean))
                    num_bits+= SHARE_COMM
                    share_Jmean = share(encode(Jmean))
                    num_bits+= SHARE_COMM                
                    share_sum_sf = share(encode(sums[SI]))
                    num_bits+= SHARE_COMM 
                    share_sum_sm = share(encode(sums[SJ]))
                    term_0 = share_sum_sfm
                    term_1 = truncate(mul(share_Jmean, share_sum_sf, triple))
                    num_bits+= MUL_COMM
                    term_2 = truncate(mul(share_Imean, share_sum_sm, triple))
                    num_bits+= MUL_COMM
                    term_3 = truncate(mul(share_Jmean, share_Imean, triple))
                    num_bits+= MUL_COMM
                    term_4 = truncate(mul_public(term_3, encode(sums[CNT])))
                    share_a = sub(term_0, term_1)
                    share_a = sub(share_a, term_2)
                    share_a = add(share_a, term_4)
                    share_factor_2[s, r, c] = share_a
                   
    return np.asarray(factors), np.asarray(share_factor_2), num_bits

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_cc_forward_backward_step_3d(floating[:, :, :, :] grad_static,
                                floating[:, :, :, :] grad_moving,
                               floating[:, :, :, :] factors,
                               Share [:, :, :] share_factor_2_3d,
                               cnp.npy_intp radius):
    r"""Gradient of the CC Metric w.r.t. the forward transformation

    Computes the gradient of the Cross Correlation metric for symmetric
    registration (SyN) [Avants2008]_ w.r.t. the displacement associated to
    the moving volume ('forward' step) as in [Avants2011]_

    Parameters
    ----------
    grad_static : array, shape (S, R, C, 3)
        the gradient of the static volume
    factors : array, shape (S, R, C, 5)
        the precomputed cross correlation terms obtained via
        precompute_cc_factors_3d
    radius : int
        the radius of the neighborhood used for the CC metric when
        computing the factors. The returned vector field will be
        zero along a boundary of width radius voxels.

    Returns
    -------
    out : array, shape (S, R, C, 3)
        the gradient of the cross correlation metric with respect to the
        displacement associated to the moving volume
    energy : the cross correlation energy (data term) at this iteration

    References
    ----------
    .. [Avants2008]_ Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C.
        (2008). Symmetric Diffeomorphic Image Registration with
        Cross-Correlation: Evaluating Automated Labeling of Elderly and
        Neurodegenerative Brain, Med Image Anal. 12(1), 26-41.
    .. [Avants2011]_ Avants, B. B., Tustison, N., & Song, G. (2011). Advanced
        Normalization Tools (ANTS), 1-35.
    """
    cdef:
        cnp.npy_intp ns = grad_static.shape[0]
        cnp.npy_intp nr = grad_static.shape[1]
        cnp.npy_intp nc = grad_static.shape[2]
        double energy = 0
        cnp.npy_intp s, r, c, num_bits
        BeaversTriple triple
        Share share_sfm, share_sff_inv, share_smm_inv, share_Ii, share_Ji, snd_term_forward, snd_term_backward
        double Ii, Ji, sfm, sff, smm, localCorrelation, temp, temp_fst, temp_snd_foward, temp_snd_backward, temp_forward, temp_backward
        floating[:, :, :, :] out_forward =\
            np.zeros((ns, nr, nc, 3), dtype=np.asarray(grad_static).dtype)
        floating[:, :, :, :] out_backward =\
            np.zeros((ns, nr, nc, 3), dtype=np.asarray(grad_static).dtype)
    with nogil:
        num_bits = 0
        triple = generate_mul_triple()
        for s in range(radius, ns-radius):
            for r in range(radius, nr-radius):
                for c in range(radius, nc-radius):
                    Ii = factors[s, r, c, 0]
                    share_Ii = share(encode(Ii))
                    num_bits+= SHARE_COMM
                    Ji = factors[s, r, c, 1]
                    share_Ji = share(encode(Ji))
                    num_bits+= SHARE_COMM
                    sfm = factors[s, r, c, 2]
                    share_sfm = share_factor_2_3d[s, r, c]
                    sff = factors[s, r, c, 3]
                    share_sff_inv = share(encode(1/factors[s, r, c, 3]))
                    num_bits+= SHARE_COMM
                    smm = factors[s, r, c, 4]
                    share_smm_inv = share(encode(1/factors[s, r, c, 4]))
                    num_bits+= SHARE_COMM
                    if sff == 0.0 or smm == 0.0:
                        continue
                    localCorrelation = 0
                    if sff * smm > 1e-5:
                        localCorrelation = sfm * sfm / (sff * smm)
                    if localCorrelation < 1:  # avoid bad values...
                        energy -= localCorrelation
                    # temp_forward = 2.0 * sfm / (sff * smm) * (Ji - sfm / sff * Ii)
                    fst_term = truncate(mul(share_sfm, share_sff_inv, triple))
                    num_bits+= MUL_COMM
                    fst_term = truncate(mul(fst_term, share_smm_inv, triple))
                    num_bits+= MUL_COMM
                    fst_term = truncate(mul_public(fst_term, encode(2)))
                    temp_fst = decode(reconstruct(fst_term))
                    snd_term_forward = truncate(mul(share_Ii, share_sfm, triple))
                    num_bits+= MUL_COMM
                    snd_term_forward = truncate(mul(snd_term_forward, share_sff_inv, triple))
                    num_bits+= MUL_COMM
                    snd_term_forward = sub(share_Ji, snd_term_forward)
                    temp_snd_foward = decode(reconstruct(snd_term_forward))
                    temp_forward = temp_fst * temp_snd_foward

                    out_forward[s, r, c, 0] -= temp_forward * grad_static[s, r, c, 0]
                    out_forward[s, r, c, 1] -= temp_forward * grad_static[s, r, c, 1]
                    out_forward[s, r, c, 2] -= temp_forward * grad_static[s, r, c, 2]

                    # temp_backward = 2.0 * sfm / (sff * smm) * (Ii - sfm / smm * Ji)
                    snd_term_backward = truncate(mul(share_Ji, share_sfm, triple))
                    num_bits+= MUL_COMM
                    snd_term_backward = truncate(mul(snd_term_backward, share_smm_inv, triple))
                    num_bits+= MUL_COMM
                    snd_term_backward = sub(share_Ii, snd_term_backward)
                    temp_snd_backward = decode(reconstruct(snd_term_backward))
                    num_bits+= RECON_COMM
                    temp_backward = temp_fst * temp_snd_backward


                    out_backward[s, r, c, 0] -= temp_backward * grad_moving[s, r, c, 0]
                    out_backward[s, r, c, 1] -= temp_backward * grad_moving[s, r, c, 1]
                    out_backward[s, r, c, 2] -= temp_backward * grad_moving[s, r, c, 2]


    return np.asarray(out_forward),np.asarray(out_backward), energy, num_bits

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int _wrap(int x, int m)nogil:
    if x < 0:
        return x + m
    return x


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int _update_factors(double[:, :, :, :] factors,
                                 Share[:, :, :] share_factor_2,
                                 floating[:, :, :] moving,
                                 floating[:, :, :] static,
                                 Share[:, :, :] share_moving_3d, 
                                 Share[:, :, :] share_static_3d,
                                 cnp.npy_intp ss, cnp.npy_intp rr, cnp.npy_intp cc,
                                 cnp.npy_intp s, cnp.npy_intp r, cnp.npy_intp c, int operation)nogil:
    cdef:
        double sval
        double mval
        cnp.npy_intp num_bits
        Share share_static
        Share share_moving
    num_bits = 0
    if s >= moving.shape[0] or r >= moving.shape[1] or c >= moving.shape[2]:
        if operation == 0:
            factors[ss, rr, cc, SI] = 0
            factors[ss, rr, cc, SI2] = 0
            factors[ss, rr, cc, SJ] = 0
            factors[ss, rr, cc, SJ2] = 0
            factors[ss, rr, cc, SIJ] = 0
            share_factor_2[ss, rr, cc] = share(encode(0.0))
            num_bits = SHARE_COMM
    else:
        sval = static[s, r, c]
        mval = moving[s, r, c]
        share_static = share_static_3d[s, r, c]
        share_moving = share_moving_3d[s, r, c]
        if operation == 0:
            triple = generate_mul_triple()
            factors[ss, rr, cc, SI] = sval
            factors[ss, rr, cc, SI2] = sval*sval
            factors[ss, rr, cc, SJ] = mval
            factors[ss, rr, cc, SJ2] = mval*mval
            factors[ss, rr, cc, SIJ] = sval*mval
            share_factor_2[ss, rr, cc] = truncate(mul(share_static, share_moving, triple))
            num_bits= MUL_COMM
        elif operation == -1:
            triple = generate_mul_triple()
            factors[ss, rr, cc, SI] -= sval
            factors[ss, rr, cc, SI2] -= sval*sval
            factors[ss, rr, cc, SJ] -= mval
            factors[ss, rr, cc, SJ2] -= mval*mval
            factors[ss, rr, cc, SIJ] -= sval*mval
            share_factor_2[ss, rr, cc] = sub(share_factor_2[ss, rr, cc], truncate(mul(share_static, share_moving, triple)))
            num_bits = MUL_COMM
        elif operation == 1:
            triple = generate_mul_triple()
            factors[ss, rr, cc, SI] += sval
            factors[ss, rr, cc, SI2] += sval*sval
            factors[ss, rr, cc, SJ] += mval
            factors[ss, rr, cc, SJ2] += mval*mval
            factors[ss, rr, cc, SIJ] += sval*mval
            share_factor_2[ss, rr, cc] = add(share_factor_2[ss, rr, cc], truncate(mul(share_static, share_moving, triple)))
            num_bits = MUL_COMM
    return num_bits



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def precompute_cc_factors_3d_fast(floating[:, :, :] static,
                             floating[:, :, :] moving,
                             cnp.npy_intp radius, num_threads=None):
    cdef:
        cnp.npy_intp ns = static.shape[0]
        cnp.npy_intp nr = static.shape[1]
        cnp.npy_intp nc = static.shape[2]
        cnp.npy_intp side = 2 * radius + 1
        cnp.npy_intp firstc, lastc, firstr, lastr, firsts, lasts
        cnp.npy_uint64 num_bits, result_num_bits
        cnp.npy_intp s, r, c, it, sides, sider, sidec
        double cnt
        cnp.npy_intp ssss, sss, ss, rr, cc, prev_ss, prev_rr, prev_cc
        double Imean, Jmean, IJprods, Isq, Jsq
        double[:, :, :, :] temp = np.zeros((2, nr, nc, 5), dtype=np.float64)
        floating[:, :, :, :] factors = np.zeros((ns, nr, nc, 5),
                                                dtype=np.asarray(static).dtype)
        Share term_0, term_1, term_2, term_3, term_4, share_Imean, share_Jmean, share_sum_sf, share_sum_sm
        Share[:, :, :] share_temp_2_3d = np.zeros((ns, nr, nc), dtype=ShareData)
        Share[:, :, :] share_factor_2_3d = np.empty((ns, nr, nc), dtype=ShareData)
        Share[:, :, :] share_static_3d = np.zeros((ns, nr, nc), dtype=ShareData)
        Share[:, :, :] share_moving_3d = np.zeros((ns, nr, nc), dtype=ShareData)
    with nogil:
        triple = generate_mul_triple()
        num_bits = 0
        result_num_bits = 0
        for s in range(ns):
            for r in range(nr):
                for c in range(nc):
                    share_static_3d[s,r,c] = share(encode(static[s, r, c]))
                    num_bits+= SHARE_COMM
                    share_moving_3d[s,r,c] = share(encode(moving[s, r, c]))
                    num_bits+= SHARE_COMM
        sss = 1
        for s in range(ns+radius):
            ss = _wrap(s - radius, ns)
            sss = 1 - sss
            firsts = _int_max(0, ss - radius)
            lasts = _int_min(ns - 1, ss + radius)
            sides = (lasts - firsts + 1)
            for r in range(nr+radius):
                rr = _wrap(r - radius, nr)
                firstr = _int_max(0, rr - radius)
                lastr = _int_min(nr - 1, rr + radius)
                sider = (lastr - firstr + 1)
                for c in range(nc+radius):
                    cc = _wrap(c - radius, nc)
                    # New corner
                    result_num_bits = _update_factors(temp, share_temp_2_3d,moving, static,share_moving_3d,share_static_3d,
                                    sss, rr, cc, s, r, c, 0)

                    num_bits+= result_num_bits

                    # Add signed sub-volumes
                    if s > 0:
                        prev_ss = 1 - sss
                        for it in range(5):
                            temp[sss, rr, cc, it] += temp[prev_ss, rr, cc, it]
                        share_temp_2_3d[sss, rr, cc] = add(share_temp_2_3d[sss, rr, cc], share_temp_2_3d[prev_ss, rr, cc])
                        if r > 0:
                            prev_rr = _wrap(rr-1, nr)
                            for it in range(5):
                                temp[sss, rr, cc, it] -= \
                                    temp[prev_ss, prev_rr, cc, it]
                            share_temp_2_3d[sss, rr, cc] = sub(share_temp_2_3d[sss, rr, cc], share_temp_2_3d[prev_ss, prev_rr, cc])
                            if c > 0:
                                prev_cc = _wrap(cc-1, nc)
                                for it in range(5):
                                    temp[sss, rr, cc, it] += \
                                        temp[prev_ss, prev_rr, prev_cc, it]
                                share_temp_2_3d[sss, rr, cc] = add(share_temp_2_3d[sss, rr, cc], share_temp_2_3d[prev_ss, prev_rr, prev_cc])
                        if c > 0:
                            prev_cc = _wrap(cc-1, nc)
                            for it in range(5):
                                temp[sss, rr, cc, it] -= \
                                    temp[prev_ss, rr, prev_cc, it]
                            share_temp_2_3d[sss, rr, cc] = sub(share_temp_2_3d[sss, rr, cc], share_temp_2_3d[prev_ss, rr, prev_cc])
                    if r > 0:
                        prev_rr = _wrap(rr-1, nr)
                        for it in range(5):
                            temp[sss, rr, cc, it] += \
                                temp[sss, prev_rr, cc, it]
                        share_temp_2_3d[sss, rr, cc] = add(share_temp_2_3d[sss, rr, cc], share_temp_2_3d[sss, prev_rr, cc])
                        if c > 0:
                            prev_cc = _wrap(cc-1, nc)
                            for it in range(5):
                                temp[sss, rr, cc, it] -= \
                                    temp[sss, prev_rr, prev_cc, it]
                            share_temp_2_3d[sss, rr, cc] = sub(share_temp_2_3d[sss, rr, cc], share_temp_2_3d[sss, prev_rr, prev_cc])
                    if c > 0:
                        prev_cc = _wrap(cc-1, nc)
                        for it in range(5):
                            temp[sss, rr, cc, it] += temp[sss, rr, prev_cc, it]
                        share_temp_2_3d[sss, rr, cc] = add(share_temp_2_3d[sss, rr, cc], share_temp_2_3d[sss, rr, prev_cc])

                    # Add signed corners
                    if s >= side:
                        result_num_bits = _update_factors(temp, share_temp_2_3d,moving, static,share_moving_3d,share_static_3d,
                                        sss, rr, cc, s-side, r, c, -1)
                        num_bits+= result_num_bits
                        if r >= side:
                            result_num_bits = _update_factors(temp, share_temp_2_3d,moving, static,share_moving_3d,share_static_3d,
                                            sss, rr, cc, s-side, r-side, c, 1)
                            num_bits+= result_num_bits
                            if c >= side:
                                result_num_bits = _update_factors(temp, share_temp_2_3d,moving, static,share_moving_3d,share_static_3d,
                                sss, rr, cc, s-side, r-side, c-side, -1)
                                num_bits+= result_num_bits
                        if c >= side:
                            result_num_bits = _update_factors(temp, share_temp_2_3d,moving, static,share_moving_3d,share_static_3d,
                                            sss, rr, cc, s-side, r, c-side, 1)
                            num_bits+= result_num_bits
                    if r >= side:
                        result_num_bits = _update_factors(temp, share_temp_2_3d,moving, static,share_moving_3d,share_static_3d,
                                        sss, rr, cc, s, r-side, c, -1)
                        num_bits+= result_num_bits
                        if c >= side:
                            result_num_bits = _update_factors(temp, share_temp_2_3d,moving, static,share_moving_3d,share_static_3d,
                                            sss, rr, cc, s, r-side, c-side, 1)
                            num_bits+=result_num_bits

                    if c >= side:
                        result_num_bits = _update_factors(temp, share_temp_2_3d,moving, static,share_moving_3d,share_static_3d,
                                        sss, rr, cc, s, r, c-side, -1)
                        num_bits+=result_num_bits
                    # Compute final factors
                    if s >= radius and r >= radius and c >= radius:
                        firstc = _int_max(0, cc - radius)
                        lastc = _int_min(nc - 1, cc + radius)
                        sidec = (lastc - firstc + 1)
                        cnt = sides*sider*sidec
                        Imean = temp[sss, rr, cc, SI] / cnt
                        Jmean = temp[sss, rr, cc, SJ] / cnt
                        IJprods = (temp[sss, rr, cc, SIJ] -
                                   Jmean * temp[sss, rr, cc, SI] -
                                   Imean * temp[sss, rr, cc, SJ] +
                                   cnt * Jmean * Imean)
                        Isq = (temp[sss, rr, cc, SI2] -
                               Imean * temp[sss, rr, cc, SI] -
                               Imean * temp[sss, rr, cc, SI] +
                               cnt * Imean * Imean)
                        Jsq = (temp[sss, rr, cc, SJ2] -
                               Jmean * temp[sss, rr, cc, SJ] -
                               Jmean * temp[sss, rr, cc, SJ] +
                               cnt * Jmean * Jmean)
                        factors[ss, rr, cc, 0] = static[ss, rr, cc] - Imean
                        factors[ss, rr, cc, 1] = moving[ss, rr, cc] - Jmean
                        factors[ss, rr, cc, 2] = IJprods
                        factors[ss, rr, cc, 3] = Isq
                        factors[ss, rr, cc, 4] = Jsq
                        share_Imean = share(encode(Imean))
                        num_bits+= SHARE_COMM
                        share_Jmean = share(encode(Jmean))
                        num_bits+= SHARE_COMM
                        share_sum_sf = share(encode(temp[sss, rr, cc, SI]))
                        num_bits+= SHARE_COMM
                        share_sum_sm = share(encode(temp[sss, rr, cc, SJ]))
                        num_bits+= SHARE_COMM
                        term_0 = share_temp_2_3d[sss, rr, cc]
                        term_1 = truncate(mul(share_Jmean, share_sum_sf, triple))
                        num_bits+= MUL_COMM
                        term_2 = truncate(mul(share_Imean, share_sum_sm, triple))
                        num_bits+= MUL_COMM
                        term_3 = truncate(mul(share_Jmean, share_Imean, triple))
                        num_bits+= MUL_COMM
                        term_4 = truncate(mul_public(term_3, encode(cnt)))
                        share_factor_2 = sub(term_0, term_1)
                        share_factor_2 = sub(share_factor_2, term_2)
                        share_factor_2 = add(share_factor_2, term_4)
                        if factors[ss, rr, cc, 2] < 1e-6:
                            share_factor_2 = share(encode(0.0))
                        share_factor_2_3d[ss, rr, cc] = share_factor_2
    return np.asarray(factors), np.asarray(share_factor_2_3d), num_bits