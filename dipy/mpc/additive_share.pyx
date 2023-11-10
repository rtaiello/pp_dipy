cimport cython
from libc.stdlib cimport rand
from libc.stdio cimport printf
cimport dipy.mpc.additive_share
from dipy.mpc.additive_share cimport *
cimport numpy as cnp

cdef cnp.npy_uint64 Q = 6497992661811505123
cdef int BASE = 10
cdef int PRECISION_INTEGRAL = 2
cdef int PRECISION_FRACTIONAL = 5
cdef int PRECISION = PRECISION_INTEGRAL + PRECISION_FRACTIONAL
cdef int BOUND = BASE**PRECISION


cdef cnp.npy_int64 generate_random() nogil:
    return rand()

cdef cnp.npy_uint64 mod(cnp.npy_uint64 a, cnp.npy_uint64 b) nogil:
    cdef mpz_t a_mpz, b_mpz, result_mpz
    mpz_init(&a_mpz)
    mpz_init(&b_mpz)
    mpz_init(&result_mpz)
    mpz_set_si(&a_mpz, a)
    mpz_set_si(&b_mpz, b)
    mpz_mod(&result_mpz, &a_mpz, &b_mpz)
    cdef cnp.npy_uint64 result = mpz_get_si(&result_mpz)
    mpz_clear(&a_mpz)
    mpz_clear(&b_mpz)
    mpz_clear(&result_mpz)
    return result


cdef cnp.npy_uint64 encode(double rational) nogil:
    cdef cnp.npy_uint64 field_element =  mod(<cnp.npy_uint64> (rational * BASE**PRECISION_FRACTIONAL), Q)
    return field_element

cdef double decode(cnp.npy_uint64 field_element) nogil:
    cdef cnp.npy_int32 upscaled = field_element if field_element <= Q/2 else field_element - Q
    cdef double rational = <double> (upscaled / (BASE**PRECISION_FRACTIONAL))
    return rational

cdef Share share(cnp.npy_uint64 secret) nogil:
    cdef Share result
    cdef cnp.npy_int64 random_share0 = generate_random()
    result.share0 = mod(random_share0,Q)
    result.share1 = mod((secret - random_share0),Q)
    return result


cdef cnp.npy_uint64 reconstruct(Share share) nogil:
    cdef mpz_t share0_mpz, share1_mpz, Q_mpz
    mpz_init(&share0_mpz)
    mpz_init(&share1_mpz)
    mpz_init(&Q_mpz)
    mpz_set_si(&share0_mpz, share.share0)
    mpz_set_si(&share1_mpz, share.share1)
    mpz_set_si(&Q_mpz, Q)
    mpz_add(&share0_mpz, &share0_mpz, &share1_mpz)
    mpz_mod(&share0_mpz, &share0_mpz, &Q_mpz)
    cdef cnp.npy_uint64 result = mpz_get_si(&share0_mpz)
    mpz_clear(&share0_mpz)
    mpz_clear(&share1_mpz)
    mpz_clear(&Q_mpz)
    return result

cdef BeaversTriple generate_mul_triple() nogil:
    
    
    cdef cnp.npy_uint64 a = mod(generate_random(),Q)
    cdef cnp.npy_uint64 b = mod(generate_random(), Q)
    cdef mpz_t a_mpz, b_mpz, c_mpz, Q_mpz
    mpz_init(&a_mpz)
    mpz_init(&b_mpz)
    mpz_init(&c_mpz)
    mpz_init(&Q_mpz)
    mpz_set_si(&a_mpz, a)
    mpz_set_si(&b_mpz, b)
    mpz_set_si(&Q_mpz, Q)
    mpz_mul(&c_mpz, &a_mpz, &b_mpz)
    mpz_mod(&c_mpz, &c_mpz, &Q_mpz)
    cdef cnp.npy_uint64 c = mpz_get_si(&c_mpz)
    mpz_clear(&a_mpz)
    mpz_clear(&b_mpz)
    mpz_clear(&c_mpz)
    mpz_clear(&Q_mpz)
    cdef BeaversTriple result
    result.a = share(a)
    result.b = share(b)
    result.c = share(c)
    return result

cdef Share add(Share x, Share y) nogil:

    cdef Share result
    cdef mpz_t x_share0_mpz, x_share1_mpz, y_share0, y_share1, Q_mpz
    mpz_init(&x_share0_mpz)
    mpz_init(&x_share1_mpz)
    mpz_init(&y_share0)
    mpz_init(&y_share1)
    mpz_init(&Q_mpz)
    mpz_set_si(&x_share0_mpz, x.share0)
    mpz_set_si(&x_share1_mpz, x.share1)
    mpz_set_si(&y_share0, y.share0)
    mpz_set_si(&y_share1, y.share1)
    mpz_set_si(&Q_mpz, Q)
    mpz_add(&x_share0_mpz, &x_share0_mpz, &y_share0)
    mpz_mod(&x_share0_mpz, &x_share0_mpz, &Q_mpz)
    mpz_add(&x_share1_mpz, &x_share1_mpz, &y_share1)
    mpz_mod(&x_share1_mpz, &x_share1_mpz, &Q_mpz)

    result.share0 = mpz_get_si(&x_share0_mpz)
    result.share1 = mpz_get_si(&x_share1_mpz)

    mpz_clear(&x_share0_mpz)
    mpz_clear(&x_share1_mpz)
    mpz_clear(&y_share0)
    mpz_clear(&y_share1)
    mpz_clear(&Q_mpz)

    
    return result
    


cdef Share sub(Share x,Share y) nogil:
    cdef Share result
    result.share0 = mod((x.share0 - y.share0),Q)
    result.share1 = mod((x.share1 - y.share1),Q)
    return result

cdef Share add_public(Share x, cnp.npy_uint64 k) nogil:
    cdef Share result
    result.share0 = mod((x.share0 + k),Q)
    result.share1 = x.share1
    return result

cdef Share sub_public(Share x, cnp.npy_uint64 k) nogil:
    cdef Share result
    result.share0 = mod((x.share0 - k),Q)
    result.share1 = x.share1
    return result

cdef Share mul_public(Share x, cnp.npy_uint64 k) nogil:
    cdef Share result
    cdef mpz_t x_share0_mpz, x_share1_mpz, k_mpz, result_mpz, Q_mpz
    mpz_init(&x_share0_mpz)
    mpz_init(&x_share1_mpz)
    mpz_init(&k_mpz)
    mpz_init(&result_mpz)
    mpz_init(&Q_mpz)
    mpz_set_si(&x_share0_mpz, x.share0)
    mpz_set_si(&x_share1_mpz, x.share1)
    mpz_set_si(&k_mpz, k)
    mpz_set_si(&Q_mpz, Q)
    mpz_mul(&result_mpz, &x_share0_mpz, &k_mpz)
    mpz_mod(&result_mpz, &result_mpz, &Q_mpz)
    result.share0 = mpz_get_si(&result_mpz)
    mpz_mul(&result_mpz, &x_share1_mpz, &k_mpz)
    mpz_mod(&result_mpz, &result_mpz, &Q_mpz)
    result.share1 = mpz_get_si(&result_mpz)
    mpz_clear(&x_share0_mpz)
    mpz_clear(&x_share1_mpz)
    mpz_clear(&k_mpz)
    mpz_clear(&result_mpz)
    mpz_clear(&Q_mpz)

    return result

cdef Share mul(Share x, Share y, BeaversTriple triple) nogil:
    cdef Share d = sub(x, triple.a)
    cdef Share e = sub(y, triple.b)
    cdef cnp.npy_uint64 alpha = reconstruct(d)
    cdef cnp.npy_uint64 beta = reconstruct(e)
    cdef mpz_t alpha_mpz, beta_mpz, f_mpz, Q_mpz
    
    mpz_init(&alpha_mpz)
    mpz_init(&beta_mpz)
    mpz_init(&f_mpz)
    mpz_init(&Q_mpz)

    mpz_set_si(&alpha_mpz, alpha)
    mpz_set_si(&beta_mpz, beta)
    mpz_set_si(&Q_mpz, Q)

    mpz_mul(&f_mpz, &alpha_mpz, &beta_mpz)
    mpz_mod(&f_mpz, &f_mpz, &Q_mpz)
    cdef cnp.npy_uint64 f = mpz_get_si(&f_mpz)

    mpz_clear(&alpha_mpz)
    mpz_clear(&beta_mpz)
    mpz_clear(&f_mpz)
    mpz_clear(&Q_mpz)
    cdef Share g = mul_public(triple.a, beta)
    cdef Share h = mul_public(triple.b, alpha)

    return add(h, add(g, add_public(triple.c, f)))

cdef Share truncate(Share x) nogil:
    cdef cnp.npy_uint64 temp = <cnp.npy_uint64> ((BASE**PRECISION_FRACTIONAL))
    cdef cnp.npy_uint64 z0 = <cnp.npy_int64> (x.share0 / temp)
    cdef cnp.npy_uint64 z1 = Q - x.share1
    z1 = <cnp.npy_uint64> (z1 / temp)
    z1 = Q - z1
    cdef Share result
    result.share0 = z0
    result.share1 = z1
    
    return result


def test(a, b):
    cdef Share x = share(encode(a))
    printf("reconstruct x %f\n", decode(reconstruct(x)))
    cdef Share y = share(encode(b))
    printf("reconstruct y %f\n", decode(reconstruct(y)))
    cdef cnp.npy_int32 k = encode(0.002)
    
    z_add = add(x, y)
    printf("reconstruct add %f\n", decode(reconstruct(z_add)))
   
    z_public_add = add_public(x, k)
    printf("reconstruct public add %f\n", decode(reconstruct(z_public_add)))
    
    z_sub = sub(x, y)
    printf("reconstruct sub %f\n", decode(reconstruct(z_sub)))

    z_public_sub = sub_public(x, k)
    printf("reconstruct public sub %f\n", decode(reconstruct(z_public_sub)))
    
    z_public_mul = mul_public(x, k)
    z_public_truncate = truncate(z_public_mul)
    printf("reconstruct public mul %f\n", decode(reconstruct(z_public_truncate)))
    
    triple= generate_mul_triple()
    z_mul = mul(x, y, triple)
    z_truncate = truncate(z_mul)
    printf("reconstruct mul %f\n", decode(reconstruct(z_truncate)))