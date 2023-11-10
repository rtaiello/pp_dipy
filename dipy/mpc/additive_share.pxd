
cimport numpy as cnp


cdef struct Share:
    cnp.npy_uint64 share0
    cnp.npy_uint64 share1
cdef struct BeaversTriple:
    Share a
    Share b
    Share c

cdef cnp.npy_uint64 encode(double rational) nogil
cdef Share share(cnp.npy_uint64 secret) nogil
cdef cnp.npy_uint64 reconstruct(Share share) nogil
cdef BeaversTriple generate_mul_triple() nogil
cdef Share add(Share x, Share y) nogil
cdef Share sub(Share x, Share y) nogil
cdef Share mul(Share x, Share y, BeaversTriple triple) nogil
cdef Share truncate(Share x) nogil
cdef double decode(cnp.npy_uint64 field_element) nogil
cdef Share mul_public(Share x, cnp.npy_uint64 k) nogil

cdef extern from "gmp.h":
    ctypedef struct mpz_t:
        pass
    void mpz_init(mpz_t* x) nogil
    void mpz_clear(mpz_t* x) nogil
    void mpz_set_si(mpz_t* rop,  long long value) nogil
    long long mpz_get_si(mpz_t* x) nogil
    void mpz_add(mpz_t* rop, mpz_t* op1, mpz_t* op2) nogil
    void mpz_sub(mpz_t* rop, mpz_t* op1, mpz_t* op2) nogil
    void mpz_mul(mpz_t* rop, mpz_t* op1, mpz_t* op2) nogil
    void mpz_tdiv_q(mpz_t* rop, mpz_t* op1, mpz_t* op2) nogil
    void mpz_mod(mpz_t* rop, mpz_t* op1, mpz_t* op2) nogil
