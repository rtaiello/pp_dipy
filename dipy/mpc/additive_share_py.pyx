# spdz.pyx
from libc.stdlib cimport rand

cdef long long Q = 6497992661811505123  # < 64 bits
cdef int BASE = 10
cdef int PRECISION_INTEGRAL = 6
cdef int PRECISION_FRACTIONAL = 3
cdef int PRECISION = PRECISION_INTEGRAL + PRECISION_FRACTIONAL
cdef int BOUND = BASE**PRECISION

def share(long long secret):
    cdef long long share0 = rand() % Q
    cdef long long share1 = (secret - share0) % Q
    return [share0, share1]

def reconstruct(list shares):
    return sum(shares) % Q

def generate_mul_triple():
    cdef long long a = rand() % Q
    cdef long long b = rand() % Q
    cdef long long c = a * b % Q
    return share(a), share(b), share(c)

def generate_square_triple():
    cdef long long a = rand() % Q
    cdef long long b = pow(a, 2, Q)
    return (a, b)

def encode(double rational):
    cdef long long upscaled = <long long> (rational * BASE**PRECISION_FRACTIONAL)
    cdef long long field_element = upscaled % Q
    return field_element

def decode(long long field_element,precision_fractional=PRECISION_FRACTIONAL):
    cdef long long upscaled = field_element if field_element <= Q/2 else field_element - Q
    cdef float rational = upscaled / BASE**precision_fractional
    return rational

def add(list x, list y):
    cdef long long z0 = (x[0] + y[0]) % Q
    cdef long long z1 = (x[1] + y[1]) % Q
    return [z0, z1]

def sub(list  x,list y):
    print("x: ", x)
    print("y: ", y)
    cdef long long z0 = (x[0] - y[0]) % Q
    cdef long long z1 = (x[1] - y[1]) % Q
    print("z0: ", z0)
    print("z1: ", z1)
    return [z0, z1]

def add_public(x, k):
    cdef long long y0 = (x[0] + k) % Q
    cdef long long y1 = x[1]
    return [y0, y1]

def sub_public(list x, long long k):
    cdef long long y0 = (x[0] - k) % Q
    cdef long long y1 = x[1]
    return [y0, y1]

def mul_public(list x, long long k):
    cdef long long y0 = (x[0] * k) % Q
    cdef long long y1 = (x[1] * k) % Q
    return [y0, y1]

def mul(list x, list y, tuple triple):
    a, b, c = triple
    # local masking
    d = sub(x, a)
    e = sub(y, b)
    # communication: the players simultaneously send their shares to the other
    alpha = reconstruct(d)
    print("alpha: ", alpha)
    beta = reconstruct(e)
    print("beta: ", beta)
    # local combination
    cdef long long f = alpha * beta % Q
    g = mul_public(a, beta)
    h = mul_public(b, alpha)
    return add(h, add(g, add_public(c, f)))

def truncate(x):
    y0 = x[0] // BASE**PRECISION_FRACTIONAL
    y1 = Q - ((Q - x[1]) // BASE**PRECISION_FRACTIONAL)
    return [y0, y1]


