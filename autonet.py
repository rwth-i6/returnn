import sys
from math import log, ceil, sqrt

S = 2

N = int(sys.argv[1])
M = int(sys.argv[2])
T = int(sys.argv[3])

I = ceil(log(N) / log(S))
#I = N
J = ceil(log(M) / log(S))

E = ceil(I * sqrt(T))
C = ceil(2 * log(J) * T / J)
Q = ceil(sqrt(T))
H = ceil(log(2 * J) * T / J)

D = ceil(log(I * J) * T )
R = ceil(log(I + J) * T )

print "proto", I
print "expand", E, Q
print "memory", D, R
print "collapse", C, H
print "template", J, H

