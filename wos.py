from random import uniform
from math import fmod, floor, pi, cos, sin
from cmath import rect

dot = lambda u, v: (u.conjugate() * v).real
clamp = lambda v, m, M: max(m, min(v, M))

# returns the point on segment s closest to x
def proj(x, s):
	u = s[1] - s[0]
	t = clamp(dot(x-s[0], u) / dot(u, u), 0, 1)
	return (1-t) * s[0] + t * s[1]

def g(z):
	s = 6
	return fmod( floor(s * z.real) + floor(s * z.imag), 2 )

scene = [
	[0.5 + 0.1j, 0.9 + 0.5j],
	[0.5 + 0.9j, 0.1 + 0.5j],
	[0.1 + 0.5j, 0.5 + 0.1j],
	[0.5 + 0.33333333j, 0.5, 0.6666666j],
	[0.33333333 + 0.5j, 0.6666666 + 0.5j]
]

def main():
	s = 128		# image size
	u = [[0] * s] * s
	for j in range(s):
		for i in range(s):
			x0 = complex(i/s, j/s)
			u[i, j] = wos(x0, scene, g)

# solves a Laplace equation Δu = 0 at x0, where the boundary is given
# by a collection of segments, and the boundary conditions are given
# by a function g that can be evaluated at any point in space
def wos(x0, segments, g):
	eps = 0.01		# stopping tolerance
	nWalks = 128	# number of Monte Carlo samples
	maxSteps = 16	# maximum walk length
	cum = 0
	for i in range(nWalks):
		x = x0
		steps = 0
		while True:
			R = 1e6
			for s in segments:
				p = proj(x, s)
				R = min(R, abs(x-p))
			theta = uniform(0, 2.* pi)
			x += rect(R, theta)
			steps += 1
			if R < eps or steps > maxSteps:
				break
		cum += g(x)
	return cum / nWalks		# Monte Carlo estimate

main()
