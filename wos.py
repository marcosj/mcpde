from random import uniform
from math import fmod, floor, pi, cos, sin, log, sqrt
from cmath import rect
import matplotlib.pyplot as plt

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

# harmonic Green's function for a 2D ball of radius R
G = lambda r, R: 0 if r == 0 else log(R/r) / 2 / pi

scene = [
	[0.5 + 0.1j, 0.9 + 0.5j],
	[0.5 + 0.9j, 0.1 + 0.5j],
	[0.1 + 0.5j, 0.5 + 0.1j],
	[0.5 + 0.9j, 0.8 + 1.0j],
	[0.8 + 1.0j, 0.9 + 0.5j]
]

def main():
	s = 64		# image size
	u = [[0] * s for i in range(s)]
	for j in range(s):
		for i in range(s):
			x0 = complex(i/s, j/s)
			u[i][j] = wos(x0, scene, lambda z: 0, g)
	plt.imshow(u,origin='lower')
	plt.show()

# solves a Laplace equation Δu = 0 at x0, where the boundary is given
# by a collection of segments, and the boundary conditions are given
# by a function g that can be evaluated at any point in space
def wos(x0, segments, f, g):
	eps = 0.001		# stopping tolerance
	nWalks = 32		# number of Monte Carlo samples
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
			# sample a point y uniformly from the ball of radius R around x
			r = R * sqrt(uniform(0,1))
			alpha = uniform(0, 2 * pi)
			y = x + rect(r, alpha)
			cum += pi * R * R * f(y) * G(r, R)
			# sample the next point x uniformly from the sphere around x
			theta = uniform(0, 2.* pi)
			x += rect(R, theta)
			steps += 1
			if R < eps or steps > maxSteps:
				break
		cum += g(x)
	return cum / nWalks		# Monte Carlo estimate

main()
