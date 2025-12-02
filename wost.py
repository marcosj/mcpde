# Python translation of https://github.com/GeometryCollective/wost-simple

from random import uniform
from math import fmod, floor, pi, cos, sin, log, sqrt
from cmath import rect, phase
import matplotlib.pyplot as plt

dot = lambda u, v: (u.conjugate() * v).real
cross = lambda u, v: u.real * v.imag - u.imag * v.real
clamp = lambda v, m, M: max(m, min(v, M))
rot90 = lambda v: complex(-v.imag, v.real)

# returns the point on segment s closest to x
def proj(x, a, b):
	u = b - a
	t = clamp(dot(x-a, u) / dot(u, u), 0, 1)
	return (1-t) * a + t * b

# returns True if the point b on the polyline abc is a silhoutte relative to x
def isSilhouette(x, a, b, c ):
	return cross(b-a, x-a) * cross(c-b, x-b) < 0

# returns the time t at which the ray x+tv intersects segment ab,
# or infinity if there is no intersection
def rayIntersection(x, v, a, b ):
	u = b - a
	w = x - a
	d = cross(v, u)
	s = cross(v, w) / d
	t = cross(u, w) / d
	if t > 0 and 0 <= s <= 1:
		return t
	return float('inf')

# returns distance from x to closest point on the given polylines P
def distancePolylines( x, P: list[list[complex]] ):
	d = float('inf')		# minimum distance so far
	for i in range(len(P)):		# iterate over polylines
		for j in range(len(P[i])-1):		# iterate over segments
			y = proj(x, P[i][j], P[i][j+1])		# distance to segment
			d = min( d, abs(x-y) )		# update minimum distance
	return d

# returns distance from x to closest silhouette point on the given polylines P
def silhouetteDistancePolylines( x, P: list[list[complex]] ):
	d = float('inf')		# minimum distance so far
	for i in range(len(P)):		# iterate over polylines
		for j in range(1, len(P[i])-1):		# iterate over segment pairs
			if isSilhouette( x, P[i][j-1], P[i][j], P[i][j+1] ):
				d = min( d, abs(x-P[i][j]) )		# update minimum distance
	return d

''' finds the first intersection y of the ray x+tv with the given polylines P,
	restricted to a ball of radius r around x.  The flag onBoundary indicates
	whether the first hit is on a boundary segment (rather than the sphere), and
	if so sets n to the normal at the hit point.
'''
def intersectPolylines(x, v, r, P, n, onBoundary):
	tMin = r		# smallest hit time so far
	n = complex(0, 0)		# first hit normal
	onBoundary = False		# will be true only if the first hit is on a segment
	for i in range(len(P)):		# iterate over polylines
		for j in range(len(P[i])-1):		# iterate over segments
			c = 1e-5		# ray offset (to avoid self-intersection)
			t = rayIntersection( x + c*v, v, P[i][j], P[i][j+1] )
			if t < tMin:		# closest hit so far
				tMin = t
				n = rot90( P[i][j+1] - P[i][j] )		# get normal
				n /= abs(n)		# make normal unit length
				onBoundary = True
	return x + tMin * v		# first hit location

# solves a Laplace equation Δu = 0 at x0, where the Dirichlet and Neumann
# boundaries are each given by a collection of polylines, the Neumann
# boundary conditions are all zero, and the Dirichlet boundary conditions
# are given by a function g that can be evaluated at any point in space
def wost(x0,	# evaluation point
		 boundaryDirichlet: list[list[complex]],	# absorbing part of the boundary
		 boundaryNeumann: list[list[complex]],		# reflecting part of the boundary
		 g		# Dirichlet boundary values
):
	eps = 0.0001		# stopping tolerance
	rMin = 0.0001		# minimum step size
	nWalks = 65536		# number of Monte Carlo samples
	maxSteps = 65536	# maximum walk length
	cum = 0				# running sum of boundary contributions
	for i in range(nWalks):
		x = x0		# start walk at the evaluation point
		n = complex(0, 0)		# assume x0 is an interior point, and has no normal
		onBoundary = False		# flag whether x is on the interior or boundary
		r = None		# radii used to define star shaped region
		dDirichlet = None
		dSilhouette = None
		steps = 0
		while True:		# loop until the walk hits the Dirichlet boundary
			# compute the radius of the largest star-shaped region
			dDirichlet = distancePolylines( x, boundaryDirichlet )
			dSilhouette = silhouetteDistancePolylines( x, boundaryNeumann )
			r = max( rMin, min( dDirichlet, dSilhouette ))
			# intersect a ray with the star-shaped region boundary
			theta = uniform(-pi, pi)
			if onBoundary:		# sample from a hemisphere around the normal
				theta = theta / 2 + phase(n)
			v = complex(cos(theta), sin(theta))		# unit ray direction
			x = intersectPolylines( x, v, r, boundaryNeumann, n, onBoundary )
			steps += 1
		while dDirichlet > eps and steps < maxSteps:
			# stop if we hit the Dirichlet boundary, or the walk is too long
			if steps >= maxSteps:
				print('Hit max steps')
		cum += g(x)		# accumulate contribution of the boundary value
	return cum / nWalks		# Monte Carlo estimate

def g(z):
	s = 6
	return fmod( floor(s * z.real), 2 )

''' for simplicity, in this code we assume that the Dirichlet and Neumann
	boundary polylines form a collection of closed polygons (possibly with holes),
	and are given with consistent counter-clockwise orientation'''
boundaryDirichlet = [
	[ 0.2 + 0.2j, 0.6 + 0.0j, 1.0 + 0.2j ],
	[ 1.0 + 1.0j, 0.6 + 0.8j, 0.2 + 1.0j ]
]
boundaryNeumann = [
	[ 1.0 + 0.2j, 0.8 + 0.6j, 1.0 + 1.0j ],
	[ 0.2 + 1.0j, 0.0 + 0.6j, 0.2 + 0.2j ]
]

# these routines are not used by WoSt itself, but are rather used to check
# whether a given evaluation point is actually inside the domain
def signedAngle(x, P):
	Theta = 0
	for i in range(len(P)):		# iterate over polylines
		for j in range(len(P[i])-1):		# iterate over segments
			Theta += phase( (P[i][j+1]- x) / (P[i][j] - x) )
	return Theta

# Returns true if the point x is contained in the region bounded by the Dirichlet
# and Neumann curves.  We assume these curves form a collection of closed polygons,
# and are given in a consistent counter-clockwise winding order.
def insideDomain(x, boundaryDirichlet, boundaryNeumann):
	Theta = signedAngle(x, boundaryDirichlet) + signedAngle(x, boundaryNeumann)
	delta = 1e-4		# numerical tolerance
	return abs(Theta - 2 * pi) < delta		# boundary winds around x exactly once

def main():
	s = 128		# image size
	u = [[0] * s for i in range(s)]
	for j in range(s):
		for i in range(s):
			x0 = complex((i+.5)/s, (j+.5)/s)
			u = 0
			if insideDomain(x0, boundaryDirichlet, boundaryNeumann):
				u[i][j] = wost(x0, boundaryDirichlet, boundaryNeumann, g)
	plt.imshow(u, origin='lower')
	plt.show()

main()
