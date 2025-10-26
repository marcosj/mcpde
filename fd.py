'''
*   To approximate the solution to the Poisson equation
*              DEL(u) = F(x,y), a <= x <= b, c <= y <= d,
*   SUBJECT TO BOUNDARY CONDITIONS:
*                   u(x,y) = G(x,y),
*       if x = a or x = b for c <= y <= d,
*       if y = c or y = d for a <= x <= b
'''
def fd(f, g, a=0, b=1, c=0, d=1, m=100, n=100, T=1e-6, M=1000000):
	'''
	Finite difference method.

	Args:
		f(x,y) (func): ...
		g(x,y) (func): ...
		a, b, c, d (int): R = [a,b] x [c,d]
		m, n (int): m x n lattice
		T (float): tolerance
		M (int): maximum iterations

	Returns:
		φ (float matrix): Potrntial at each (x,y)
	'''
	assert m > 2, 'Error: m < 3'
	assert n > 2, 'Error: n < 3'
	# crea el retículo
	h = (b - a) / n
	k = (d - c) / m
	x = [a + h * i for i in range(n+1)]
	y = [c + k * j for j in range(m+1)]
	φ = [[0] * (m+1) for j in range(n+1)]
	# condiciones de frontera
	for i in range(n+1):
		φ[i][0] = g(x[i], y[0])
		φ[i][m] = g(x[i], y[m])
	for j in range(m+1):
		φ[0][j] = g(x[0], y[j])
		φ[n][j] = g(x[n], y[j])
	λ = h * h / k / k
	μ = 2 * (1 + λ)
	for e in range(M):
		# Resuelve el sistema por Gauß-Seidel
		z = (-h*h * f(x[1],y[m-1]) + g(a,y[m-1]) + λ * g(x[1],d) + λ * φ[1][m-2] + φ[2][m-1]) / μ
		N = abs(φ[1][m-1] - z)
		φ[1][m-1] = z
		for i in range(2, n-1):
			z = (-h*h * f(x[i],y[m-1]) + λ * g(x[i],d) + φ[i-1][m-1] + φ[i+1][m-1] + λ * φ[i][m-2]) / μ
			if N < abs(φ[i][m-1] - z):
				N = abs(φ[i][m-1] - z)
			φ[i][m-1] = z
		z = (-h*h * f(x[n-1],y[m-1]) + g(b,y[m-1]) + λ * g(x[n-1],d) + φ[i-2][m-1] + λ * φ[n-1][m-2]) / μ
		if N < abs(φ[n-1][m-1] - z):
			N = abs(φ[n-1][m-1] - z)
		φ[n-1][m-1] = z

		for j in range(m-2, 1, -1):
			z = (-h*h * f(x[1],y[j]) + g(a,y[j]) + λ * φ[1][j+1] + λ * φ[1][j-1] + φ[2][j]) / μ
			if N < abs(z - φ[1][j]):
				N = abs(z - φ[1][j])
			φ[1][j] = z
			for i in range(2, n-1):
				z = (-h*h * f(x[i],y[j]) + φ[i-1][j] + λ * φ[i][j+1] + φ[i+1][j] + λ * φ[i][j-1]) / μ
				if N < abs(z - φ[i][j]):
					N = abs(z - φ[i][j])
				φ[i][j] = z
			z = (-h*h * f(x[n-1],y[j]) + g(b,y[j]) + φ[n-2][j] + λ * φ[n-1][j+1] + λ * φ[n-1][j-1]) / μ
			if N < abs(φ[n-1][j] - z):
				N = abs(φ[n-1][j] - z)
			φ[n-1][j] = z

		z = (-h*h * f(x[1],y[1]) + g(a,y[1]) + λ * g(x[1],c) + λ * φ[1][2] + φ[2][1]) / μ
		if N < abs(φ[1][1] - z):
			N = abs(φ[1][1] - z)
		φ[1][1] = z
		for i in range(2, n-1):
			z = (-h*h * f(x[i],y[1]) + λ * g(x[i],c) + φ[i-1][1] + λ * φ[i][2] + φ[i+1][1]) / μ
			if N < abs(φ[i][1] - z):
				N = abs(φ[i][1] - z)
			φ[i][1] = z
		z = (-h*h * f(x[n-1],y[1]) + g(b,y[1]) + λ * g(x[n-1],c) + φ[n-2][1] + λ * φ[n-1][2]) / μ
		if N < abs(φ[n-1][1] - z):
			N = abs(φ[n-1][1] - z)
		φ[n-1][1] = z

		if N < T:
			return x, y, φ
	raise Exception('fd failed to converge')

from math import exp
f=lambda x,y:x*exp(y)
def g(x,y):
	if x==0:
		g = 0
	if x==2 or y==1:
		g=f(x,y)
	if y==0:
		g=x
	return g
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
x,y,w=fd(f, g, 0,2,0,1,5, 6,1e-10)
y, x = np.meshgrid(x,y)
z = x * np.exp(y)
#plt.pcolormesh(x, y, w)
#sns.heatmap(w)
plt.imshow(w,origin='lower')
plt.show()
plt.imshow(z,origin='lower')
plt.show()
