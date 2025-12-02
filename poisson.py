''' Ecuación de Poisson
	∇²φ(x,y) = f(x,y)	∈ [a,b]×[c,d]
	∂φ(x,y) = g(x,y)
'''

from time import time
from math import log as ln
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from fd import fd
from mc import mc

def main():
	'''print('Experimento 1:')
	# Variando el tamaño de la rejilla
	for n in range(10, 51, 10):
		print(f'{n=}')
		ej4(n, 100)
	print('Experimento 2:')
	# Variando la longitud de las caminatas
	for N in range(100, 901, 200):
		print(f'{N=}')
		ej4(25, N)'''
	print('Ejemplo 1:')
	ej1(25, 250)
	print('Ejemplo 2:')
	ej2(25, 250)
	print('Ejemplo 3:')
	ej3(25, 250)
	print('Ejemplo 4:')
	ej4(25, 250)

def ej1(n, N):
	''' Ejemplo 1
		∇²φ(x,y) = 0	∈ [0,½]×[0,½]
		∂φ(0,y) = ∂φ(x,0) = 0
		∂φ(x,½) = 200x		∂φ(½,y) = 200y
	'''
	a, b, c, d = 0, 0.5, 0, 0.5
	f = lambda x, y: 0
	g = lambda x, y: 0 if x == a or y == c else 200*x if y == d else 200*y if x == b else 0
	φ = lambda x, y: 400*x*y	# solución analítica
	poisson(f, g, a, b, c, d, n, n, φ, N)

def ej2(n, N):
	''' Ejemplo 2
		∇²φ(x,y) = x·e^y	∈ [0,2]×[0,1]
		∂φ(0,y) = 0		∂φ(2,y) = 2·e^y
		∂φ(x,0) = x		∂φ(x,1) = x·e^1
	'''
	a, b, c, d = 0, 2, 0, 1
	f = lambda x, y: x * np.exp(y)
	g = lambda x, y: 0 if x == a else f(x,y) if x == b else x if y == c else f(x,y) if y == d else 0
	φ = lambda x, y: x * np.exp(y)		# solución analítica
	poisson(f, g, a, b, c, d, n, n, φ, N)

def ej3(n, N):
	''' Ejemplo 3
		∇²φ(x,y) = 4	∈ [0,1]×[0,2]
		∂φ(0,y) = y^2		∂φ(1,y) = (y-1)^2
		∂φ(x,0) = x^2		∂φ(x,2) = (x-2)^2
	'''
	a, b, c, d = 0, 1, 0, 2
	f = lambda x, y: 4
	g = lambda x, y: y*y if x == a else (y-1)*(y-1) if x == b else x*x if y == c else (x-2)*(x-2) if y == d else 0
	φ = lambda x, y: np.square(x - y)		# solución analítica
	poisson(f, g, a, b, c, d, n, n, φ, N)

def ej4(n, N):
	''' Ejemplo 4
		∇²φ(x,y) = 0	∈ [1,2]×[0,1]
		∂φ(1,y) = ln(y²+1)		∂φ(2,y) = ln(y²+4)
		∂φ(x,0) = 2 ln x		∂φ(x,1) = ln(x²+1)
	'''
	a, b, c, d = 1, 2, 0, 1
	f = lambda x, y: 0
	g = lambda x, y: ln(y*y+1) if x == a else ln(y*y+4) if x == b else 2*ln(x) if y == c else ln(x*x+1) if y == d else 0
	φ = lambda x, y: np.log(x*x + y*y)		# solución analítica
	poisson(f, g, a, b, c, d, n, n, φ, N)

def poisson(f, g, a, b, c, d, m, n, φ = None, N = None):
	assert m > 2, 'Error: m < 3'
	assert n > 2, 'Error: n < 3'
	# crea el retículo
	h = (b - a) / n
	k = (d - c) / m
	x = [a + h * i for i in range(n)] + [b]
	y = [c + k * j for j in range(m)] + [d]
	ti = time()
	ψ = fd(f, g, x, y, a, b, c, d, m, n)
	tf = time() - ti
	print(f'Relajación duró: {tf}s')
	ti = time()
	ω = mc(f, g, x, y, a, b, c, d, m, n, N)
	tf = time() - ti
	print(f'Monte Carlo duró: {tf}s')
	if φ:
		X, Y = np.meshgrid(x, y, indexing='ij')
		χ = φ(X,Y)
		#print(f'{χ=}')
		rmse = sqrt((np.square(χ - ψ)).mean())
		print(f'Relajación: {rmse=}')
		rmse = sqrt((np.square(χ - ω)).mean())
		print(f'Monte Carlo: {rmse=}')
		plot(x, y, χ.T)
	plot(x, y, np.array(ψ).T)
	plot(x, y, np.array(ω).T)

def plot(x, y, φ):
	shw = plt.pcolormesh(x, y, φ)
	bar = plt.colorbar(shw)
	plt.show()

main()
