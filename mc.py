''' Método de Monte Carlo en 2D
	con caminatas aleatorias
'''

from random import randint

def mc(f, g, x, y, a, b, c, d, m, n, N):
	h = (b - a) / n
	k = (d - c) / m
	φ = [[0] * (m+1) for j in range(n+1)]
	# condiciones de frontera
	for i in range(n+1):
		φ[i][0] = g(x[i], y[0])
		φ[i][m] = g(x[i], y[m])
	for j in range(m+1):
		φ[0][j] = g(x[0], y[j])
		φ[n][j] = g(x[n], y[j])
	# puntos internos
	for i in range(1, n):
		for j in range(1, m):
			φ[i][j] = mc2d(f, g, x, y, (i, j), (a, b, c, d), h, k, N)
	return φ

def mc2d(f, g, x, y, r0, bnd, h, k, N):
	# returns φ(r0)
	φ = 0
	F = 0
	ε = 1e-3
	a, b, c, d = bnd
	for i in range(N):
		i, j = r0
		#w = []
		while (0 < i < len(x)-1) and (0 < j < len(y)-1):
			#w.append((x,y))
			F += f(x[i], y[j]) * h * k / N
			n = randint(1, 4)
			match n:
				case 1:		# este
					i += 1
				case 2:		# oeste
					i -= 1
				case 3:		# norte
					j += 1
				case 4:		# sur
					j -= 1
		φ += g(x[i], y[j]) / N
		#print(f'{w=}\n{F=}\n{φ=}')
	φ -= F
	return φ
