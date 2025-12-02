''' Método de Monte Carlo en 3D
	con caminatas aleatorias
'''

from random import randint

def mc(f, g, x, y, z, a, b, c, d, u, v, m, n, l, N):
	'''
	3D Monte Carlo solver

	Args:
		f(x,y,z) (func): -ρ(x,y,z) / ε_0
		g(x,y,z) (func): boundary conditions
		x, y, z (list(float)): ticks
		a, b, c, d, u, v (float): R = [a,b] x [c,d] x [u,v]
		m, n, l (int): lattice divisions
		N (int): number of random walks

	Returns:
		φ (matrix3D(float)): Potential at each (x,y,z)
	'''
	h = (b - a) / n
	k = (d - c) / m
	w = (u - v) / l
	φ = [[[0] * (l+1) for j in range(m+1)] for i in range(n+1)]
	# condiciones de frontera
	for i in range(n+1):
		for j in range(m+1):
			φ[i][j][0] = g(x[i], y[j], z[l])
			φ[i][j][l] = g(x[i], y[j], z[0])
	for i in range(n+1):
		for j in range(l+1):
			φ[i][0][j] = g(x[i], y[0], z[j])
			φ[i][m][j] = g(x[i], y[m], z[j])
	for j in range(m+1):
		for i in range(l+1):
			φ[0][j][i] = g(x[0], y[j], z[j])
			φ[n][j][i] = g(x[n], y[j], z[j])
	# puntos internos
	for i in range(1, n):
		for j in range(1, m):
			for o in range(1, l):
				φ[i][j][o] = mc2d(f, g, x, y, z, (i, j, o), (a, b, c, d, u, v), h, k, w, N)
	return φ

def mc2d(f, g, x, y, r0, bnd, h, k, w, N):
	'''
	Random walks

	Args:
		f(x,y,z) (func): -ρ(x,y,z) / ε_0
		g(x,y,z) (func): boundary conditions
		x, y, z (list(float)): ticks
		r0 ((int, int, int)): (i,j,o) | x0 = x[i] & y0 = y[j] & z0 = z[o]
		bnd (tuple(float)): = (a, b, c, d, u, v) | R = [a,b] x [c,d] x [u,v]
		h, k, w (int): lattice spacings
		N (int): number of random walks

	Returns:
		φ(x0, y0, z0) (float): Potential at (x0,y0,z0)
	'''
	φ = 0
	F = 0
	a, b, c, d, u, v = bnd
	for i in range(N):
		i, j, o = r0
		while (0 < i < len(x)-1) and (0 < j < len(y)-1) and (0 < o < len(z)-1):
			F += f(x[i], y[j], z[o]) * h * k * w / N
			n = randint(1, 6)
			match n:
				case 1:		# x+
					i += 1
				case 2:		# x-
					i -= 1
				case 3:		# y+
					j += 1
				case 4:		# y-
					j -= 1
				case 5:		# z+
					o -= 1
				case 6:		# z-
					o -= 1
		φ += g(x[i], y[j], z[o]) / N
	φ -= F
	return φ
