''' Ecuación de Poisson
	∇²φ(x,y) = f(x,y)	∈ [a,b]×[c,d]
	∂φ(x,y) = g(x,y)
'''

def fd(f, g, x, y, a, b, c, d, m, n, T=1e-6, M=100000):
	'''
	Finite difference method. (Burden, Numerical analysis, 2010)

	Args:
		f(x,y) (func): -ρ(x,y) / ε_0
		g(x,y) (func): boundary conditions
		a, b, c, d (int): R = [a,b] x [c,d]
		m, n (int): lattice divisions
		T (float): tolerance
		M (int): maximum iterations

	Returns:
		φ (float matrix): Potential at each (x,y)
	'''
	φ = [[0] * (m+1) for j in range(n+1)]
	h = (b - a) / n
	k = (d - c) / m
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

		z = (-h*h * f(x[n-1],y[m-1]) + g(b,y[m-1]) + λ * g(x[n-1],d) + φ[n-2][m-1] + λ * φ[n-1][m-2]) / μ
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
			return φ
	raise Exception('fd failed to converge')
