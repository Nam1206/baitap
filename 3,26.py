def grad(x):
    return 2*x - 4
def cost(x):
    return x**2 - 4*x + 5

x = 5
eta = 0.2

for it in range(1,5):
    x = x - eta * grad(x)
    print("Bước %d: x = %.4f, cost = %.4f" % (it, x, cost(x)))
