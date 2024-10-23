import numpy

def f(u):
    if u >= 0:
        return 1
    else:
        return -1

def findOutput(data, w):
    u = 0.0
    for i in range(0, len(data)):
        u += w[i]*data[i]
    return f(u)

# initialization
p = [[1,1,-1],[1,-1,-1],[-1,1,-1],[-1,-1,-1]] # conjunto de valores de entrada ampliados com a entrada dummy
d = [1, 1, 1, -1] # saidas desejadas
w = numpy.random.rand(len(p[0])) # inicializacao randomica dos pesos

c = 0.5          #taxa de aprendizado
d_error = 0.01   #erro desejado

iter = 0
while True:
    error = 0
    for i in range(0, len(p)):
        o = findOutput(p[i], w)
        error += ((d[i]-o)**2)/2
        learningSignal = c*(d[i]-o)
        for k in range(0, len(p[i])):
            w[k] += learningSignal*p[i][k]

    iter += 1
    print(error, " ## ", w)
    if error < d_error:
        print('N. iterations:', iter)
        break

print(findOutput([1,1,-1],w))
print(findOutput([1,-1,-1],w))
print(findOutput([-1,1,-1],w))
print(findOutput([-1,-1,-1],w))
# print result