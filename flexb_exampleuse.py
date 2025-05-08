import flexb
import random

nn = flexb.network([
    [
        flexb.neuron(flexb.funcs.swish, flexb.funcs.swishDeriv, [1,-1], 0),
        flexb.neuron(flexb.funcs.swish, flexb.funcs.swishDeriv, [1,-1], 0)
    ],
    [
        flexb.neuron(flexb.funcs.swish, flexb.funcs.swishDeriv, [1,-1], 0)
    ]
])

xyz = 0
while xyz <= 100000:
    a = random.randint(0,10)
    b = random.randint(0,10)
    x = a-b
    if x >= 0:
        nn.forward([a,b])
        nn.backward([x],0.006)
        xyz += 1

print(nn.forward([6,2]))