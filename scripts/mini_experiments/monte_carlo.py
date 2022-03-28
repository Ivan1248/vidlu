import numpy as np

ys = []
for i in range(1000000):
    x, y = (np.random.rand() for i in range(2))
    ys.append((1 - x ** 2) ** 0.5)  # alternative: ys.append(int(x**2+y**2<=1))
    if i == 0 or str(i + 1)[0] == '1' and set(str(i + 1)[1:]) == {'0'}:
        print(f"{i + 1}: {4 * np.mean(ys)}")
