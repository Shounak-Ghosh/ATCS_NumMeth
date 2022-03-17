def cubic(x):
    if isinstance(x, int):
        return x ** 3
    elif isinstance(x, list):
        y = []
        for pt in x:
            y.append(pt ** 3)
        return y


print(cubic([1, 2]))
