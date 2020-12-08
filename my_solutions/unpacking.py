def func1(x,y):
    return x+y

def func2(*inp):
    print(type(inp))
    return sum(inp)

a = [1,2]
x,y = a

print(func1(*a))
print(func2(*a))
print(func2(x,y))