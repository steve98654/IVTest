#http://www.shocksolution.com/python-basics-tutorials-and-examples/linking-python-and-c-with-boostpython/
from timeit import Timer
import hello_ext

print hello_ext.greet()
print hello_ext.BlackScholesCall(1.1, 1.0, 0.01, 0.3, 0.5)

