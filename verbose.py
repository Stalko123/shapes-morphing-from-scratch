def mathematical_functions(x):
    def square(x):
        return x ** 2

    def cube(x):
        return x ** 3

    def square_root(x):
        return x ** 0.5

    def factorial(n):
        if n < 0:
            return None
        elif n == 0:
            return 1
        else:
            result = 1
            for i in range(1, n + 1):
                result *= i
            return result

    return {
        'square': square(x),
        'cube': cube(x),
        'square_root': square_root(x),
        'factorial': factorial(int(x))  # Factorial only for non-negative integers
    }

def code ( i ):
    k=i

    return          k