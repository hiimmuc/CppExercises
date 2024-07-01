import time
import math

# # Python Program to find prime numbers in a range
# import time


# def SieveOfEratosthenes(n):

#     # Create a boolean array "prime[0..n]" and
#     # initialize all entries it as true. A value
#     # in prime[i] will finally be false if i is
#     # Not a prime, else true.
#     prime = [True for i in range(n+1)]

#     p = 2
#     while (p * p <= n):

#         # If prime[p] is not changed, then it is
#        # a prime
#         if (prime[p] == True):

#             # Update all multiples of p
#             for i in range(p * p, n + 1, p):
#                 prime[i] = False
#         p += 1
#     c = 0

#     # Print all prime numbers
#     for p in range(2, n):
#         if prime[p]:
#             c += 1
#     return c


# # Driver function
# t0 = time.time()
# c = SieveOfEratosthenes(100000)
# print("Total prime numbers in range:", c)

# t1 = time.time()
# print("Time required:", t1 - t0)


def isPrime(number):
    if number <= 1:
        return False

    if number % 2 == 0:
        return number == 2

    max_div = math.floor(math.sqrt(number))
    for i in range(3, 1 + max_div, 2):
        if number % i == 0:
            return False
    return True


def findDualPrimes(L: int, R: int):
    """Find  number of dual prime numbers in list which satisfied those conditions:
    - Is a prime
    - Sum of digits also a prime

    Args:
        L (int): Left boundary  
        R (int): Right boundary

    Returns:
        int: number of dual primes found
    """
    prime_1digit = {2, 3, 5, 7}
    prime_number_set = set()
    dual_prime_number_list = list()

    time_start = time.time()
    # check condition of L and R
    if not (L <= R) and not (L > 1 and R <= 1e6):
        return -1

    for number in range(L, R + 1):
        # check prime
        if isPrime(number):
            prime_number_set.add(number)

            # check digits in number
            # for digit in str(number):
            #     if int(digit) not in prime_1digit:
            #         continue
            #     else:
            #         sum_digits += digit

            sum_digits = sum([int(x) for x in str(number)])

            # if sum of digits in one of two list, that number satisfied conditions
            if (sum_digits in prime_1digit) or (sum_digits in prime_number_set):
                dual_prime_number_list.append(number)
            else:
                continue

        else:
            continue

    print(f"Running time for n = {len(range(L, R))}", time.time() - time_start)

    return dual_prime_number_list


if __name__ == "__main__":
    with open('./INP/PRIME.INP') as f:
        data = f.read()
    L, R = map(int, data.split(" "))

    results = sorted(findDualPrimes(L, R))
    # print(results)

    with open("./OUT/PRIME.OUT", 'w') as wf:
        wf.write(str(len(results)))
