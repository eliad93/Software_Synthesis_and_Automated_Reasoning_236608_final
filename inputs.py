from collections import Counter


def c1(p):
    return len(p) > 3


def c2(p):
    return not str.isdigit(p[0]) and not str.isdigit(p[-1])


def c3(p):
    if "x_input" not in p:
        return True
    for i in range(1, 30):
        x_input = i
        if eval(p) < i:
            return False
    return True


def c4(p):
    if "x_input" not in p and eval(p) > 6:
        return False
    return True


def c5(p):
    if p.count('+') > 3 or p.count('*') > 3 or p.count('/') > 1 or \
            p.count("x_input") > 6 or p.count('1') > 2 or p.count('2') > 1 \
            or p.count('6') > 1:
        return False
    return True


def c6(p):
    letter_counts = Counter(p)
    if letter_counts['6'] > 1 or letter_counts['-'] > 1 or letter_counts['/'] \
            > 1 or letter_counts['*'] > 3 or letter_counts['1'] > 2 or \
            letter_counts['+'] > 2 or letter_counts["x_input"] > 3:
        return False
    return True


def c7(p: str):
    letter_counts = Counter(p)
    if letter_counts['6'] > 0 or letter_counts['-'] > 0 or \
            letter_counts['/'] > 1 or letter_counts['*'] > 4 or \
            letter_counts['1'] > 2 or letter_counts['+'] > 2 \
            or p.count("x_input") > 4:
        return False
    return True


# BENCHMARK 1.1: should return the input: x_input
g1_b1 = [(1, 1), (2, 2)]

# BENCHMARK 1.2: x_input*2
g1_b2 = [(1, 2), (2, 4)]

# BENCHMARK 1.3 :  area of a cube : n*n*n
g1_b3 = [(1, 1), (2, 8), (3, 27), (4, 64), (5, 125)]

# BENCHMARK 1.4 :  area of a rectangular triangle: n*n/2
g1_b4 = [(2, 2), (3, 4.5), (4, 8), (5, 12.5)]

# BENCHMARK 1.5: Sum from 1 to n, the combinatorial formula is: n(n + 1)/2
g1_b5 = [(1, 1), (2, 3), (3, 6), (4, 10), (5, 15)]

# BENCHMARK 1.6: Sum of the square numbers from 1 to n,
# the combinatorial formula is: n(n+1)(2n+1)/6
g1_b6 = [(1, 1), (2, 5), (3, 14), (4, 30), (5, 55)]

# BENCHMARK 1.7: Sum of the power of 3 for numbers from 1 to n,
# the formula is : n*n*(n+1)*(n+1)/4
g1_b7 = [(1, 1), (2, 9), (3, 36), (4, 100), (5, 225)]

g1_b1_n = []
g1_b2_n = []
g1_b3_n = []
g1_b4_n = []
g1_b5_n = [(1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
           (1, 8), (1, 9), (3, 3), (4, 3), (5, 10)]
g1_b6_n = [(1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13)]
g1_b7_n = [(1, 5), (1, 7), (1, 10), (1, 11), (1, 13), (2, 7), (2, 5), (4, 11), (4, 30), (5, 49)]

g1_c1 = [c1, c2]
g1_c2 = [c1, c2]
g1_c3 = [c1, c2]
g1_c4 = [c1, c2]
g1_c5 = []
g1_c6 = [c6]
g1_c7 = [c7]

# BENCHMARK 2.1: should return the input: x_input
g2_b1 = [("abcd", "abcd"), ("a", "a")]

# BENCHMARK 2.2: should duplicate the input: Concat( x_input, x_input)
g2_b2 = [("ab", "abab"), ("123", "123123")]

# BENCHMARK 2.3: should return the 2 first characters of the string:
# Substr( x_input, 0 , 2 )
g2_b3 = [("abcd", "ab"), ("lkjh", "lk")]

# BENCHMARK 2.4: should return the third character of the string:
# Substr( x_input, 2,3)
g2_b4 = [("1234", "3"), ("flkmfvlr;rf", "k"),
         ("hello world", "l"), ("a good day", "g")]

# BENCHMARK 2.5 : should reverse the string :
# Substr(Concat(x_input,x_input),1,3) or
# Concat(Substr(x_input,1,2),Substr(x_input,0,1))
g2_b5 = [("ab", "ba"), ("lk", "kl"), ("12", "21")]

# BENCHMARK 2.6 : should return the last character of the string:
# Substr(x_input, Len(x_input)-1, Len(x_input))
g2_b6 = [("a", "a"), ("abc", "c"), ("12345", "5")]

g2_b1_n = []
g2_b2_n = []
g2_b3_n = []
g2_b4_n = []
g2_b5_n = [(1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
           (1, 8), (1, 9), (3, 3), (4, 3), (5, 10)]
g2_b6_n = []
g2_b7_n = []

# BENCHMARK 3.1: should return the input: x_input
g3_b1 = [([1, 2, 3], [1, 2, 3]), (["abc", "def"], ["abc", "def"])]

# BENCHMARK 3.2: should sort the list: Sort( x_input )
g3_b2 = [([2, 8, 7, 4], [2, 4, 7, 8]),
         (['c', 'f', 'l', 'a'], ['a', 'c', 'f', 'l'])]

# BENCHMARK 3.3: should reverse the list : Reverse( x_input )
g3_b3 = [([3, 2, 1], [1, 2, 3]), (['a', 'c', 'f', 'l'], ['l', 'f', 'c', 'a'])]

# BENCHMARK 3.4: should slice the list tp keep only
# the second entry of the list: Slice(x_input, 1 , 2 )
g3_b4 = [([1, 2, 3], [2]), (["abc", "def"], ["def"])]

# BENCHMARK 3.5: should sort the list descendant: Reverse( Sort( x_input )
g3_b5 = [([2, 8, 7, 4], [8, 7, 4, 2]),
         (['c', 'f', 'l', 'a'], ['l', 'f', 'c', 'a'])]

# BENCHMARK 3.4: should slice the list tp keep only
# the second entry of the list: Slice( Sort(x_input), 1 , 3 )
g3_b6 = [([1, 2, 3, 0], [1, 2]), (["z", "abc", "def", "w"], ["def", "w"])]

# BENCHMARK 3.6: should sort the list descendant and return only
# the two first entries : Slice( Reverse( Sort( x_input ) , 0 , 2 )
g3_b7 = [([2, 8, 7, 4], [8, 7]), (['c', 'f', 'l', 'a'], ['l', 'f'])]

g3_b1_n = [([1, 2], [1]), ([1, 2], [2])]
g3_b2_n = [([2, 8, 7, 4], [8, 7]), ([1, 2], [1]), ([1, 2], [2])]
g3_b3_n = []
g3_b4_n = []
g3_b5_n = []
g3_b6_n = []
g3_b7_n = [([1, 3, 0, 5, 2], [1]), ([1, 3, 0, 5, 2], [3]),
           ([1, 3, 0, 5, 2], [0]), ([1, 3, 0, 5, 2], [5]),
           ([1, 3, 0, 5, 2], [2]), ([1, 3, 0, 5, 2], [5, 2]),
           ([1, 3, 0, 5, 2], [0, 5, 2]), ([1, 3, 0, 5, 2], [3, 0, 5, 2]),
           ([1, 3, 0, 5, 2], [3, 0, 5]), ([1, 3, 0, 5, 2], [0, 5])]
