from collections import Counter


def c1(p):
    if p.find("x_input") == -1:
        return False
    return True


def c2(p):
    if p.find("x_input") == -1:
        return False
    return True


def c3(p):
    if p.find("x_input") == -1:
        return False
    return True


def c4(p: str) -> bool:
    if p.find('1') != -1 or p.find('+') != -1 or p.find('-') != -1 or \
            p.count('2') > 1:
        return False
    return True


def c5(p):
    letter_counts = Counter(p)
    if letter_counts['-'] > 0 or letter_counts['*'] > 1 or letter_counts['+'] \
            > 1 or letter_counts['/'] > 1 or letter_counts['1'] > 1 \
            or letter_counts['2'] > 1:
        return False
    return True


def c6(p):
    letter_counts = Counter(p)
    if letter_counts['6'] > 1 or letter_counts['-'] > 1 or letter_counts['/'] \
            > 1 or letter_counts['*'] > 3 or letter_counts['1'] > 2 or \
            letter_counts['+'] > 2 or letter_counts["x_input"] > 3:
        return False
    return True


def begin_with_sub(p: str) -> bool:
    if p.find("std_substring") == 0:
        return True
    else:
        return False


def list_c4(p: str) -> bool:
    return p.find("x_input") != -1


def list_c5(p: str) -> bool:
    return p[-1] == ")"


def list_c6(p: str) -> bool:
    return p.find("x_input") != -1


def list_c7(p: str) -> bool:
    return p.find("std_reverse") == -1


def list_c8(p: str) -> bool:
    return p.find("1") == -1


def str_c4(p: str) -> bool:
    return p.find("1") == -1


def str_c6(p: str) -> bool:
    return p.find("std_concat") == -1


def str_c7(p: str) -> bool:
    return p.find("len") == -1 and p.find("1") == -1


def str_c8(p: str) -> bool:
    letters_count = Counter(p)
    return letters_count['1'] <= 1 and p.count("x_input") <= 2 and \
        p.count("sub") <= 2 and p.count("con") <= 1


def str_c9(p: str) -> bool:
    letters_count = Counter(p)
    return letters_count['1'] <= 0 and letters_count['3'] <= 1 and \
        p.count("x_input") <= 3 and p.count("sub") <= 2 and p.count(
        "con") <= 1


def str_plus_c4(p: str) -> bool:
    letters_count = Counter(p)
    cond = True
    if len(p) > 25:
        cond = p.find("std_concat") == 0 or p.find("slice_str") == 0
    return cond and letters_count['1'] <= 0 and letters_count['2'] <= 2 and \
        letters_count['3'] <= 0 and letters_count['4'] <= 0 and \
        letters_count['5'] <= 2 and letters_count['-'] <= 1 and \
        letters_count['0'] <= 2 and letters_count['x'] <= 2 and\
        p.find("reverse") == -1 and p.count("concat") <= 1 and \
        p.find("len") == -1 and p.count("slice") <= 2


def str_plus_c5(p: str) -> bool:
    letters_count = Counter(p)
    cond = True
    if len(p) > 25:
        cond = p.find("std_concat") == 0 or p.find("slice_str") == 0
    return cond and letters_count['1'] <= 1 and letters_count['2'] <= 0 and \
        letters_count['3'] <= 0 and letters_count['4'] <= 1 and \
        letters_count['5'] <= 1 and letters_count['-'] <= 1 and \
        letters_count['0'] <= 2 and letters_count['x'] <= 3 and\
        p.find("reverse") == -1 and p.count("concat") <= 1 and \
        p.count("len") <= 1 and p.count("slice") <= 2


# noinspection PyUnusedLocal
def c_false(p):
    return False


# BENCHMARK 1.1: should return the input: x_input
g1_b1 = [(1, 1), (2, 2)]

# BENCHMARK 1.2: x_input * 2 or x_input + x_input
g1_b2 = [(1, 2), (2, 4)]

# BENCHMARK 1.3 :  area of a cube : x_input * x_input * x_input
g1_b3 = [(1, 1), (2, 8), (3, 27), (4, 64), (5, 125)]

# BENCHMARK 1.4 :  area of a rectangular triangle: n*n/2
g1_b4 = [(2, 2), (3, 4.5), (4, 8), (5, 12.5)]

# BENCHMARK 1.5: Sum from 1 to n, the combinatorial formula is:
# x_input * (x_input + 1) / 2
g1_b5 = [(1, 1), (2, 3), (3, 6), (4, 10), (5, 15)]

# BENCHMARK 1.6: Sum of the square numbers from 1 to n,
# the combinatorial formula is: n(n+1)(2n+1)/6
g1_b6 = [(1, 1), (2, 5), (3, 14), (4, 30), (5, 55)]

# Non realizable! input examples not consistent
g1_b7 = [(1, 1), (1, 2)]

# Non realizable! input examples not consistent with negative examples
g1_b8 = [(1, 1)]

# Non realizable! reaching to a dead end in new programs
g1_b9 = [(1, 1), (2, 4)]

g1_b1_n = []
g1_b2_n = []
g1_b3_n = []
g1_b4_n = []
g1_b5_n = [(1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
           (1, 8), (1, 9), (3, 3), (4, 3), (5, 10)]

g1_b6_n = [(1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13)]
g1_b7_n = []
g1_b8_n = [(1, 1)]
g1_b9_n = [(1, 2)]

g1_c1 = [c1]
g1_c2 = [c2]
g1_c3 = [c3]
g1_c4 = [c4]
g1_c5 = [c5]
g1_c6 = [c6]
g1_c7 = []
g1_c8 = []
g1_c9 = [c_false]

numeric_benchmarks = [
    [g1_b1, g1_b1_n, g1_c1],
    [g1_b2, g1_b2_n, g1_c2],
    [g1_b3, g1_b3_n, g1_c3],
    [g1_b4, g1_b4_n, g1_c4],
    [g1_b5, g1_b5_n, g1_c5],
    [g1_b6, g1_b6_n, g1_c6],
    [g1_b7, g1_b7_n, g1_c7],
    [g1_b8, g1_b8_n, g1_c8],
    [g1_b9, g1_b9_n, g1_c9]
]

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
# Substr(x_input, Len(x_input)-3, Len(x_input))
g2_b6 = [("qwerty", "rty"), ("abcde", "cde"), ("12345", "345")]

# std_concat(x_input, std_substr(x_input, 2, 6))
g2_b7 = [("aaabbb", "aaabbbaaa"), ("hhgsst", "hhgssthhg")]

# std_concat(std_substr(x_input,0,2),std_substr(x_input,3,0-1))
g2_b8 = [("a1b1c1", "a11c"), ("xyxyts", "xyyt")]

# std_concat(std_substr(x_input,0,2),std_substr(x_input,3,0-1))
g2_b9 = [("123456", "12456")]

# Non realizable! input examples not consistent
g2_b10 = [("ab", "a"), ("ab", "ab")]

# Non realizable! input examples not consistent with negative examples
g2_b11 = [("ab", "a")]

# Non realizable! reaching to a dead end in new programs
g2_b12 = [(1, 1), (2, 4)]

g2_b1_n = [("a", "")]
g2_b2_n = [("a", "")]
g2_b3_n = [("a", "")]
g2_b4_n = [("ab", "a")]
g2_b5_n = []
g2_b6_n = [("abcd", "abcd"), ("qwerty", "q")]
g2_b7_n = [("ab", "ab")]
g2_b8_n = [("abcd", "abcd"), ("abc", "a"), ("ab", "a")]
g2_b9_n = [("123", "123"), ("121", "21")]
g2_b10_n = []
g2_b11_n = [("ab", "ab")]
g2_b12_n = []

g2_c1 = []
g2_c2 = []
g2_c3 = []
g2_c4 = [str_c4]
g2_c5 = []
g2_c6 = [str_c6]
g2_c7 = [str_c7]
g2_c8 = [str_c8]
g2_c9 = [str_c9]
g2_c10 = []
g2_c11 = []
g2_c12 = [c_false]

string_benchmarks = [
    [g2_b1, g2_b1_n, g2_c1],
    [g2_b2, g2_b2_n, g2_c2],
    [g2_b3, g2_b3_n, g2_c3],
    [g2_b4, g2_b4_n, g2_c4],
    [g2_b5, g2_b5_n, g2_c5],
    [g2_b6, g2_b6_n, g2_c6],
    [g2_b7, g2_b7_n, g2_c7],
    [g2_b8, g2_b8_n, g2_c8],
    [g2_b9, g2_b9_n, g2_c9],
    [g2_b10, g2_b10_n, g2_c10],
    [g2_b11, g2_b11_n, g2_c11],
    [g2_b12, g2_b12_n, g2_c12]
]

g3_b1 = [([1], [])]

# BENCHMARK 3.1: should return the input: x_input
g3_b2 = [([1, 2, 3], [1, 2, 3]), (["abc", "def"], ["abc", "def"])]

# BENCHMARK 3.2: should sort the list: Sort( x_input )
g3_b3 = [([2, 8, 7, 4], [2, 4, 7, 8]),
         (['c', 'f', 'l', 'a'], ['a', 'c', 'f', 'l'])]

# BENCHMARK 3.3: should reverse the list : Reverse( x_input )
g3_b4 = [([3, 2, 1], [1, 2, 3]), (['a', 'c', 'f', 'l'], ['l', 'f', 'c', 'a'])]

# BENCHMARK 3.4: should slice the list tp keep only
# the second entry of the list: Slice(x_input, 1 , 2 )
g3_b5 = [([1, 2, 3], [2]), (["abc", "def"], ["def"])]

# BENCHMARK 3.5: should sort the list descendant: Reverse( Sort( x_input )
g3_b6 = [([2, 8, 7, 4], [8, 7, 4, 2]),
         (['c', 'f', 'l', 'a'], ['l', 'f', 'c', 'a'])]

# BENCHMARK 3.6: should slice the list tp keep only
# the second entry of the list: Slice( Sort(x_input), 1 , 3 )
g3_b7 = [([1, 2, 3, 0], [1, 2]), (["z", "abc", "def", "w"], ["def", "w"])]

# BENCHMARK 3.7: should sort the list descendant and return only
# the two first entries : Slice( Reverse( Sort( x_input ) , 0 , 2 )
g3_b8 = [([2, 8, 7, 4], [8, 7]), (['c', 'f', 'l', 'a'], ['l', 'f'])]

# Non realizable! input examples not consistent
g3_b9 = [([1, 2], [1, 2]), ([1, 2], [1])]

# Non realizable! input examples not consistent with negative examples
g3_b10 = [([1, 2], [2])]

# Non realizable! reaching to a dead end in new programs
g3_b11 = [([1, 2], [1])]

g3_b1_n = []
g3_b2_n = [([1, 2], [1]), ([1, 2], [2])]
g3_b3_n = [([2, 8, 7, 4], [8, 7]), ([1, 2], [1]), ([1, 2], [2])]
g3_b4_n = [([1, 2], [1, 2])]
g3_b5_n = [([1, 2], [1, 2])]
g3_b6_n = [([1, 2], [1])]
g3_b7_n = [([1, 3, 2, 0], [1, 3])]
g3_b8_n = [([1, 3, 0, 5, 2], [1]), ([1, 3, 0, 5, 2], [3]),
           ([1, 3, 0, 5, 2], [0]), ([1, 3, 0, 5, 2], [5]),
           ([1, 3, 0, 5, 2], [2]), ([1, 3, 0, 5, 2], [5, 2]),
           ([1, 3, 0, 5, 2], [0, 5, 2]), ([1, 3, 0, 5, 2], [3, 0, 5, 2]),
           ([1, 3, 0, 5, 2], [3, 0, 5]), ([1, 3, 0, 5, 2], [0, 5])]
g3_b9_n = []
g3_b10_n = [([1, 2], [1])]
g3_b11_n = []

g3_c1 = []
g3_c2 = []
g3_c3 = []
g3_c4 = [list_c4]
g3_c5 = [list_c5]
g3_c6 = [list_c6]
g3_c7 = [list_c7]
g3_c8 = [list_c8]
g3_c9 = []
g3_c10 = []
g3_c11 = [c_false]

list_benchmarks = [
    [g3_b1, g3_b1_n, g3_c1],
    [g3_b2, g3_b2_n, g3_c2],
    [g3_b3, g3_b3_n, g3_c3],
    [g3_b4, g3_b4_n, g3_c4],
    [g3_b5, g3_b5_n, g3_c5],
    [g3_b6, g3_b6_n, g3_c6],
    [g3_b7, g3_b7_n, g3_c7],
    [g3_b8, g3_b8_n, g3_c8],
    [g3_b9, g3_b9_n, g3_c9],
    [g3_b10, g3_b10_n, g3_c10],
    [g3_b11, g3_b11_n, g3_c11]
]

# slice_str(x_input,5,0,-1)
g2_plus_b1 = [("abcde", "edcb")]
# std_concat(reverse_str(x_input),reverse_str(x_input))
g2_plus_b2 = [("abc", "cbacba")]
# std_concat(x_input,reverse_str(x_input))
g2_plus_b3 = [("abc1abc2", "abc1abc22cba1cba")]
# std_concat(slice_str(x_input,0,5,2),std_substr(x_input,0,-5))
g2_plus_b4 = [("123456", "1351")]
# std_concat(slice_str(x_input,0,-1,4),slice_str(x_input,0,len(x_input),5))
g2_plus_b5 = [("123456", "1516")]

g2_plus_b1_n = []
g2_plus_b2_n = []
g2_plus_b3_n = []
g2_plus_b4_n = []
g2_plus_b5_n = []

g2_plus_c1 = []
g2_plus_c2 = []
g2_plus_c3 = []
g2_plus_c4 = [str_plus_c4]
g2_plus_c5 = [str_plus_c5]

string_plus_benchmarks = [
    [g2_plus_b1, g2_plus_b1_n, g2_plus_c1],
    [g2_plus_b2, g2_plus_b2_n, g2_plus_c2],
    [g2_plus_b3, g2_plus_b3_n, g2_plus_c3],
    [g2_plus_b4, g2_plus_b4_n, g2_plus_c4],
    [g2_plus_b5, g2_plus_b5_n, g2_plus_c5],
]
