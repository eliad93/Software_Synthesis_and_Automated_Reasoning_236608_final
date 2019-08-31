import grammars as syn
import pandas as pd
from multiprocessing import Process, Queue
import timeit

TIME = 10

# GRAMMAR 1: ARITHMETIC OPERATORS
g1 = syn.Grammar(syn.GrammarType.NUM,
                 {"E", "OP", "NUM"},
                 {"E": {"( E ) OP ( E )", "NUM", "x_input"},
                  "OP": {"+", "-", "*", "/"},
                  "NUM": {"1", "2", "6"}},
                 "E")

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


# GRAMMAR 2: STRING MANIPULATION
g2 = syn.Grammar(syn.GrammarType.STR,
                 {"E", "I"},
                 {"E": {"Concat( E , E )", "Substr( E , I , I )", "x_input"},
                  "I": {"0", "1", "2", "3", "I - I" "Len( E )",
                        "IndexOf( E , E )"}},
                 "E")

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


g3 = syn.Grammar(syn.GrammarType.LIST,
                 {"E", "NUM"},
                 {"E": {"x_input", "Sort( E )", "Reverse( E )",
                        "Slice( E , NUM , NUM )"},
                  "NUM": {"0", "1", "2", "NUM + NUM"}},
                 "E")

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


def run_function(grammar: syn.Grammar, benchmark: list, queue):
    s = syn.Synthesizer(grammar)
    start = timeit.timeit()
    prog = s.bottom_up(benchmark)
    end = timeit.timeit()
    output = [end - start, prog]
    queue.put(output)


def test_function(grammar: syn.Grammar, benchmark: list):
    queue = Queue()  # using to get the result todo: fix comment below
    # creation of a process calling longfunction with the specified arguments
    proc = Process(target=run_function, args=(grammar, benchmark, queue))
    proc.start()  # launching the process on another thread
    try:
        # getting the result under timeout second or stop
        my_res = queue.get(timeout=TIME)
        # proper delete if the computation has take less than timeout seconds
        proc.join()
        return my_res
    except:  # catching every exception type
        proc.terminate()  # kill the process
        return -1


if __name__ == '__main__':
    # todo: eliminate code duplicates - (hint: define a function)
    # todo: run the tests with all classes, adjust output for classes
    data = []
    # NUMERIC TEST
    num_bench = [g1_b1, g1_b2, g1_b3, g1_b4, g1_b5]
    i = 1
    for test in num_bench:
        res = test_function(g1, test)
        if res == -1:
            # print(["numeric",i,"Not Found",time])
            # todo: use string from GrammarType instead of "numeric" for example
            data.append(["numeric", i, "Not Found", res])
        else:
            # print(["numeric",i,"Found",time])
            data.append(["numeric", i, "Found", abs(res[0])])
            f = open('g1_b' + str(i) + '.txt', 'w+')
            f.write(res[1])
            f.close()
        i += 1

    # STRING TEST
    string_bench = [g2_b1, g2_b2, g2_b3, g2_b4, g2_b5]
    i = 1
    for test in string_bench:
        res = test_function(g2, test)
        if res == -1:
            # print(["string",i,"Not Found",time])
            data.append(["string", i, "Not Found", res])
        else:
            # print(["string",i,"Found",time])
            data.append(["string", i, "Found", abs(res[0])])
            f = open('g2_b' + str(i) + '.txt', 'w+')
            f.write(res[1])
            f.close()
        i += 1

    # LIST TEST
    list_bench = [g3_b1, g3_b2, g3_b3, g3_b4, g3_b5]
    i = 1
    for test in list_bench:
        res = test_function(g3, test)
        if res == -1:
            # print(["list",i,"Not Found",time])
            data.append(["list", i, "Not Found", res])
        else:
            # print(["list",i,"Found",time])
            data.append(["list", i, "Found", abs(res[0])])
            f = open('g3_b' + str(i) + '.txt', 'w+')
            f.write(res[1])
            f.close()
        i += 1

    df = pd.DataFrame(data, columns=["Grammar Type", "Benchmark", "Outcome",
                                     "Running Time"])
    print(df)




