import copy
import time
from enum import Enum
from itertools import product
from multiprocessing.pool import Pool

from inputs import *

# noinspection PyGlobalUndefined
global pool


def set_global_pool():
    global pool
    pool = Pool(7)  # type: Pool


class GrammarType(Enum):
    NUM = "numeric"
    STR = "string"
    LIST = "list"


class Grammar:
    def __init__(self, grammar_type: GrammarType, rules_groups, starting):
        self.grammar_type = grammar_type
        self.rules_groups = rules_groups
        self.starting = starting
        self.trivial_programs = self.create_trivial_programs()

    def create_trivial_programs(self):
        if self.grammar_type == GrammarType.NUM:
            return self.numeric_trivial_programs()
        if self.grammar_type == GrammarType.STR:
            return self.str_trivial_programs()
        if self.grammar_type == GrammarType.LIST:
            return self.list_trivial_programs()

    @staticmethod
    def numeric_trivial_programs():
        return ["((x_input)+(x_input))", "((x_input)-(x_input))",
                "(x_input)/(x_input)", "(1)*", "*(1)", "(0)+", "+(0)", "*(0)",
                "(0)*", "(0)-", "-(0)", "/(0)", "(0)/"]

    @staticmethod
    def str_trivial_programs():
        return ["slice_str(slice_str(", "reverse_str(reverse_str(",
                "std_substr(std_substr("]

    @staticmethod
    def list_trivial_programs():
        return ["std_sort(std_sort(", "std_reverse(std_reverse(",
                "std_slice(std_slice(", "0+", "+0", "1+1"]


# noinspection PyPep8Naming,PyBroadException
class Synthesizer:

    def __init__(self, g: Grammar):
        self.g = g
        self.P = {}
        self.non_checked_programs = []
        self.E = None
        self.sizes = {}

    def bottom_up(self, E):
        self.init_program(E)
        if self.is_unrealizable():
            return None
        # print(len(self.non_checked_programs))
        while True:
            self.save_sizes()
            self.grow()
            if self.is_stuck():
                return None
            # print(len(self.non_checked_programs))
            for p in self.non_checked_programs:
                try:
                    if is_program_match(p, self.E):
                        return p
                except:
                    pass
            self.non_checked_programs = []

    def init_program(self, E):
        """
        deduce the base programs by the grammar
        :return:
        """
        self.E = E[0]
        self.init_starting_programs()

    def grow(self):
        new_P = copy.deepcopy(self.P)
        for r_g in self.g.rules_groups:
            for r in self.g.rules_groups[r_g]:
                if not ground(r):
                    list_products = map(lambda x: "".join(x), self.produce(r))
                    for p in list_products:
                        try:
                            if self.insert_program_condition(p, r_g):
                                new_P[r_g].append(p)
                                if r_g == self.g.starting:
                                    self.non_checked_programs.append(p)
                        except:
                            pass
        self.P = new_P

    def is_new_program(self, p, r_g):
        for p2 in self.P[r_g]:
            if is_observational_equivalent(p, p2, self.E):
                return False
        return True

    def produce(self, r: str):
        tokens = r.split()
        lists_for_product = []
        for t in tokens:
            if t.isupper():
                lists_for_product.append(self.P[t])
            else:
                lists_for_product.append([t])
        split_programs = product(*lists_for_product)
        return split_programs

    def init_starting_programs(self):
        self.P = {}
        self.non_checked_programs = []
        for r in self.g.rules_groups:
            for t in self.g.rules_groups[r]:
                if ground(t):
                    if r in self.P:
                        self.P[r].append(t)
                    else:
                        self.P[r] = [t]
                    if r == self.g.starting:
                        self.non_checked_programs.append(t)

    def insert_program_condition(self, p, r_g):
        return not any(s in p for s in self.g.trivial_programs) \
               and self.is_new_program(p, r_g)

    def is_unrealizable(self):
        input_dict = {}
        for i, o in self.E:
            try:  # make sure it's immutable!
                value = tuple(i)
            except TypeError:
                value = i
            if value in input_dict:
                if input_dict[value] != o:
                    return True
            else:
                input_dict[value] = o
        return False

    def save_sizes(self):
        self.sizes = {}
        for r in self.P:
            self.sizes[r] = len(self.P[r])

    def is_stuck(self):
        for r in self.P:
            if self.sizes[r] != len(self.P[r]):
                return False
        return True


class BadExamplesSynthesizer(Synthesizer):
    def __init__(self, g):
        super().__init__(g)
        self.E_not = None

    def init_program(self, E):
        self.E = E[0]
        self.E_not = E[1]
        self.init_starting_programs()

    # noinspection PyUnusedLocal
    def is_counter_example(self, t):
        for i, o in self.E_not:
            x_input = i
            if eval(t) == o:
                return True
        return False

    def insert_program_condition(self, p, r_g):
        return super().insert_program_condition(p, r_g) \
               and not self.is_counter_example(p)

    def is_unrealizable(self):
        input_dict = {}
        for i, o in self.E:
            try:  # make sure it's immutable!
                value = tuple(i)
            except TypeError:
                value = i
            if value in input_dict:
                if input_dict[value] != o:
                    return True
            else:
                input_dict[value] = o
        for i, o in self.E_not:
            try:  # make sure it's immutable!
                value = tuple(i)
            except TypeError:
                value = i
            if value in input_dict:
                if input_dict[value] == o:
                    return True
        return False


# noinspection PyPep8Naming
class ConstraintsSynthesizer(BadExamplesSynthesizer):
    def __init__(self, g):
        super().__init__(g)
        self.C = None

    def init_program(self, E):
        super().init_program(E)
        self.C = E[2]

    def insert_program_condition(self, p, r_g):
        return super().insert_program_condition(p, r_g) \
               and self.is_matching_constraints(p)

    def is_matching_constraints(self, p):
        return all([c(p) for c in self.C])


# noinspection PyPep8Naming,PyBroadException
class OptimizedSynthesizer(ConstraintsSynthesizer):
    def __init__(self, g):
        super().__init__(g)
        self.solution = None

    def bottom_up(self, E):
        self.init_program(E)
        if self.is_unrealizable():
            return None
        self.solution = None
        try:
            for p in self.P[self.g.starting]:
                if ground(p) and is_program_match(p, self.E):
                    return p
        except:
            pass

        while self.solution is None:
            # print(len(self.P[self.g.starting]))
            self.save_sizes()
            self.grow()
            if self.solution is None and self.is_stuck():
                return None
        return self.solution

    def grow(self):
        new_P = copy.deepcopy(self.P)
        for r_g in self.g.rules_groups:
            for r in self.g.rules_groups[r_g]:
                if not ground(r):
                    list_products = map(lambda x: "".join(x), self.produce(r))
                    for p in list_products:
                        try:
                            if self.insert_program_condition(p, r_g):
                                new_P[r_g].append(p)
                                if r_g == self.g.starting:
                                    if is_program_match(p, self.E):
                                        self.solution = p
                                        return
                        except:
                            pass
        self.P = new_P


# noinspection PyBroadException
class NoDuplicatesSynthesizer(OptimizedSynthesizer):
    def __init__(self, g):
        super().__init__(g)
        self.solution = None
        self.new_P = None

    def grow(self):
        self.new_P = {}
        for r in self.g.rules_groups:
            self.new_P[r] = []
        for r_g in self.g.rules_groups:
            for r in self.g.rules_groups[r_g]:
                if not ground(r):
                    list_products = map(lambda x: "".join(x), self.produce(r))
                    for p in list_products:
                        try:
                            if self.insert_program_condition(p, r_g):
                                self.new_P[r_g].append(p)
                                if r_g == self.g.starting:
                                    if is_program_match(p, self.E):
                                        self.solution = p
                                        return
                        except:
                            pass
        for p_g in self.new_P:
            for p in self.new_P[p_g]:
                self.P[p_g].append(p)
        self.new_P = None

    def is_new_program(self, p, r_g):
        for p2 in self.P[r_g]:
            if is_observational_equivalent(p, p2, self.E):
                return False
        for p2 in self.new_P[r_g]:
            if is_observational_equivalent(p, p2, self.E):
                return False
        return True


# noinspection PyBroadException
class ParallelSynthesizer(NoDuplicatesSynthesizer):
    def __init__(self, g: Grammar):
        super().__init__(g)
        self.depth = 0

    def bottom_up(self, E):
        self.depth = 0
        return super().bottom_up(E)

    def grow(self):
        self.depth += 1
        if self.depth < 3:
            super().grow()
        else:
            global pool
            x = pool  # type: Pool
            self.new_P = {}
            for r in self.g.rules_groups:
                self.new_P[r] = []
            for r_g in self.g.rules_groups:
                for r in self.g.rules_groups[r_g]:
                    if not ground(r):
                        list_products = map(lambda var: ("".join(var), r_g),
                                            self.produce(r))
                        for p, insert_res, match_res in x.imap_unordered(
                                self.insert_program_condition_parallel,
                                list_products, chunksize=32):
                            if match_res:
                                self.solution = p
                                return
                            if insert_res:
                                self.new_P[r_g].append(p)
            for p_g in self.new_P:
                for p in self.new_P[p_g]:
                    self.P[p_g].append(p)
            self.new_P = None

    def insert_program_condition_parallel(self, A):
        p = A[0]
        r_g = A[1]
        insert_res = False
        try:
            insert_res = super().insert_program_condition(p, r_g)
            match_res = is_program_match(p, self.E)
            return p, insert_res, match_res
        except:
            return None, insert_res, False


# noinspection PyUnusedLocal
def is_observational_equivalent(p1, p2, E):
    for i, o in E:
        x_input = i
        if eval(p1) != eval(p2):
            return False
    return True


def ground(t: str) -> bool:
    """
    check if a string represents program is terminal.
    a terminal program is a program that cannot be
    further produced
    :param t:
    :return:
    """
    tokens = t.split()
    for s in tokens:
        if s.isupper():
            return False
    return True


# noinspection PyUnusedLocal
def is_program_match(p, d):
    for i, o in d:
        x_input = i
        if eval(p) != o:
            return False
    return True


def std_concat(s1: str, s2: str) -> str:
    return s1 + s2


def std_substr(s: str, i: int, j: int) -> str:
    return s[i:j]


def std_sort(l: list) -> list:
    return sorted(l)


def std_reverse(m_list: list) -> list:
    m_list = m_list[:]
    m_list.reverse()
    return m_list


def std_slice(s: list, i: int, j: int) -> list:
    return s[i:j]


def reverse_str(s: str) -> str:
    t = s[::-1]
    return t


def slice_str(s: str, i: int, j: int, k: int) -> str:
    return s[i:j:k]


# GRAMMAR 1: ARITHMETIC OPERATORS
g1 = Grammar(GrammarType.NUM,
             {"E": ["( E ) OP ( E )", "NUM", "x_input"],
              "OP": ["+", "-", "*", "/"],
              "NUM": ["1", "2", "6"]},
             "E")

g2 = Grammar(GrammarType.STR,
             {"E": ["std_concat( E , E )", "std_substr( E , I , I )",
                    "x_input"],
              "I": ["0", "1", "2", "3", "I - I", "len( E )"]},
             "E")
g2_plus = Grammar(GrammarType.STR,
                  {"E": ["std_concat( E , E )",
                         "reverse_str( E )", "slice_str( E , I , I , I )",
                         "x_input"],
                   "I": ["-1", "-2", "-3", "-4", "-5", "0", "1", "2", "3", "4",
                         "5", "len( E )"]},
                  "E")

g3 = Grammar(GrammarType.LIST,
             {"E": ["x_input", "[]", "std_sort( E )", "std_reverse( E )",
                    "std_slice( E , NUM , NUM )"],
              "NUM": ["0", "1", "2", "NUM + NUM"]},
             "E")


def main():
    s = time.time()
    syn = Synthesizer(g1)
    print(syn.bottom_up([g1_b1]))
    print(syn.bottom_up([g1_b2]))
    print(syn.bottom_up([g1_b3]))
    print(syn.bottom_up([g1_b4]))
    print(syn.bottom_up([g1_b5]))
    print(syn.bottom_up([g1_b7]))
    print(syn.bottom_up([g1_b8]))
    print(syn.bottom_up([g1_b9]))
    e = time.time()
    print("{} {}".format(type(syn), e - s))

    s = time.time()
    syn = BadExamplesSynthesizer(g1)
    print(syn.bottom_up([g1_b1, g1_b1_n]))
    print(syn.bottom_up([g1_b2, g1_b2_n]))
    print(syn.bottom_up([g1_b3, g1_b3_n]))
    print(syn.bottom_up([g1_b4, g1_b4_n]))
    print(syn.bottom_up([g1_b5, g1_b5_n]))
    print(syn.bottom_up([g1_b7, g1_b7_n]))
    print(syn.bottom_up([g1_b8, g1_b8_n]))
    print(syn.bottom_up([g1_b9, g1_b9_n]))
    e = time.time()
    print("{} {}".format(type(syn), e - s))

    s = time.time()
    syn = ConstraintsSynthesizer(g1)
    print(syn.bottom_up([g1_b1, g1_b1_n, g1_c1]))
    print(syn.bottom_up([g1_b2, g1_b2_n, g1_c2]))
    print(syn.bottom_up([g1_b3, g1_b3_n, g1_c3]))
    print(syn.bottom_up([g1_b4, g1_b4_n, g1_c4]))
    print(syn.bottom_up([g1_b5, g1_b5_n, g1_c5]))
    print(syn.bottom_up([g1_b7, g1_b7_n, g1_c7]))
    print(syn.bottom_up([g1_b8, g1_b8_n, g1_c8]))
    print(syn.bottom_up([g1_b9, g1_b9_n, g1_c9]))
    e = time.time()
    print("{} {}".format(type(syn), e - s))

    s = time.time()
    syn = OptimizedSynthesizer(g1)
    print(syn.bottom_up([g1_b1, g1_b1_n, g1_c1]))
    print(syn.bottom_up([g1_b2, g1_b2_n, g1_c2]))
    print(syn.bottom_up([g1_b3, g1_b3_n, g1_c3]))
    print(syn.bottom_up([g1_b4, g1_b4_n, g1_c4]))
    print(syn.bottom_up([g1_b5, g1_b5_n, g1_c5]))
    print(syn.bottom_up([g1_b7, g1_b7_n, g1_c7]))
    print(syn.bottom_up([g1_b8, g1_b8_n, g1_c8]))
    print(syn.bottom_up([g1_b9, g1_b9_n, g1_c9]))
    e = time.time()
    print("{} {}".format(type(syn), e - s))

    s = time.time()
    syn = NoDuplicatesSynthesizer(g1)
    print(syn.bottom_up([g1_b1, g1_b1_n, g1_c1]))
    print(syn.bottom_up([g1_b2, g1_b2_n, g1_c2]))
    print(syn.bottom_up([g1_b3, g1_b3_n, g1_c3]))
    print(syn.bottom_up([g1_b4, g1_b4_n, g1_c4]))
    print(syn.bottom_up([g1_b5, g1_b5_n, g1_c5]))
    print(syn.bottom_up([g1_b7, g1_b7_n, g1_c7]))
    print(syn.bottom_up([g1_b8, g1_b8_n, g1_c8]))
    print(syn.bottom_up([g1_b9, g1_b9_n, g1_c9]))
    e = time.time()
    print("{} {}".format(type(syn), e - s))

    s = time.time()
    syn = Synthesizer(g2)
    print(syn.bottom_up([g2_b1]))
    print(syn.bottom_up([g2_b2]))
    print(syn.bottom_up([g2_b3]))
    print(syn.bottom_up([g2_b4]))
    print(syn.bottom_up([g2_b5]))
    print(syn.bottom_up([g2_b6]))
    print(syn.bottom_up([g2_b7]))
    print(syn.bottom_up([g2_b10]))
    print(syn.bottom_up([g2_b11]))
    e = time.time()
    print("{} {}".format(type(syn), e - s))

    s = time.time()
    syn = BadExamplesSynthesizer(g2)
    print(syn.bottom_up([g2_b1, g2_b1_n]))
    print(syn.bottom_up([g2_b2, g2_b2_n]))
    print(syn.bottom_up([g2_b3, g2_b3_n]))
    print(syn.bottom_up([g2_b4, g2_b4_n]))
    print(syn.bottom_up([g2_b5, g2_b5_n]))
    print(syn.bottom_up([g2_b6, g2_b6_n]))
    print(syn.bottom_up([g2_b7, g2_b7_n]))
    print(syn.bottom_up([g2_b10, g2_b10_n]))
    print(syn.bottom_up([g2_b11, g2_b11_n]))
    e = time.time()
    print("{} {}".format(type(syn), e - s))

    s = time.time()
    syn = ConstraintsSynthesizer(g2)
    print(syn.bottom_up([g2_b1, g2_b1_n, g2_c1]))
    print(syn.bottom_up([g2_b2, g2_b2_n, g2_c2]))
    print(syn.bottom_up([g2_b3, g2_b3_n, g2_c3]))
    print(syn.bottom_up([g2_b4, g2_b4_n, g2_c4]))
    print(syn.bottom_up([g2_b5, g2_b5_n, g2_c5]))
    print(syn.bottom_up([g2_b6, g2_b6_n, g2_c6]))
    print(syn.bottom_up([g2_b7, g2_b7_n, g2_c7]))
    print(syn.bottom_up([g2_b10, g2_b10_n, g2_c10]))
    print(syn.bottom_up([g2_b11, g2_b11_n, g2_c11]))
    print(syn.bottom_up([g2_b12, g2_b12_n, g2_c12]))
    e = time.time()
    print("{} {}".format(type(syn), e - s))

    s = time.time()
    syn = OptimizedSynthesizer(g2)
    print(syn.bottom_up([g2_b1, g2_b1_n, g2_c1]))
    print(syn.bottom_up([g2_b2, g2_b2_n, g2_c2]))
    print(syn.bottom_up([g2_b3, g2_b3_n, g2_c3]))
    print(syn.bottom_up([g2_b4, g2_b4_n, g2_c4]))
    print(syn.bottom_up([g2_b5, g2_b5_n, g2_c5]))
    print(syn.bottom_up([g2_b6, g2_b6_n, g2_c6]))
    print(syn.bottom_up([g2_b7, g2_b7_n, g2_c7]))
    print(syn.bottom_up([g2_b8, g2_b8_n, g2_c8]))
    print(syn.bottom_up([g2_b9, g2_b9_n, g2_c9]))
    print(syn.bottom_up([g2_b10, g2_b10_n, g2_c10]))
    print(syn.bottom_up([g2_b11, g2_b11_n, g2_c11]))
    print(syn.bottom_up([g2_b12, g2_b12_n, g2_c12]))
    e = time.time()
    print("{} {}".format(type(syn), e - s))

    s = time.time()
    syn = NoDuplicatesSynthesizer(g2)
    print(syn.bottom_up([g2_b1, g2_b1_n, g2_c1]))
    print(syn.bottom_up([g2_b2, g2_b2_n, g2_c2]))
    print(syn.bottom_up([g2_b3, g2_b3_n, g2_c3]))
    print(syn.bottom_up([g2_b4, g2_b4_n, g2_c4]))
    print(syn.bottom_up([g2_b5, g2_b5_n, g2_c5]))
    print(syn.bottom_up([g2_b6, g2_b6_n, g2_c6]))
    print(syn.bottom_up([g2_b7, g2_b7_n, g2_c7]))
    print(syn.bottom_up([g2_b8, g2_b8_n, g2_c8]))
    print(syn.bottom_up([g2_b9, g2_b9_n, g2_c9]))
    print(syn.bottom_up([g2_b10, g2_b10_n, g2_c10]))
    print(syn.bottom_up([g2_b11, g2_b11_n, g2_c11]))
    print(syn.bottom_up([g2_b12, g2_b12_n, g2_c12]))
    e = time.time()
    print("{} {}".format(type(syn), e - s))

    s = time.time()
    syn = Synthesizer(g3)
    print(syn.bottom_up([g3_b1]))
    print(syn.bottom_up([g3_b2]))
    print(syn.bottom_up([g3_b3]))
    print(syn.bottom_up([g3_b4]))
    print(syn.bottom_up([g3_b5]))
    print(syn.bottom_up([g3_b6]))
    print(syn.bottom_up([g3_b7]))
    print(syn.bottom_up([g3_b8]))
    print(syn.bottom_up([g3_b9]))
    print(syn.bottom_up([g3_b10]))
    print(syn.bottom_up([g3_b11]))
    e = time.time()
    print("{} {}".format(type(syn), e - s))

    s = time.time()
    syn = BadExamplesSynthesizer(g3)
    print(syn.bottom_up([g3_b1, g3_b1_n]))
    print(syn.bottom_up([g3_b2, g3_b2_n]))
    print(syn.bottom_up([g3_b3, g3_b3_n]))
    print(syn.bottom_up([g3_b4, g3_b4_n]))
    print(syn.bottom_up([g3_b5, g3_b5_n]))
    print(syn.bottom_up([g3_b6, g3_b6_n]))
    print(syn.bottom_up([g3_b7, g3_b7_n]))
    print(syn.bottom_up([g3_b8, g3_b8_n]))
    print(syn.bottom_up([g3_b9, g3_b9_n]))
    print(syn.bottom_up([g3_b10, g3_b10_n]))
    print(syn.bottom_up([g3_b11, g3_b11_n]))
    e = time.time()
    print("{} {}".format(type(syn), e - s))

    s = time.time()
    syn = ConstraintsSynthesizer(g3)
    print(syn.bottom_up([g3_b1, g3_b1_n, g3_c1]))
    print(syn.bottom_up([g3_b2, g3_b2_n, g3_c2]))
    print(syn.bottom_up([g3_b3, g3_b3_n, g3_c3]))
    print(syn.bottom_up([g3_b4, g3_b4_n, g3_c4]))
    print(syn.bottom_up([g3_b5, g3_b5_n, g3_c5]))
    print(syn.bottom_up([g3_b6, g3_b6_n, g3_c6]))
    print(syn.bottom_up([g3_b7, g3_b7_n, g3_c7]))
    print(syn.bottom_up([g3_b8, g3_b8_n, g3_c8]))
    print(syn.bottom_up([g3_b9, g3_b9_n, g3_c9]))
    print(syn.bottom_up([g3_b10, g3_b10_n, g3_c10]))
    print(syn.bottom_up([g3_b11, g3_b11_n, g3_c11]))
    e = time.time()
    print("{} {}".format(type(syn), e - s))

    s = time.time()
    syn = OptimizedSynthesizer(g3)
    print(syn.bottom_up([g3_b1, g3_b1_n, g3_c1]))
    print(syn.bottom_up([g3_b2, g3_b2_n, g3_c2]))
    print(syn.bottom_up([g3_b3, g3_b3_n, g3_c3]))
    print(syn.bottom_up([g3_b4, g3_b4_n, g3_c4]))
    print(syn.bottom_up([g3_b5, g3_b5_n, g3_c5]))
    print(syn.bottom_up([g3_b6, g3_b6_n, g3_c6]))
    print(syn.bottom_up([g3_b7, g3_b7_n, g3_c7]))
    print(syn.bottom_up([g3_b8, g3_b8_n, g3_c8]))
    print(syn.bottom_up([g3_b9, g3_b9_n, g3_c9]))
    print(syn.bottom_up([g3_b10, g3_b10_n, g3_c10]))
    print(syn.bottom_up([g3_b11, g3_b11_n, g3_c11]))
    e = time.time()
    print("{} {}".format(type(syn), e - s))

    s = time.time()
    syn = NoDuplicatesSynthesizer(g3)
    print(syn.bottom_up([g3_b1, g3_b1_n, g3_c1]))
    print(syn.bottom_up([g3_b2, g3_b2_n, g3_c2]))
    print(syn.bottom_up([g3_b3, g3_b3_n, g3_c3]))
    print(syn.bottom_up([g3_b4, g3_b4_n, g3_c4]))
    print(syn.bottom_up([g3_b5, g3_b5_n, g3_c5]))
    print(syn.bottom_up([g3_b6, g3_b6_n, g3_c6]))
    print(syn.bottom_up([g3_b7, g3_b7_n, g3_c7]))
    print(syn.bottom_up([g3_b8, g3_b8_n, g3_c8]))
    print(syn.bottom_up([g3_b9, g3_b9_n, g3_c9]))
    print(syn.bottom_up([g3_b10, g3_b10_n, g3_c10]))
    print(syn.bottom_up([g3_b11, g3_b11_n, g3_c11]))
    e = time.time()
    print("{} {}".format(type(syn), e - s))

    s = time.time()
    syn = NoDuplicatesSynthesizer(g2_plus)
    print(syn.bottom_up([g2_plus_b1, g2_plus_b1_n, g2_plus_c1]))
    print(syn.bottom_up([g2_plus_b2, g2_plus_b2_n, g2_plus_c2]))
    print(syn.bottom_up([g2_plus_b3, g2_plus_b3_n, g2_plus_c3]))
    print(syn.bottom_up([g2_plus_b4, g2_plus_b4_n, g2_plus_c4]))
    print(syn.bottom_up([g2_plus_b5, g2_plus_b5_n, g2_plus_c5]))
    e = time.time()
    print("{} {}".format(type(syn), e - s))

    set_global_pool()

    s = time.time()
    syn = ParallelSynthesizer(g1)
    print(syn.bottom_up([g1_b1, g1_b1_n, g1_c1]))
    print(syn.bottom_up([g1_b2, g1_b2_n, g1_c2]))
    print(syn.bottom_up([g1_b3, g1_b3_n, g1_c3]))
    print(syn.bottom_up([g1_b4, g1_b4_n, g1_c4]))
    print(syn.bottom_up([g1_b5, g1_b5_n, g1_c5]))
    print(syn.bottom_up([g1_b7, g1_b7_n, g1_c7]))
    print(syn.bottom_up([g1_b8, g1_b8_n, g1_c8]))
    print(syn.bottom_up([g1_b9, g1_b9_n, g1_c9]))
    e = time.time()
    print("{} {}".format(type(syn), e - s))

    s = time.time()
    print(syn.bottom_up([g2_b1, g2_b1_n, g2_c1]))
    print(syn.bottom_up([g2_b2, g2_b2_n, g2_c2]))
    print(syn.bottom_up([g2_b10, g2_b10_n, g2_c10]))
    print(syn.bottom_up([g2_b12, g2_b12_n, g2_c12]))
    e = time.time()
    print("{} {}".format(type(syn), e - s))

    s = time.time()
    print(syn.bottom_up([g3_b1, g3_b1_n, g3_c1]))
    print(syn.bottom_up([g3_b2, g3_b2_n, g3_c2]))
    print(syn.bottom_up([g3_b4, g3_b4_n, g3_c4]))
    print(syn.bottom_up([g3_b5, g3_b5_n, g3_c5]))
    print(syn.bottom_up([g3_b6, g3_b6_n, g3_c6]))
    print(syn.bottom_up([g3_b9, g3_b9_n, g3_c9]))
    print(syn.bottom_up([g3_b11, g3_b11_n, g3_c11]))
    e = time.time()
    print("{} {}".format(type(syn), e - s))


if __name__ == "__main__":
    main()
