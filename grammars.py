import copy
import time
from enum import Enum
from itertools import product

from inputs import *


class GrammarType(Enum):
    NUM = "numeric"
    STR = "string"
    LIST = "list"


class Grammar:
    def __init__(self, grammar_type: GrammarType, non_terminals, rules_groups,
                 starting):
        self.grammar_type = grammar_type
        self.syntax = non_terminals
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

    def str_trivial_programs(self):
        return []

    def list_trivial_programs(self):
        return ["Sort(Sort(", "Reverse(Reverse(", "Slice(Slice(",
                "0+", "+0", "1+1"]


class Synthesizer:

    def __init__(self, g: Grammar):
        self.g = g
        self.P = {}
        self.non_checked_programs = []
        self.E = None

    def bottom_up(self, E):
        self.init_program(E)
        print(len(self.non_checked_programs))
        # todo: create non ground rules dict
        while True:
            self.grow()
            print(len(self.non_checked_programs))
            for p in self.non_checked_programs:
                try:
                    if is_program_match(p, E):
                        return p
                except:
                    pass
            self.non_checked_programs = []

    def init_program(self, E):
        """
        deduce the base programs by the grammar
        :param grammar:
        :return:
        """
        self.E = E
        self.init_starting_programs()

    def grow(self):
        new_P = copy.deepcopy(self.P)
        for r_g in self.g.rules_groups:
            for r in self.g.rules_groups[r_g]:
                if not ground(r):
                    list_products = self.produce(r)
                    for p in list_products:
                        p = "".join(p)
                        try:
                            if self.insert_program_condition(p):
                                new_P[r_g].append(p)
                                if r_g == self.g.starting:
                                    self.non_checked_programs.append(p)
                        except:
                            pass
        self.P = new_P

    def is_new_program(self, p):
        for p2 in self.P[self.g.starting]:
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
        splitted_programs = product(*lists_for_product)
        return splitted_programs

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

    def insert_program_condition(self, p):
        return not any(s in p for s in self.g.trivial_programs) \
               and self.is_new_program(p)


class BadExamplesSynthesizer(Synthesizer):
    def __init__(self, g):
        super().__init__(g)
        self.E_not = None

    def bottom_up(self, E):
        self.init_program(E)
        print(len(self.non_checked_programs))
        while True:
            self.grow()
            print(len(self.non_checked_programs))
            for p in self.non_checked_programs:
                try:
                    if is_program_match(p, self.E):
                        return p
                except:
                    pass
            self.non_checked_programs = []

    def init_program(self, E):
        self.E = E[0]
        self.E_not = E[1]
        self.init_starting_programs()

    def is_counter_example(self, t):
        for i, o in self.E_not:
            x_input = i
            if eval(t) == o:
                return True
        return False

    def insert_program_condition(self, p):
        return super().insert_program_condition(p) \
               and not self.is_counter_example(p)


class ConstraintsSynthesizer(BadExamplesSynthesizer):
    def __init__(self, g):
        super().__init__(g)
        self.C = None

    def init_program(self, E):
        super().init_program(E)
        self.C = E[2]

    def insert_program_condition(self, p):
        return super().insert_program_condition(p) \
               and self.is_matching_constraints(p)

    def is_matching_constraints(self, p):
        return all([c(p) for c in self.C])


class OptimizedSynthesizer(ConstraintsSynthesizer):
    def __init__(self, g):
        super().__init__(g)
        self.solution = None

    def bottom_up(self, E):
        self.init_program(E)
        self.solution = None
        try:
            for p in self.P[self.g.starting]:
                if ground(p) and is_program_match(p, self.E):
                    return p
        except:
            pass

        while self.solution is None:
            print(len(self.P[self.g.starting]))
            self.grow()
        return self.solution

    def grow(self):
        new_P = copy.deepcopy(self.P)
        for r_g in self.g.rules_groups:
            for r in self.g.rules_groups[r_g]:
                if not ground(r):
                    list_products = self.produce(r)
                    for p in list_products:
                        p = "".join(p)
                        try:
                            if self.insert_program_condition(p):
                                new_P[r_g].append(p)
                                if r_g == self.g.starting:
                                    if is_program_match(p, self.E):
                                        self.solution = p
                                        return
                        except:
                            pass
        self.P = new_P


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


def is_program_match(p, d):
    for i, o in d:
        x_input = i
        if eval(p) != o:
            return False
    return True


def format_splited_programs(splitted_programs):
    formatted_programs = ["".join(p) for p in splitted_programs]
    return formatted_programs


def Concat(s1: str, s2: str) -> str:
    return s1 + s2


def Substr(s: str, i: int, j: int) -> str:
    return s[i:j]


def Len(s: str) -> int:
    return len(s)


def IndexOf(s1: str, s2: str) -> int:
    return s1.find((s2))


def Sort(l: list) -> list:
    return sorted(l)


def Reverse(l: list) -> list:
    l = l[:]
    l.reverse()
    return l


def Slice(s: list, i: int, j: int) -> list:
    return s[i:j]


# GRAMMAR 1: ARITHMETIC OPERATORS
g1 = Grammar(GrammarType.NUM,
             {"E", "OP", "NUM"},
             {"E": {"( E ) OP ( E )", "NUM", "x_input"},
              "OP": {"+", "-", "*", "/"},
              "NUM": {"1", "2", "6"}},
             "E")

g3 = Grammar(GrammarType.LIST,
             {"E", "NUM"},
             {"E": {"x_input", "Sort( E )", "Reverse( E )",
                    "Slice( E , NUM , NUM )"},
              "NUM": {"0", "1", "2", "NUM + NUM"}},
             "E")


def main():
    syn = BadExamplesSynthesizer(g1)
    print(syn.bottom_up([g1_b1, g1_b1_n]))
    print(syn.bottom_up([g1_b2, g1_b2_n]))
    print(syn.bottom_up([g1_b3, g1_b3_n]))
    print(syn.bottom_up([g1_b4, g1_b4_n]))
    print(syn.bottom_up([g1_b5, g1_b5_n]))

    s = time.time()
    syn = Synthesizer(g3)
    print(syn.bottom_up(g3_b1))
    print(syn.bottom_up(g3_b2))
    print(syn.bottom_up(g3_b3))
    print(syn.bottom_up(g3_b4))
    print(syn.bottom_up(g3_b5))
    print(syn.bottom_up(g3_b6))
    print(syn.bottom_up(g3_b7))
    e = time.time()
    print(e - s)

    s = time.time()
    syn = BadExamplesSynthesizer(g3)
    print(syn.bottom_up([g3_b1, g3_b1_n]))
    print(syn.bottom_up([g3_b2, g3_b2_n]))
    print(syn.bottom_up([g3_b3, g3_b3_n]))
    print(syn.bottom_up([g3_b4, g3_b4_n]))
    print(syn.bottom_up([g3_b5, g3_b5_n]))
    print(syn.bottom_up([g3_b6, g3_b6_n]))
    print(syn.bottom_up([g3_b7, g3_b7_n]))
    e = time.time()
    print(e - s)

    s = time.time()
    syn = ConstraintsSynthesizer(g3)
    print(syn.bottom_up([g3_b1, g3_b1_n, g1_c1]))
    print(syn.bottom_up([g3_b2, g3_b2_n, g1_c2]))
    print(syn.bottom_up([g3_b3, g3_b3_n, g1_c3]))
    print(syn.bottom_up([g3_b4, g3_b4_n, g1_c4]))
    print(syn.bottom_up([g3_b5, g3_b5_n, g1_c5]))
    print(syn.bottom_up([g3_b6, g3_b6_n, []]))
    print(syn.bottom_up([g3_b7, g3_b7_n, g1_c7]))
    e = time.time()
    print(e - s)

    s = time.time()
    syn = OptimizedSynthesizer(g3)
    print(syn.bottom_up([g3_b1, g3_b1_n, []]))
    print(syn.bottom_up([g3_b2, g3_b2_n, g1_c2]))
    print(syn.bottom_up([g3_b3, g3_b3_n, g1_c3]))
    print(syn.bottom_up([g3_b4, g3_b4_n, g1_c4]))
    print(syn.bottom_up([g3_b5, g3_b5_n, g1_c5]))
    print(syn.bottom_up([g3_b6, g3_b6_n, []]))
    print(syn.bottom_up([g3_b7, g3_b7_n, g1_c7]))
    e = time.time()
    print(e - s)


if __name__ == "__main__":
    main()
