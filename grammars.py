import copy
from enum import Enum
from itertools import product


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
        return ["((x_input)-(x_input))", "(x_input)/(x_input)", "(1)*", "*(1)",
                "(0)+", "+(0)", "*(0)", "(0)*", "(0)-", "-(0)", "/(0)", "(0)/"]

    def str_trivial_programs(self):
        return []

    def list_trivial_programs(self):
        return []


def init_program(grammar: Grammar) -> dict:
    """
    deduce the base programs by the grammar
    :param grammar:
    :return:
    """
    d = {}
    for r in grammar.rules_groups:
        for t in grammar.rules_groups[r]:
            if ground(t):
                if r in d:
                    d[r].append(t)
                else:
                    d[r] = [t]
    return d


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


def is_observational_equivalent(p1, p2, d):
    for i, o in d:
        x_input = i
        if eval(p1) != eval(p2):
            return False
    return True


def bottom_up(grammar: Grammar, d):
    P = init_program(grammar)
    # todo: create non ground rules dict
    while True:
        print(len(P[grammar.starting]))
        P = grow(grammar, P, d)
        for p in P[grammar.starting]:
            try:
                if is_program_match(p, d):
                    return p
            except:
                pass


def is_new_program(p, P_S, d):
    for p2 in P_S:
        if is_observational_equivalent(p, p2, d):
            return False
    return True


# noinspection PyBroadException
def grow(grammar: Grammar, P: dict, d):
    new_P = copy.deepcopy(P)
    for r_g in grammar.rules_groups:
        for r in grammar.rules_groups[r_g]:
            if not ground(r):
                list_products = produce(r, P)
                for p in list_products:
                    try:
                        if not any(s in p for s in grammar.trivial_programs) \
                                and is_new_program(p, P[grammar.starting], d):
                            new_P[r_g].append(p)
                    except:
                        pass
                # todo: is this ok?
                # new_P[r_g].extend(list_products)
    return new_P


def format_splited_programs(splitted_programs):
    formatted_programs = ["".join(p) for p in splitted_programs]
    return formatted_programs


def produce(r: str, P: dict):
    tokens = r.split()
    lists_for_product = []
    for t in tokens:
        if t.isupper():
            lists_for_product.append(P[t])
        else:
            lists_for_product.append([t])
    splitted_programs = product(*lists_for_product)
    return format_splited_programs(splitted_programs)


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


# def main():
#     g1 = Grammar(GrammarType.NUM,
#                  {"E", "OP", "NUM"},
#                  {"E": {"( E ) OP ( E )", "x_input", "NUM"},
#                   "OP": {"+", "-", "*", "/"},
#                   "NUM": {"0", "1", "2", "3", "4", "5", "6"}},
#                  "E")
#     print(bottom_up(g1, [(1, 7), (2, 12), (3, 15)]))
#
#
# if __name__ == "__main__":
#     main()
