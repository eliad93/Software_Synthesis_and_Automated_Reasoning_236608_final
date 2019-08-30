import copy
from enum import Enum
from itertools import product
from queue import PriorityQueue


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
        return ["Sort(Sort(x_input))", "Reverse(Reverse(x_input))"]

    def list_trivial_programs(self):
        return []


class Synthesizer:

    def __init__(self, g: Grammar):
        self.g = g
        self.P = {}
        self.non_checked_programs = []
        self.E = None

    def bottom_up(self, E):
        self.E = E
        self.init_program()
        print(len(self.non_checked_programs))
        # todo: create non ground rules dict
        while True:
            self.grow()
            print(len(self.non_checked_programs))
            for p in self.non_checked_programs:  # self.P[self.g.starting]:
                try:
                    if is_program_match(p, E):
                        return p
                except:
                    pass
            # Ranking
            self.non_checked_programs = []

    def init_program(self):
        """
        deduce the base programs by the grammar
        :param grammar:
        :return:
        """
        self.P = {}
        self.non_checked_programs = []
        for r in self.g.rules_groups:
            for t in self.g.rules_groups[r]:
                if ground(t):
                    if r in self.P:
                        self.P[r].append(t)
                    else:
                        self.P[r] = [t]
                    # Ranking
                    if r == self.g.starting:
                        self.non_checked_programs.append(t)

    def grow(self):
        new_P = copy.deepcopy(self.P)
        for r_g in self.g.rules_groups:
            for r in self.g.rules_groups[r_g]:
                if not ground(r):
                    list_products = self.produce(r)
                    for p in list_products:
                        try:
                            if not any(s in p for s in self.g.trivial_programs) \
                                    and self.is_new_program(p):
                                new_P[r_g].append(p)
                                # Ranking
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
        return format_splited_programs(splitted_programs)


class Ranking(Synthesizer):
    def __init__(self, g):
        super().__init__(g)
        self.target_program = None

    def init_program(self):
        self.P = {}
        self.non_checked_programs = PriorityQueue()
        for r in self.g.rules_groups:
            for t in self.g.rules_groups[r]:
                if ground(t):
                    if r in self.P:
                        self.P[r].append(t)
                    else:
                        self.P[r] = [t]
                    # Ranking
                    if r == self.g.starting:
                        try:
                            self.insert_new_program(t)
                        except:
                            pass

    def bottom_up(self, E):
        self.E = E
        self.init_program()
        # todo: create non ground rules dict
        while True:
            self.target_program = self.non_checked_programs.get()
            try:
                if is_program_match(self.target_program, E):
                    return self.target_program
                else:
                    self.grow()
            except:
                pass

    def insert_new_program(self, t):
        priority = self.calculate_priority(t)
        self.non_checked_programs.put(priority, t)

    def calculate_priority(self, t):
        num_correct_outputs = 0
        for i, o in self.E:
            x_input = i
            if eval(t) == o:
                num_correct_outputs += 1
        return num_correct_outputs / len(self.E)


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


g3 = Grammar(GrammarType.LIST,
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


def main():
    g1 = Grammar(GrammarType.NUM,
                 {"E", "OP", "NUM"},
                 {"E": {"( E ) OP ( E )", "x_input", "NUM"},
                  "OP": {"+", "-", "*", "/"},
                  "NUM": {"0", "1", "2", "3", "4", "5", "6"}},
                 "E")
    syn = Synthesizer(g3)
    print(syn.bottom_up(g3_b1))
    print(syn.bottom_up(g3_b2))
    print(syn.bottom_up(g3_b3))
    print(syn.bottom_up(g3_b4))
    print(syn.bottom_up(g3_b5))
    print(syn.bottom_up(g3_b6))


if __name__ == "__main__":
    main()
