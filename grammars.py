import copy
import time
from enum import Enum
from itertools import product
from multiprocessing.pool import Pool

from inputs import *

# noinspection PyGlobalUndefined
global pool  # Enables pool reuse


def set_global_pool():
    """
    initiates pool with python processes globally
    """
    global pool
    pool = Pool(7)  # type: Pool


class GrammarType(Enum):
    """
    Helper Enum class to identify different types
    of grammars
    """
    NUM = "numeric"
    STR = "string"
    LIST = "list"


class Grammar:
    """
    Exposes a convenient API for grammar specification
    """
    def __init__(self, grammar_type: GrammarType, rules_groups, starting):
        """
        creates Grammar object
        :param grammar_type: Enum identifies the type of grammar
        :param rules_groups: Mapping from left side of rule to all of
        its right side products. E := 1|2|3 is equivalent to
        rules_groups[E] = [1, 2, 3]
        :param starting: the starting expression of the grammar
        """
        self.grammar_type = grammar_type
        self.rules_groups = rules_groups
        self.starting = starting
        self.trivial_programs = self.create_trivial_programs()

    def create_trivial_programs(self):
        """
        creating trivial programs of the grammar, such as
        reverse(reverse(list))
        :return: list of trivial programs
        """
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
    """
    Synthesizer class implementing the heart of the program synthesis
    including it's algorithms, bottom-up with observational equivalence
    pruning.
    """
    def __init__(self, g: Grammar):
        """
        P is a the set of lists of all programs in a given produced mapped to
        their left side expression (rule name).
        non_checked_programs is a list of all programs not yet checked.
        E is the set of examples.
        sizes is a set mapping the sizes of all the lists in P to their
        left side expression (rule name).
        :param g: the grammar used by the Synthesizer.
        """
        self.g = g
        self.P = {}
        self.non_checked_programs = []
        self.E = None
        self.sizes = {}

    def bottom_up(self, E):
        """
        The heart of the synthesis process. it is the implementation of the
        classic bottom-up algorithm with observational equivalence
        :param E: Examples the synthesizer works by
        :return: A solution if found or None if non-realizability detected
        """
        self.init_program(E)  # save examples and creating root programs
        if self.is_unrealizable():  # check if non-realizable
            return None
        # print(len(self.non_checked_programs))
        while True:
            self.save_sizes()  # save sizes to check for progress
            self.grow()  # perform BFS 1-level programs creation
            if self.is_stuck():  # check if in dead end
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
        Deduce the base programs by the grammar
        """
        self.E = E[0]
        self.init_starting_programs()

    def grow(self):
        """
        The BFS programs generation step of the bottom-up algorithm.
        performs a single level of BFS in the programs tree of the
        grammar
        """
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
        """
        Create new programs out of some rule and the existing programs
        in the program tree
        :param r: A right side of a rule (not ground program)
        :return: A generator object of programs in a tuple format
        """
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
        """
        Creates the root programs from them the algorithm can keep produce
        new programs
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
                    if r == self.g.starting:
                        self.non_checked_programs.append(t)

    def insert_program_condition(self, p, r_g):
        """
        Check if a program satisfies a condition for being inserted
        into the programs tree
        :param p: A program in a string format
        :param r_g: The rules group left-side expression
        :return: True if p satisfies the condition, False otherwise
        """
        return not any(s in p for s in self.g.trivial_programs) \
               and self.is_new_program(p, r_g)

    def is_unrealizable(self):
        """
        Check if the target program can be declared unrealizable. a program is
        unrealizable if the examples from which it should synthesize the
        program are inconsistent
        :return: True if unrealizable, False otherwise
        """
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
        """
        Saving the sizes of all lists in self.P
        """
        self.sizes = {}
        for r in self.P:
            self.sizes[r] = len(self.P[r])

    def is_stuck(self):
        """
        Check if the synthesizer does'nt produce any more new programs
        :return: True if stuck, False otherwise
        """
        for r in self.P:
            if self.sizes[r] != len(self.P[r]):
                return False
        return True


class BadExamplesSynthesizer(Synthesizer):
    """
    Adds on the classic Synthesizer a negative-examples component. Each program
    p must satisfy p(i) != o for each (i,o) in negative-examples in order
    to be in the programs tree
    """
    def __init__(self, g):
        super().__init__(g)
        self.E_not = None

    def init_program(self, E):
        """
        :param E: E[0] is a list of examples, E[1] is a list of negative
        examples
        """
        self.E = E[0]
        self.E_not = E[1]
        self.init_starting_programs()

    # noinspection PyUnusedLocal
    def is_counter_example(self, p):
        """
        Check if p is not satisfying p(i) != o for each (i,o)
        in negative-examples
        :param p: A program to check in a a string format
        :return: True if not not satisfying p(i) != o for each (i,o)
        in negative-examples, False otherwise
        """
        for i, o in self.E_not:
            x_input = i
            if eval(p) == o:
                return True
        return False

    def insert_program_condition(self, p, r_g):
        return super().insert_program_condition(p, r_g) \
               and not self.is_counter_example(p)

    def is_unrealizable(self):
        """
        Same as in Synthesizer with an addition of contradictions between
        examples and negative-examples
        :return: True if unrealizable, False otherwise
        """
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
    """
    Adds on the BadExamplesSynthesizer a user-defined constraints component.
    Each program p must satisfy c(p) == True for each c in constraints in
    order to be in the programs tree
    """
    def __init__(self, g):
        super().__init__(g)
        self.C = None

    def init_program(self, E):
        """

        :param E: E[0] is a list of examples, E[1] is a list of negative
        examples, E[2] is a list of user-defined constraints
        """
        super().init_program(E)
        self.C = E[2]

    def insert_program_condition(self, p, r_g):
        return super().insert_program_condition(p, r_g) \
               and self.is_matching_constraints(p)

    def is_matching_constraints(self, p):
        """
        Check that a program p satisfies all user-defined constraints
        :param p: A program to check in a a string format
        :return: True if p satisfies all user-defined constraints,
        False otherwise
        """
        return all([c(p) for c in self.C])


# noinspection PyPep8Naming,PyBroadException
class OptimizedSynthesizer(ConstraintsSynthesizer):
    """
    Adds on the ConstraintsSynthesizer a performance optimization. Drastically
    improves performance and finds programs that lower performaning synthesizers
    do not find in reasonable time
    """
    def __init__(self, g):
        super().__init__(g)
        self.solution = None

    def bottom_up(self, E):
        """
        A slightly modified bottom-up algorithm
        :param E: E[0] is a list of examples, E[1] is a list of negative
        examples, E[2] is a list of user-defined constraints
        :return: A solution if found or None if non-realizability detected
        """
        self.init_program(E)
        if self.is_unrealizable():
            return None
        self.solution = None
        # first check all starting programs
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
                                    # The optimization - check a program right
                                    # after it is created
                                    if is_program_match(p, self.E):
                                        self.solution = p
                                        return
                        except:
                            pass
        self.P = new_P


# noinspection PyBroadException
class NoDuplicatesSynthesizer(OptimizedSynthesizer):
    """
    Adds on the OptimizedSynthesizer a performance optimization. Removes
    more duplicates than other synthesizers. Can lead to non-solved programs
    where OptimizedSynthesizer would have been solved them
    """
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
        # Check also programs in the same level of BFS
        for p2 in self.new_P[r_g]:
            if is_observational_equivalent(p, p2, self.E):
                return False
        return True


# noinspection PyBroadException
class ParallelSynthesizer(NoDuplicatesSynthesizer):
    """
    Adds on the NoDuplicatesSynthesizer a parallel component. Uses python
    multiprocessing to use more resources in order speed up program synthesis.
    Can solve programs that NoDuplicatesSynthesizer would not have solved, or
    even declared un-realizable because duplicated programs (in terms of OE)
    can slip into the programs tree
    """
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
            global pool  # use global pool to save overhead!
            x = pool  # type: Pool
            self.new_P = {}
            for r in self.g.rules_groups:
                self.new_P[r] = []
            for r_g in self.g.rules_groups:
                for r in self.g.rules_groups[r_g]:
                    if not ground(r):
                        list_products = map(lambda var: ("".join(var), r_g),
                                            self.produce(r))
                        # parallel run insert_program_condition_parallel
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
        """
        A dedicated parallel version of insert_program_condition
        :param A: A[0] is a program in string format, A[1] is a rules gropu
        left-side expression
        :return: True if A[0] satisfies the condition, False otherwise
        """
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
    """
    Check if p1 and p2 are Observational Equivalent
    :param p1: A program in string format
    :param p2: A program in string format
    :param E: A list of input-output pairs
    :return: True if p1 ==(OE) p2, False otherwise
    """
    for i, o in E:
        x_input = i
        if eval(p1) != eval(p2):
            return False
    return True


def ground(p: str) -> bool:
    """
    Check if a program is terminal. a terminal program is a program
    that cannot be further produced
    :param p: a program in string format
    :return: True if p is ground, False otherwise
    """
    tokens = p.split()
    for s in tokens:
        if s.isupper():
            return False
    return True


# noinspection PyUnusedLocal
def is_program_match(p, E):
    """
    Check if a program p is matching the examples
    :param p: A program in string format
    :param E: A list of input-output pairs
    :return: True if p matches all the examples in E, False otherwise
    """
    for i, o in E:
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
