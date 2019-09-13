import os

import grammars as syn
import pandas as pd
from multiprocessing import Process, Queue
import time
from inputs import *
import sys


def run_synthesis(synthesizer, benchmark: list, queue):
    start = time.perf_counter()
    program = synthesizer.bottom_up(benchmark)
    end = time.perf_counter()
    queue.put((program, end - start))


def get_time_breakdown(seconds):
    millis = seconds * 1000
    seconds = int(seconds)
    millis -= (seconds / 1000)
    micros = millis * 1000
    millis = int(millis)
    micros -= (millis / 1000)
    micros = int(micros)
    return seconds, millis, micros


# noinspection PyBroadException
def test_grammar(grammar: syn.Grammar, grammar_name, benchmarks: list,
                 synthesizer_classes) -> pd.DataFrame:
    output_dir_path = "results"
    os.makedirs(output_dir_path, exist_ok=True)
    grammar_dir_path = output_dir_path + "/" + grammar_name
    os.makedirs(grammar_dir_path, exist_ok=True)
    synthesizers = [s(grammar) for s in synthesizer_classes]
    outputs = []
    grammar_type_str = str(grammar.grammar_type).split('.')[1]
    for i, benchmark in enumerate(benchmarks):
        for synthesizer in synthesizers:
            synthesizer_name = type(synthesizer).__name__
            output = [grammar_type_str, grammar_name, i, synthesizer_name]
            if not issubclass(type(synthesizer), syn.BadExamplesSynthesizer):
                output.append("N\A")
            else:
                output.append(len(benchmark[1]))
            if issubclass(type(synthesizer), syn.ConstraintsSynthesizer):
                if len(benchmark[2]) > 0:
                    output.append("Yes")
                else:
                    output.append("No")
            else:
                output.append("N\A")
            time_measured = 0
            iterations = 3
            process = None
            program = None
            try:
                for ite in range(iterations):
                    queue = Queue()
                    process = Process(target=run_synthesis,
                                      args=(synthesizer, benchmark, queue))
                    process.start()
                    result = queue.get(timeout=TIME)
                    process.join()
                    time_measured += result[1]
                    program = result[0]
                seconds, millis, micros = get_time_breakdown(
                    time_measured / iterations)
                if program is None:
                    output.append("Unrealizable")
                else:
                    output.append(program)
                output.append("{}::{}::{}".format(seconds, millis, micros))
            except:
                if process:
                    process.terminate()
                output.append("-")
                output.append("TIMEOUT=" + str(TIME))
            if len(output) != 8:
                print(output)
            outputs.append(output[:])
    df = pd.DataFrame(outputs,
                      columns=["Grammar Type", "Grammar Name", "Benchmark",
                               "Synthesizer", "Negative Examples",
                               "Constraints", "Solution",
                               "Avg Runtime[seconds::millis::micros]"])
    df.to_csv(grammar_dir_path + "/" + "results.csv")
    return df


if __name__ == '__main__':
    if sys.argv[0]:
        print(sys.argv[0])
    TIME = 150
    synthesizers_classes = [syn.Synthesizer,
                            syn.BadExamplesSynthesizer,
                            syn.ConstraintsSynthesizer,
                            syn.OptimizedSynthesizer,
                            syn.NoDuplicatesSynthesizer]
    num_df = test_grammar(syn.g1, "num_grammar", numeric_benchmarks,
                          synthesizers_classes)
    print(num_df)
    string_df = test_grammar(syn.g2, "str_grammar", string_benchmarks,
                             synthesizers_classes)
    print(string_df)
    list_df = test_grammar(syn.g3, "list_grammar", list_benchmarks,
                           synthesizers_classes)
    print(list_df)
    string_plus_df = test_grammar(syn.g2_plus, "str_plus_grammar",
                                  string_plus_benchmarks,
                                  [syn.NoDuplicatesSynthesizer])
    print(string_plus_df)
