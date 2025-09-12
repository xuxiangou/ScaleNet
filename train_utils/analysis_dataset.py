import json
from typing import List


def read_data(data_file, output_file=None):
    with open(data_file, 'r') as data:
        docs = json.load(data)
        f = open(output_file, "a")
        for catalyst_doc in docs:
            for chemical_symbol in catalyst_doc["atoms"]["chemical_symbols"]:
                f.write(chemical_symbol)
                f.write(str(catalyst_doc["atoms"]["symbol_counts"][chemical_symbol]))

            miller_str = " [" + ", ".join([str(i) for i in catalyst_doc["miller"]]) + "] "
            energy = str(catalyst_doc["energy"])
            f.write(miller_str)
            f.write(f" {energy}")
            f.write("\n")


def calc_non_repeated_data(data_file):
    molecule_list = []
    with open(data_file, 'r') as data:
        docs = json.load(data)
        for catalyst_doc in docs:
            molecule = []
            for chemical_symbol in catalyst_doc["atoms"]["chemical_symbols"]:
                molecule.append(chemical_symbol)
                molecule.append(str(catalyst_doc["atoms"]["symbol_counts"][chemical_symbol]))
            molecule = "".join(molecule)
            if molecule not in molecule_list:
                molecule_list.append(molecule)

    return len(molecule_list)


if __name__ == '__main__':
    # read_data("../data/h_data.json", "./h_data.txt")
    print(calc_non_repeated_data("../../data/co_data.json"))
