import clingo
import pandas as pd
from pandas import DataFrame
import warnings
import re

class ActionDecider:

    def __init__(self, config):
        super().__init__()

        f = open(config["constraints"])
        self.constraints = f.read()
        f.close()

        f = open(config["show_constraints"])
        self.show = f.read()
        f.close()

    def decide(self, current_facts):

        # pass to solver.
        clingo_symbols = Solver().solve(self.constraints, current_facts, self.show)

        symbols = list(map(lambda x: self.convert_symbol_to_term(x), clingo_symbols))

        actions = []
        for symbol in symbols:
            number = re.search(r'\((\d+)\)', symbol).group(1)
            number = int(number)
            actions.append(number)

        return actions

    def convert_symbol_to_term(self, symbol: clingo.Symbol):
        name = symbol.name
        arguments = symbol.arguments

        term = "" + name + "("
        argstring = ",".join(map(str, arguments))
        term += argstring
        term += ")."

        return term


class Solver:
    def __init__(self):
        super().__init__()
        self.atoms = []

    def solve(self, constraints, facts, show):

        control = clingo.Control(message_limit=0)
        control.configuration.solve.models = 0

        # add asp
        control.add("base", [], constraints)
        control.add("base", [], facts)
        control.add("base", [], show)

        control.ground([("base", [])])

        handle = control.solve(on_model=self.on_model)

        if handle.satisfiable:
            return self.atoms

        return None

    def on_model(self, model):
        """
        This is the observer callback for the clingo Control object after solving. It is done in a separate thread,
        which is why we use a new instantiation of the Solver class, otherwise its state is not thread safe
        :param model:
        """
        # print("Found solution:", model)
        symbols = model.symbols(shown=True)
        for symbol in symbols:
            # print(symbol)
            self.atoms.append(symbol)



