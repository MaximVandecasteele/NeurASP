import clingo
import re

class Advisor():
    def __init__(self, config):

        self.memory = True

        f = open(config["game_rules"])
        self.positions = f.read()
        f.close()

        f = open(config["show_constraints"])
        self.show = f.read()
        f.close()

        f = open(config["show_airborne"])
        self.airborne = f.read()
        f.close()

        self.on_ground = True






    def advise(self, action, facts, on_ground):
        perform_No_Op = False
        advice_given = False
        action = int(action)

        clingo_symbols = Solver().solve(facts, self.positions, self.show)
        symbols = list(map(lambda x: self.convert_symbol_to_term(x), clingo_symbols))

        pattern = r'action\((.*?)\).'

        values = [re.search(pattern, s).group(1) for s in symbols]

        values = [int(val) for val in values]

        if self.memory == False and on_ground == True:
            perform_No_Op = True

        if perform_No_Op == True and (2 or 4 in values):
            action = 0
            advice_given = True
        elif 2 in values:
            action = 2
            advice_given = True


        perform_No_Op = False
        self.memory = on_ground



        # if self.on_ground and (2 in values or 4 in values):
        #     action = 2
        #     self.on_ground = False
        #     advice_given = True
        # elif (not self.on_ground) and 8 in values and (2 in values or 4 in values):
        #     action = 0
        #     self.on_ground = True
        #     advice_given = True
        # elif (not self.on_ground) and 8 in values:
        #     self.on_ground = True
        #     advice_given = True
        # elif (not self.on_ground) and (2 in values or 4 in values):
        #     action = 2
        #     advice_given = True



            # advice_given = True
        return action, advice_given

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

    def solve(self, facts, positions, show):

        control = clingo.Control(message_limit=0)
        control.configuration.solve.models = 0

        # add asp
        control.add("base", [], facts)
        control.add("base", [], positions)
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