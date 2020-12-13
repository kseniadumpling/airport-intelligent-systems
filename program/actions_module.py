import rdflib
import formulas_module as frml
import algorithms_module as alg
import ml_module as ml

def split_res(string):
    return string.split(':')[-1]

def is_correct_action(name):
    g = rdflib.Graph()
    g.load("../knowledge_base.n3", format="n3")

    flag = False
    for row in g.query("SELECT ?x WHERE { ?x rdf:type mc:Action .}"):
        act = split_res(row.x)
        if act == name:
            flag = True

    return flag

def call_action(act_name, data_obj = {}, res = [], additional = []):
    if is_correct_action(act_name) == False: 
        raise ValueError('Undefined action: {}'.format(act_name))

    g = rdflib.Graph()
    g.load("../knowledge_base.n3", format="n3")

    req_for_needs = 'SELECT ?s WHERE { mo:' + act_name + ' mp:needs ?s .}'
    for row in g.query(req_for_needs):
        module = split_res(row.s)
        req_for_calls = 'SELECT ?s WHERE { mo:' + act_name + ' mp:calls ?s .}'

        if module == 'formula':
            for row in g.query(req_for_calls):
                formula = split_res(row.s)
                frml.call_formulas(formula, data_obj, res)
        elif module == 'algorithm':
            for row in g.query(req_for_calls):
                algorithm = split_res(row.s)
                alg.call_algorithms(algorithm, data_obj, res, additional)
        elif module == 'neural_network':
            for row in g.query(req_for_calls):
                nn = split_res(row.s)
                ml.call_nn(nn, res)


if __name__ == "__main__":
    print('action.py file')