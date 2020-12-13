import rdflib

def call_formulas(func_name, obj = {}, res = []):
    g = rdflib.Graph()
    g.load("../knowledge_base.n3", format="n3")

    if func_name == 'formula_1':
        return 10**-2
    elif func_name == 'formula_2':
        return 10**-2
    else:
        raise ValueError('Undefined func_name: {}'.format(func_name))

if __name__ == "__main__":
    print('formulas_module.py file')