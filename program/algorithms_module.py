import rdflib

def split_res(string):
    return string.split(':')[-1]

def call_algorithms(alg_name, obj = {}, res = [], additional = []):
    g = rdflib.Graph()
    g.load("../knowledge_base.n3", format="n3")

    if alg_name == 'analyze_fuel':
        lower_bounds = 0
        upped_bounds = 0
        critical_value = 0

        for row in g.query("SELECT ?s WHERE { mo:fuel_percent mp:no_less_than ?s .}"):
            lower_bounds = int(split_res(row.s))
        for row in g.query("SELECT ?s WHERE { mo:fuel_percent mp:no_more_than ?s .}"):
            upped_bounds = int(split_res(row.s))
        for row in g.query("SELECT ?s WHERE { mo:fuel_percent mp:critical_value ?s .}"):
            critical_value = int(split_res(row.s))

        if 'fuel_percent' in obj:
            if obj['fuel_percent'] < lower_bounds or obj['fuel_percent'] > upped_bounds:
                res.append('fuel_percent')
            elif obj['fuel_percent'] <= critical_value:
                obj['emergency'] = 1

    elif alg_name == 'analyze_emergency':
        priority_val = 0
        for row in g.query("SELECT ?s WHERE { mo:fuel_percent mp:priority_restrict_value ?s .}"):
            priority_val = int(split_res(row.s))

        if 'emergency' in obj:
            if obj['emergency'] == 0:
                if obj['priority'] < priority_val: 
                    obj['priority'] = priority_val
            else:
                if obj['priority'] > priority_val: 
                    obj['priority'] = priority_val

    elif alg_name == 'analyze_directions':
        wind_dir = additional[0]
        wind_speed = additional[1]

        if wind_speed > 5:
            if wind_dir == 'n' or wind_dir == 's':
                if obj['direction'] == 'e' or obj['direction'] == 'w':
                    obj['state'] = 'closed'
                else:
                    obj['state'] = 'available'
            elif wind_dir == 'e' or wind_dir == 'w':
                if obj['direction'] == 'n' or obj['direction'] == 's':
                    obj['state'] = 'closed'
                else:
                    obj['state'] = 'available'

    elif alg_name == 'analyze_order':
        aircrafts = additional[0]
        runways = additional[1]

        for aircraft in aircrafts:
            runway_obj = min(runways, key=lambda x: x['queue'])
            runway_obj['queue'] = runway_obj['queue'] + 1

            tmp = {
                "aircraft": aircraft['name'],
                "runway": runway_obj['id']
            }

            res.append(tmp)

    else:
        raise ValueError('Undefined alg_name: {}'.format(alg_name))

if __name__ == "__main__":
    print('algorithms_module.py file')