import rdflib
import pandas as pd

import db_module as db
import ml_module as ml
import actions_module as act

def split_res(string):
    return string.split(':')[-1]

def prepare_aircraft_info(aircraft):
    print("\n[Aircraft section] Processing info about next aircraft... \n")
    #---------https://github.com/RDFLib/rdflib/
    g = rdflib.Graph()
    g.load("../knowledge_base.n3", format="n3")

    field_list = []
    faulted_field_list = []

    print("[Aircraft section] Step 1: check the existance of all needed fields")
    for row in g.query("SELECT ?s WHERE { mo:aircraft mp:contains ?s .}"):
        field = split_res(row.s)
        field_list.append(field)

        if field not in aircraft:
            faulted_field_list.append(field)
        

    print("[Aircraft section] Step 2: check the state of the fields")
    for row in g.query("SELECT ?s WHERE { mo:aircraft mp:calls ?s .}"):
        call_arg = split_res(row.s)
        act.call_action(call_arg, aircraft, faulted_field_list)

    print("[Aircraft section] Step 2: fix the aircraft info if needed")
    if len(faulted_field_list) != 0:
        aircraft = ml.predict_fields(aircraft, faulted_field_list)

    print("[Aircraft section] Step 3: update list of aircrafts")
    db.write_aircraft_to_csv(aircraft)

    return aircraft

def sort_aircraft_list(aircraft_list):
    print("\n[Aircraft section] Sorting aircrafts by priority and amount of fuel... \n")
    aircraft_list.sort(key=lambda x: (x['priority'], x['fuel_percent']))


def prepare_runway_info():
    print('\n[Runway section]   Processing info about runways... \n')

    runways = db.get_json_info('../json_files/runways.json')

    g = rdflib.Graph()
    g.load("../knowledge_base.n3", format="n3")
    wind_info = []

    print('[Runway section]   Step 1: get weather prediction ')
    for row in g.query("SELECT ?s WHERE { mo:runway mp:needs ?s .}"):
        call_arg = split_res(row.s)
        act.call_action(call_arg, res = wind_info)

    print('[Runway section]   Step 2: validate runway based upon weather ')
    for runway in runways:
        for row in g.query("SELECT ?s WHERE { mo:runway mp:calls ?s .}"):
            call_arg = split_res(row.s)
            act.call_action(call_arg, runway, additional = wind_info)

    print("[Runway section]   Step 3: update runway db")
    db.set_json_info('../json_files/runways.json', runways)

    print('[Runway section]   Step 4: filtering available runways ')
    working_runways = []
    for runway in runways:
        if runway['state'] == 'available':
            working_runways.append(runway)

    return working_runways


def get_boarding_order(aircrafts, runways):
    print('\n\n[Order section]    Processing boarding order... \n')

    g = rdflib.Graph()
    g.load("../knowledge_base.n3", format="n3")

    order_list = []
    info = []
    info.append(aircrafts)
    info.append(runways)

    print("[Order section]    Step 1: call algorithm for setting order")
    for row in g.query("SELECT ?s WHERE { mo:order mp:calls ?s .}"):
        call_arg = split_res(row.s)
        act.call_action(call_arg, res = order_list, additional=info)

    return order_list

def run():
    # ---- Aircraft section ----
    input_data = db.get_json_info('../json_files/aircrafts.json')['aircraft']
    aircraft_list = []

    for plane in input_data:
        aircraft_list.append(prepare_aircraft_info(plane))
    sort_aircraft_list(aircraft_list)

    # ---- Runways section ----
    working_runways = prepare_runway_info()

    # ---- Order section ----
    if len(working_runways) == 0:
        print("\n\nOh, the weather outside is frightful...")
        print("\nNONE AVAILABLE RUNWAYS FOUND, TRY ANOTHER TIME.\n")
    else:
        order_list = get_boarding_order(aircraft_list, working_runways)
        print(order_list)
        print("SUCCESS. Writing to output file...")
        db.set_json_info('../json_files/output.json', order_list)
        print("DONE")

run()

