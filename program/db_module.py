import json

def get_json_info(file): 
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def set_json_info(file, data):
    with open(file, 'w') as outfile:
        json.dump(data, outfile)

def write_aircraft_to_csv(craft):
    output_str = '\n{},{},{},{},{}'.format(
        craft['model'], craft['fuel_percent'], craft['priority'], craft['emergency'], craft['airtime'])
    with open('../csv_files/aircraft.csv', 'a') as f:
        f.write(output_str)  

if __name__ == "__main__":
    print('db_module.py file')