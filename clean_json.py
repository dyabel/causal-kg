import json
oil_dict = json.load(open('data/oil/gas.json','r'))
print(oil_dict)
for key,value in oil_dict.items():
    oil_dict[key] = value.replace('Expand','').replace('Collapse','').replace('\n','')
json.dump(oil_dict, open('data/oil/gas_clean.json','w'))