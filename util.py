import os, glob, pathlib
import pandas as pd

def readJSONFiles(dir, folder):
    dataframe = pd.DataFrame()
    path = pathlib.Path('.').resolve()/dir/folder
    files = glob.glob(os.path.join(path, "*.json.gz"))
    for file in files:
        otherdf = pd.read_json(file, lines=True)
        dataframe = dataframe._append(otherdf)
    
    return dataframe