import os
import argparse

from utils import load_json

def json_to_metadata(json_path, metadata_path):

    result = []
    #json_path = os.path.(json_path)
    data = load_json(json_path)
    for key in data.keys():
        if len(data[key][0]) == 1:
            result.append(key + '|' + str(data[key]) +'|' + str(data[key]) )
        else:    
            result.append(key + '|' + data[key][0] +'|' + data[key][0] )

    with open(metadata_path, "w", encoding="utf-8") as f:
        for item in result:
            f.write("%s\n" % item)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recognition_path', required=True)
    parser.add_argument('--alignment_filename', default="alignment.json")
    parser.add_argument('--metadata_filename', default="metadata.csv")
    config, unparsed = parser.parse_known_args()

    base_dir = os.path.dirname(config.recognition_path)
    alignment_path = \
            os.path.join(base_dir, config.alignment_filename)
    
    metadata_path = \
            os.path.join(base_dir, config.metadata_filename)

    json_to_metadata(alignment_path, metadata_path)