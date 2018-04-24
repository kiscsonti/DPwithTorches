import json

def filter_needed(outfile, infile):
    counter = 0
    with open(outfile, "w"):
        pass
    with open(infile, "r") as f:
        for line in f.readlines():
            if counter % 250 == 0:
                print(counter)

            their_data =  json.loads(line)
            my_datas = dict()
            my_datas['p_words'] = their_data["d_words"]
            my_datas['q_words'] = their_data["q_words"]
            my_datas['c_words'] = their_data["c_words"]
            my_datas['label'] = their_data["label"]
            with open(outfile, "a") as out:
                json.dump(my_datas, out)
                out.write('\n')

            counter += 1


    pass

if __name__ == '__main__':

    filter_needed(outfile="data/my_processed_train.json",
                  infile="/media/kiscsonti/521493CD1493B289/git/live/commonsense-rc/data/train-data-processed.json")
    filter_needed(outfile="data/my_processed_dev.json",
                  infile="/media/kiscsonti/521493CD1493B289/git/live/commonsense-rc/data/dev-data-processed.json")
