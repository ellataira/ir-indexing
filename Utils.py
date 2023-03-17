import pickle


class Utils:

    """
        saves a given dictionary to a pickle files
        @input filename / path to save to
        @input dictionary to be saved """
    def save_dict(self, filename, dict):
        file = open(filename, "wb")
        pickle.dump(dict, file)
        file.close()
        print("saved dictionary to " + filename + "\n")

    def read_pickle(self, filepath):
        infile = open(filepath, 'rb')
        dict = pickle.load(infile)
        infile.close()
        return dict

