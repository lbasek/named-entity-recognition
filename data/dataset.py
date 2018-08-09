import pandas as pd
import os
from tqdm import tqdm


class Dataset:
    def __init__(self, filename, output_filename, debug=False):
        self.filename = filename
        self.debug = debug
        self.output_filename = output_filename
        self.data_frame = pd.DataFrame(columns=['Sentence #', 'Word', 'POS', 'Chunk', 'Tag'])

    def create_csv(self):

        if not os.path.isdir("./csv") or not os.path.exists(self.filename):
            raise ValueError("Filename or csv dir is wrong.")

        print("-------Creating dataset-------")

        f = open(self.filename, "r+")

        non_allow_chars = ['"', ',', '-X-']

        iterator = 1
        for line in tqdm(f.readlines()):
            row = self.__process_line(line)

            if not (row is None or row[0] in non_allow_chars or row[1] in non_allow_chars):
                word = row[0]
                pos = row[1]
                chunk = row[2]
                tag = row[3]

                self.data_frame.loc[iterator] = [iterator, word, pos, chunk, tag]
                iterator += 1
                if self.debug:
                    print(
                        '#{}, \t Word:{}, \t POS:{}, \t Chunk:{}, \t Tag:{}'.format(str(iterator), word, pos, chunk,
                                                                                    tag))

        self.data_frame.to_csv(self.output_filename + '.csv', encoding='utf-8')

        print("Done! File: " + self.output_filename + ".csv")

    @staticmethod
    def __process_line(line):
        parts = line.strip().split()
        if len(parts) == 4:
            return parts


if __name__ == '__main__':
    dataset = Dataset("./raw/test.txt", "./csv/test")
    dataset.create_csv()
