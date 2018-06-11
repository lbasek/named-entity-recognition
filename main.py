import dataset

print("Hello!")

dataset = dataset.DatasetReader("./dataset/eng.train")
sentences = dataset.process_words()

print("stop")