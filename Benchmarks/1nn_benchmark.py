from sklearn import neighbors

def read_data(file_name):
    f = open(file_name)
    # Ignore the header
    f.readline()
    samples = []
    target = []
    for line in f:
        line = line.strip().split(",")
        sample = [float(x) for x in line[1:]]
        samples.append(sample)
        target.append(line[0])
    return (samples,target)

def write_delimited_file(file_path, data,header=None, delimiter=","):
    f_out = open(file_path,"w")
    if header is not None:
        f_out.write(delimiter.join(header) + "\n")
    for line in data:
        f_out.write(delimiter.join(line) + "\n")
    f_out.close()

def main():
    training, target = read_data("../Data/features/features_train.csv")
    target = [float(x[:x.index("_")]) for x in target]
    test, file_numbers = read_data("../Data/features/features_test.csv")

    knn = neighbors.NeighborsClassifier(n_neighbors = 1)
    knn.fit(training, target)
    predictions = ["%d" % x for x in knn.predict(test)]

    write_delimited_file("../Submissions/1nn_benchmark.csv",
                         zip(file_numbers,predictions),
                         ["image_ID","writer"])

if __name__=="__main__":
    main()
