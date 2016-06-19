from sklearn.ensemble import RandomForestClassifier
import numpy as np

def main():
    #Create training and test sets, and skip the header row ([1:])in CSV.
    dataset = np.genfromtxt(open('data/train.csv', 'r'), delimiter=',', dtype='f8')[1:]
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    test = np.genfromtxt(open('data/test.csv', 'r'), delimiter=',', dtype='f8')[1:]

    forest = trainForest(train, target)

    #save forest

    classify = forest.predict(test).astype(int)

    result = np.zeros((len(classify), 2))
    for index, prediction in enumerate(classify):
        print(index)
        result[index] = [(index + 1), prediction]

    np.savetxt('output/forest.csv', result, delimiter=',', fmt='%i')


def trainForest(train, target):
    rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    rf.fit(train, target)
    return rf

if __name__ == '__main__':
    main()
