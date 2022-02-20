= test_X.shape[0]
    num_train = train_X2.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
        for j in range(num_train):
            test_line = list(map(float,test_X[i]))
            train_line = list(map(float,train_X2[j]))
            temp = np.subtract(test_line, train_line)
            temp = np.power(temp, 2)
            dists[i][j] = np.sqrt(temp.sum())
    return dists