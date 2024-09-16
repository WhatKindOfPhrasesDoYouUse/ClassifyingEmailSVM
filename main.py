import numpy as np
import scipy.io
from sklearn import svm

from process_email import process_email
from process_email import email_features
from process_email import get_dictionary

def email_text():
    email = ''
    with open('email.txt', 'r') as file:
        email = file.read().replace('\n', '')
    return email

def classifier_learning():
    data = scipy.io.loadmat('train.mat')
    X = data['X']
    y = data['y'].flatten()
    print("Тренировка SVM - классификатора с линейным ядром...")
    clf = svm.SVC(C=0.1, kernel='linear', tol=1e-3)
    model = clf.fit(X, y)
    p = model.predict(X)
    #return np.sum(p == y)
    return np.mean(p == y) * 100

def classifier_learning_test_sample():
    data = scipy.io.loadmat('test.mat')
    X = data['Xtest']
    y = data['ytest'].flatten()
    print("Тренировка SVM - классификатора с линейным ядром на тестовой выборке...")
    clf = svm.SVC(C=0.1, kernel='linear', tol=1e-3)
    model = clf.fit(X, y)
    p = model.predict(X)
    # return np.sum(p == y)
    return np.mean(p == y) * 100

def main():
    #email = email_text()
    #print(f"Текст письма: {email}")
    #indexes = process_email(email)
    #print(f"Индексы: {indexes}")
    #features = email_features(indexes)
    #print(f"Размер вектора признаков: {len(features)}")
    #print(f"Размер не нулевых признаков: {sum(features > 0)}")
    #accuracy = classifier_learning()
    #print(f"Точность на обучающей выборке: {accuracy}")
    test_accuracy = classifier_learning_test_sample()
    print(f"Точность на тестовой выборке: {test_accuracy}")


if __name__ == '__main__':
    main()