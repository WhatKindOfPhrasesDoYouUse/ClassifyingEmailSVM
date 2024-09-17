import numpy as np
import scipy.io
from sklearn import svm

from process_email import process_email
from process_email import email_features
from process_email import get_dictionary

from collections import OrderedDict

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
    print(f"Точность на обучающей выборке: {np.mean(p == y) * 100}")
    return model

def classifier_learning_test_sample():
    data = scipy.io.loadmat('test.mat')
    X = data['Xtest']
    y = data['ytest'].flatten()
    print("Тренировка SVM - классификатора с линейным ядром на тестовой выборке...")
    clf = svm.SVC(C=0.1, kernel='linear', tol=1e-3)
    model = clf.fit(X, y)
    p = model.predict(X)
    # return np.sum(p == y)
    #return np.mean(p == y) * 100
    print(f"Точность на тестовой выборке: {np.mean(p == y) * 100}")
    return model

def top_spam_words(model):
    t = sorted(list(enumerate(model.coef_[0])), key=lambda e: e[1], reverse=True)
    d = OrderedDict(t)
    idx = list(d.keys())
    weight = list(d.values())
    dictionary = get_dictionary()
    print('Топ-15 слов в письмах со спамом: ')
    for i in range(15):
        print(' %-15s (%f)' % (dictionary[idx[i]], weight[i]))

def main():
    email = email_text()
    print(f"Текст письма: {email}")
    indexes = process_email(email)
    print(f"Индексы: {indexes}")
    features = email_features(indexes)
    print(f"Размер вектора признаков: {len(features)}")
    print(f"Размер не нулевых признаков: {sum(features > 0)}")
    model1 = classifier_learning()
    #print(f"Точность на обучающей выборке: {accuracy}")
    model2 = classifier_learning_test_sample()
    #print(f"Точность на тестовой выборке: {test_accuracy}")
    top_spam_words(model1)
    top_spam_words(model2)

if __name__ == '__main__':
    main()