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

def main():
    email = email_text()
    print(f"Текст письма: {email}")
    indexes = process_email(email)
    print(f"Индексы: {indexes}")
    features = email_features(indexes)
    print(f"Размер вектора признаков: {len(features)}")
    print(f"Размер не нулевых признаков: {sum(features > 0)}")


if __name__ == '__main__':
    main()