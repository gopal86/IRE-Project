import csv
from sklearn.model_selection import train_test_split


def read_file(filename):
    '''
        The dataset has 3 fields (ID, Text, Labels)

        Two of the fields(Text and Labels) are put into seperate files under the ../train folder.
        - ../train/comments.txt
        - ../train/labels.txt

        Further, the comments.txt will only contain the text from the current comment.
    '''

    global data
    data = list()

    labels = {"Other Hateful Sarcasm":0, "Racist Sarcasm":1, "Sexist Sarcasm":2,
            "None of the above (Neutral + Only sarcastic + Only hateful)":3}

    comments_writer = open("../train/comments.txt", "w")
    labels_writer = open("../train/labels.txt", "w")

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            text = row[1]
            comment = text.split("Comment:")[-1]
            label = labels[row[2]]
            data.append((comment, str(label)))
            comments_writer.write(comment + "\n")
            labels_writer.write(str(label) + "\n")

def split_data():
    global X_train, X_test, y_train, y_test, X_validation, y_validation
    X = [d[0] for d in data]
    y = [d[1] for d in data]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5)
    print(len(data))
    print(len(X_train), len(y_train))
    print(len(X_test), len(y_test))
    print(len(X_validation), len(y_validation))

    comments_writer_train = open("../train/comments.txt", "a")
    labels_writer_train = open("../train/labels.txt", "a")

    comments_writer_test = open("../test/comments.txt", "a")
    labels_writer_test = open("../test/labels.txt", "a")

    comments_writer_validation = open("../validation/comments.txt", "a")
    labels_writer_validation = open("../validation/labels.txt", "a")

    for i, comment in enumerate(X_train):
        comments_writer_train.write(comment + "\n")
        labels_writer_train.write(str(y_train[i]) + "\n")

    for i, comment in enumerate(X_test):
        comments_writer_test.write(comment + "\n")
        labels_writer_test.write(str(y_test[i]) + "\n")

    for i, comment in enumerate(X_validation):
        comments_writer_validation.write(comment + "\n")
        labels_writer_validation.write(str(y_validation[i]) + "\n")


if __name__ == "__main__":
    read_file("../dataset.csv")
    # split_data()
