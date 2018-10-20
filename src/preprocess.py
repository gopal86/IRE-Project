import csv


def read_file(filename):
    ''' 
        The dataset has 3 fields (ID, Text, Labels)
        
        Two of the fields(Text and Labels) are put into seperate files under the ../train folder.
        - ../train/comments.txt
        - ../train/labels.txt

        Further, the comments.txt will only contain the text from the current comment.
    '''

    labels = {"Other Hateful Sarcasm":0, "Racist Sarcasm":1, "Sexist Sarcasm":2, 
            "None of the above (Neutral + Only sarcastic + Only hateful)":3}
    
    comments_writer = open("../train/comments.txt", "a")
    labels_writer = open("../train/labels.txt", "a")

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            text = row[1]
            comment = text.split("Comment:")[-1]
            label = labels[row[2]]
            comments_writer.write(comment + "\n")
            labels_writer.write(str(label) + "\n")

if __name__ == "__main__":
    read_file("../dataset.csv")
