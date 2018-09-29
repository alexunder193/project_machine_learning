import itertools

def lcs(a, b):
    lengths = [[0 for j in range(len(b)+1)] for i in range(len(a)+1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            #if x == y:
            if haversine(x,y)<=0.2:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
    # read the substring out from the matrix
    result = []
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x-1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y-1]:
            y -= 1
        else:
            assert haversine(a[x-1],b[y-1])<=0.2
            result .append(a[x-1] )
            x -= 1
            y -= 1
    return result ,len(result)

def count_common_subsequences(seq_1, seq_2):
    """Find the number of common subsequences between two collections.

    This function finds the number of common subsequences between two
    collections but does not return an actual listing of those subsequences.
    This is more space efficient O(len(seq_1)) than find_common_subsequences.

    > number_of_subsequences = count_common_subsequences('qwer', 'qewr')
    > print number_of_subsequences
    12

    @param seq_1: The first collection to find subsequences in.
    @type seq_1: Any integer indexable collection (list, tuple, etc.)
    @param seq_2: The second collection to find subsequences in.
    @type seq_2: Any integer indexable collection (list, tuple, etc.)
    @return: The number of common subsequences between seq_1 and seq_2.
    @rtype: int
    """
    # Ensure the smaller of the two sequences is used to create the columns for
    # the DP table.
    if len(seq_1) < len(seq_2):
        new_seq_1 = seq_2
        seq_2 = seq_1
        seq_1 = new_seq_1

    # Use length plus one to provide a row and column in the subsequence table,
    # a row / column not corresponding to an element. This provides
    # initialization values to the algorithm and handles the edge case of
    # calculating an element in the subsequence table when that element is
    # either in the first row of the table or is the first element of a row.
    # This also includes / handles the empty string as a substring.
    seq_1_len = len(seq_1)
    seq_2_len = len(seq_2)
    seq_1_len_plus_1 = seq_1_len + 1
    seq_2_len_plus_1 = seq_2_len + 1
    
    # Initialize the first two rows of a "2D array" consisting of
    # seq_1_len_plus_1 by seq_2_len_plus_1 values. Note that, due to space
    # optimizations, only two rows are ever maintained in memory.
    subseq_last_row = [1] * seq_2_len_plus_1
    subseq_current_row = [1] + [0] * seq_2_len

    for row in xrange(1, seq_1_len_plus_1):

        for col in xrange(1, seq_2_len_plus_1):

            #if seq_1[row-1] == seq_2[col-1]:
            #print(haversine(seq_1[row-1],seq_2[col-1]))
            #print(seq_1[row-1],seq_2[col-1])
            if haversine(seq_1[row-1],seq_2[col-1])<=0.2: 
                new_cell_value = 2 * subseq_last_row[col - 1]
            else:
                new_cell_value = subseq_last_row[col]
                new_cell_value += subseq_current_row[col-1]
                new_cell_value -= subseq_last_row[col-1]
            subseq_current_row[col] = new_cell_value

        subseq_last_row = subseq_current_row
        subseq_current_row = [1] + [0] * seq_2_len

    return subseq_last_row[seq_2_len]


def add_matched_element(element, target_set, sep):
    """Append an element to the end of all elements in a set.

    Creates a new copy of target_set with an element appended to the end of all
    items in that target_set. Returns a union between the target_set and the
    newly created copy of target_set.

    @param element: The element to add.
    @param target_set: Collection of items to append element to.
    @type target_set: set
    @return: The union between a copy of target_set with element appended to
        all of its items and the original target_set.
    @rtype: set
    """
    new_elements = map(lambda x: x + sep + str(element), target_set)
    return target_set.union(new_elements)


def find_common_subsequences(seq_1, seq_2, sep='', empty_val=''):
    """Find the number of common subsequences between two collections.

    This function finds the common subsequences between two collections and
    returns an actual listing of those subsequences. This is less space
    efficient (O(len(seq_1)^2)) than count_common_subsequences.

    > subsequences = find_common_subsequences('qwer', 'qewr')
    > print subsequences
    set(['', 'qer', 'wr', 'qwr', 'er', 'qr', 'e', 'qw', 'q', 'r', 'qe', 'w'])

    @param seq_1: The first collection to find subsequences in.
    @type seq_1: Any integer indexable collection (list, tuple, etc.)
    @param seq_2: The second collection to find subsequences in.
    @type seq_2: Any integer indexable collection (list, tuple, etc.)
    @keyword sep: Seperator to put between elements when constructing a
        subsequence. Defaults to ''.
    @keyword empty_val: The value to use to represent the empty set.
    @return: Set of subsequences in common between seq_1 and seq_2.
    @rtype: set
    """
    # Ensure the smaller of the two sequences is used to create the columns for
    # the DP table.
    if len(seq_1) < len(seq_2):
        new_seq_1 = seq_2
        seq_2 = seq_1
        seq_1 = new_seq_1

    # Use length plus one to provide a row and column in the subsequence table,
    # a row / column not corresponding to an element. This provides
    # initialization values to the algorithm and handles the edge case of
    # calculating an element in the subsequence table when that element is
    # either in the first row of the table or is the first element of a row.
    # This also includes / handles the empty string as a substring.
    seq_1_len = len(seq_1)
    seq_2_len = len(seq_2)
    seq_1_len_plus_1 = seq_1_len + 1
    seq_2_len_plus_1 = seq_2_len + 1
    
    # Initialize the first two rows of a "2D array" consisting of
    # seq_1_len_plus_1 by seq_2_len_plus_1 values. Note that, due to space
    # optimizations, only two rows are ever maintained in memory.
    subseq_last_row = [set([empty_val])] * seq_2_len_plus_1
    subseq_current_row = [set([empty_val])] + [set()] * seq_2_len

    for row in xrange(1, seq_1_len_plus_1):

        for col in xrange(1, seq_2_len_plus_1):

            #if seq_1[row-1] == seq_2[col-1]:
            if haversine(seq_1[row-1],seq_2[col-1])<=0.2:
                diagonal_cell_value = subseq_last_row[col - 1]
                matched_element = seq_1[row-1]
                new_cell_value = add_matched_element(matched_element,
                    diagonal_cell_value, sep)
            else:
                above_set = subseq_last_row[col]
                left_set = subseq_current_row[col-1]
                new_cell_value = above_set.union(left_set)
            subseq_current_row[col] = new_cell_value

        subseq_last_row = subseq_current_row
        subseq_current_row = [set([empty_val])] + [set()] * seq_2_len

    return subseq_last_row[seq_2_len]



import pandas as pd
import gmplot
from ast import literal_eval
from haversine import haversine
import numpy as np
import matplotlib.pyplot as plt
#test_seq_1 = 'alex'
#test_seq_2 = 'alex'
#print(py_common_subseq.count_common_subsequences(test_seq_1, test_seq_2))
#print(py_common_subseq.find_common_subsequences(test_seq_1, test_seq_2))

testSet = pd.read_csv('test_set_a2.csv',converters={"Trajectory": literal_eval},sep="\t")
trainSet = pd.read_csv('train_set.csv',converters={"Trajectory": literal_eval},index_col='tripId')
trainSet=trainSet[0:100]
testSet=testSet
for test in testSet['Trajectory']:
    print("ANOTHER TEST")
    testobject=[]
    for x in test: 
        data=(x[1],x[2])
        testobject.append(data)
    for i in range(0, len(trainSet)):	#gia kathe trajectory tou train
        trainobject=[]
        train=trainSet.iloc[i]['Trajectory']
        journey=trainSet.iloc[i]['journeyPatternId']
        for y in train:
            data=(y[1],y[2])
            trainobject.append(data)
        path,length=lcs(testobject,trainobject)
        print(path)
        print(length)
        print("-------------------------")
        #print(count_common_subsequences(testobject,trainobject))
        #print(find_common_subsequences(testobject,trainobject))

#testobject=[(3.14,3.24),(5,6),(7,8),(8,9),(1.2,5.4),(4.3,5.6)]
#trainobject=[(2,3),(3.14,3.24),(7,8),(1,2)]
#print(count_common_subsequences(testobject,trainobject))
#print(find_common_subsequences(testobject,trainobject))