# bin bash stuff
# tic tac toe


import os
import random

from sklearn.ensemble import GradientBoostingClassifier

os.system('cls' if os.name == 'nt' else 'clear')
#namespace TicBrain
#class ToeModel
#class ToeGame
#class ToeView
#class ToeController
#class ToeBrain
#class InputBrain
asc = {
    "X": [' \   / ',
          '   x   ',
          ' /   \ '],
    "O": ['  /-\  ',
          ' ( O ) ',
          '  \-/  '],
    " ": ['       ',
          '       ',
          '       ']
}

board = [" ", " ", " ",
         " ", " ", " ",
         " ", " ", " "]

players = ('X', 'O')

wins = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)]


def drawBoard(xo):
    '''
    given a list with 9 characters, this draws them to the screen using
    '''
    os.system('cls' if os.name == 'nt' else 'clear')
    #print("\n\n")
    print(asc[xo[0]][0], asc[xo[1]][0], asc[xo[2]][0], sep='|')
    print(asc[xo[0]][1], asc[xo[1]][1], asc[xo[2]][1], sep='|')
    print(asc[xo[0]][2], asc[xo[1]][2], asc[xo[2]][2], sep='|')
    print('-'*23)
    print(asc[xo[3]][0], asc[xo[4]][0], asc[xo[5]][0], sep='|')
    print(asc[xo[3]][1], asc[xo[4]][1], asc[xo[5]][1], sep='|')
    print(asc[xo[3]][2], asc[xo[4]][2], asc[xo[5]][2], sep='|')
    print('-'*23)
    print(asc[xo[6]][0], asc[xo[7]][0], asc[xo[8]][0], sep='|')
    print(asc[xo[6]][1], asc[xo[7]][1], asc[xo[8]][1], sep='|')
    print(asc[xo[6]][2], asc[xo[7]][2], asc[xo[8]][2], sep='|')


def play(letter, pos1to9, updateScreen=True):
    if (letter in players) and board[pos1to9-1] == ' ':
        board[pos1to9-1] = letter
    else:
        return "BadInput"
    if updateScreen:
        drawBoard(board)
    if testWin(letter, board):
        return letter
    if ' ' in board:
        return ''
    return 'Draw'


def resetBoard():
    for i in range(9):
        board[i] = " "


def testWin(letter, xo):
    for w in wins:
        if "".join([xo[i] for i in w]) == letter*3:
            return True
    return False


drawBoard(board)
player = 1
outcome = ''


# while outcome == '':
#     player = -player + 1
#     choice = int(input())
#     outcome = play(players[player], choice)

# if outcome != 'Draw':
#     outcome = outcome + ' wins!'

# print(outcome)

# our goal is given an empty or partially filled board of X's and O's
# represented by a list of 9 elements like [0,1,2,
#                                           2,0,1,
#                                           1,0,2]
# it will pick a number 1-9 ( 9 classes) that maximizes a win in fewest moves


# we will generate a bunch of valid games as a sequence.
# whenever x wins, we will construct a series of "examples of how I did it"
# where for each step that I did, I will label that board as choosing the class
# (1-9) of the next move that I ended up doing.  So ideally, when it sees that
# position again, it will be trained to choose the number that it saw in the
# example. It will end up "overfitting" by having a preference for certain
# moves, but it should have the right answer for the final winning moves
# Note, while I could systematically generate all winning games, I'll go with
# just a random sample of n randomly chosen winning games

# data looks like [[0,1,2,2,0,1,1,0,2],...] and label will be like [2,9,5....]

def boardFromSequence(sequence):
    newBoard = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(sequence)):
        newBoard[sequence[i] - 1] = (i % 2) + 1
    return newBoard


def getValidChoice(sequence):
    return random.choice([x + 1 for x in range(9) if (x + 1) not in sequence])


def getRandomWinSequence(winner, show=True, text=""):
    players = ('X', 'O')
    outcome = ''
    while outcome != winner:
        resetBoard()
        sequence = []
        player = 1
        outcome = ''
        while outcome == '':
            player = -player + 1
            sequence.append(getValidChoice(sequence))
            outcome = play(players[player], sequence[-1], show)
    if show is True:
        print(text)
    return sequence


def getWinSequences(player, n, showStep=1):
    return [getRandomWinSequence(player, (x % showStep) == 0,
            "progress: " + str(round((x+1)/n*100)) + "%") for x in range(n)]


def samplesFromWin(winsequence):
    ''' Take a sequence of wins, and for each step of the way,
        add an observation/sample consisting of the board as
        it looked just before the next step, and the label is
        the next step (1-9). So "when you see this board,
        predict the next step"
    '''
    return [boardFromSequence(winsequence[:n])+[winsequence[n]] for n
            in range(0 if len(winsequence) % 2 else 1, len(winsequence), 2)]


def samplesFromWins(winsequences):
    samples = []
    for sequence in winsequences:
        for sample in samplesFromWin(sequence):
            samples.append(sample)
    return samples

print("generating training data")

winsX = getWinSequences('X', 100000, 1000)
samplesX = samplesFromWins(winsX)

#winsO = getWinSequences('O', 5000, 1000)
#samplesO = samplesFromWins(winsO)

X = [row[:9] for row in samplesX]
y = [row[9] for row in samplesX]

#print(str(y))

X_train, X_test = X[:70000], X[70000:]
y_train, y_test = y[:70000], y[70000:]

print("got data, ready to train")

clf = GradientBoostingClassifier(n_estimators=800, learning_rate=.1,
                                 loss='deviance', max_depth=1,
                                 random_state=0, verbose=2
                                 ).fit(X_train, y_train)

print("done training.  Calculating score on test set:")
print(clf.score(X_test, y_test))

#clf.classes_ = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(zip(clf.classes_, clf.predict_proba([[2,2,0,
                                            1,1,0,
                                            0,0,0]])[0]))
#print(repr(boardFromSequence(winsX[0])))
#[print(repr(x)) for x in samplesX]


#def generateGamesData():
#    for n in range(10):
#        sample = getRandomGame

input()
