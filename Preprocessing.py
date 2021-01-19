import chess.pgn
import pandas as pd 
import numpy as np 
import pgn
import sys
import pickle
import time 

#have data for 50k chess games in pgn format, want to extract white elo, black elo, result, evals, opening and variation to pandas df
ts = time.time()

pgns = open('new_data.pgn')
f = open('new_data.pgn')
pgn_text = f.read()
f.close()


#extract move lists using pgn library 
games = pgn.loads(pgn_text)
moves = [game.moves for game in games]

#extract other info using chess.pgn library

pgns = open('new_data.pgn')
openings = []
variations = []
wElo = []
bElo = []
results = []

#note the 'try, except' clauses in place because only half the data has elo data (train vs test set), and some games have no variation header


for i in range(50000):
    game = chess.pgn.read_game(pgns)
    openings.append(game.headers['Opening'])
    try:
        variations.append(game.headers['Variation'])
    except KeyError:
        variations.append('None')
    try:
        wElo.append(game.headers['WhiteElo'])
    except KeyError:
        wElo.append('None')
        
    try:
        bElo.append(game.headers['BlackElo'])
    except KeyError:
        bElo.append('None')
    results.append(game.headers['Result'])
    if i%1000 == 0:
        print('step: ',i)



data = pd.concat([pd.Series(wElo),pd.Series(bElo),pd.Series(results)], axis = 1)

data.columns = ['white','black','result']

#convert the results to float for classification
for i in range(len(data['result'])):
    if data['result'][i] == '1/2-1/2':
        data['result'][i] = 1/2
    elif data['result'][i] == '1-0':
        data['result'][i] = 1
    elif data['result'][i] == '0-1':
        data['result'][i] = 0

#evals come nicely organised, but in strings
evals = pd.read_csv('stockfish.csv')

evals.columns = ['Event','eval']

evalist = [i.split(' ') for i in evals['eval']]


#might try to do this with map(float, etc) but fails on 'NA', probably a way around this without making this function

def floatify(x):
    floats = []
    for i in x:
        mini = []
        for j in i:
            if j != 'NA' and j != '':
                mini.append(float(j))
        floats.append(mini)
    return floats


floats = floatify(evalist)


data1 = pd.concat([data,pd.Series(floats),pd.Series(openings),pd.Series(variations), pd.Series(moves)], axis = 1)
data1.columns = ['white','black','result', 'evals', 'opening', 'variation', 'moves']



            
data1.to_pickle('chess_data.pkl')         

print(time.time()- ts)
