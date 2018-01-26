import nflgame
import nflgame.update_sched
import operator
import numpy as np
import statsmodels.api as sm
import matplotlib
import pandas
import itertools
from sklearn import linear_model
from scipy import stats
from matplotlib import pyplot as plt

##get all games over a given time period
##assign each one a unique identifier and post it to a dictionary
def create_game_dictionary():
    ##declare dictionary of all games
    all_game_dict = {}

    ##list of all nfl teams
    ##obviously needs to be expanded
    all_teams = ["DET", "CAR"]

    
    games = nflgame.games([2016,2017])
    i = 0
    for item in games:
        game_id = str(item.home) + str(item.away) + str(item.stats_home) + str(item.stats_away)
        all_game_dict[game_id] = item
    last_game = all_game_dict[game_id]
    print last_game.home
    
        
    

##take home and away arrays and combine into one
##adds all items in away list to the end of the home list
def home_away_combine(home_list, away_list):
    combination = []
    for item in home_list:
        combination.append(item)
    for item in away_list:
        combination.append(item)
    return combination

##new analysis on shit
def new_tester(games):
    ## p = protagonist
    ## a = antagonist
    p_first_downs = []
    p_passing_yds = []
    p_rushing_yds = []
    p_penalty_yds = []
    a_penalty_yds = []
    p_punt_cnt = []
    a_punt_cnt = []
    p_turnovers = []
    a_turnovers = []
    p_home = []
    p_points = []
    p_comp_pct = []
    for item in games:
        home_pass_att =  0
        home_pass_comp = 0
        for p in item.players.passing():
            if p.team == item.home:
                home_pass_att = home_pass_att + p.passing_att
                home_pass_comp = home_pass_comp + p.passing_cmp
        p_comp_pct.append((float(home_pass_comp)/float(home_pass_att)))
        p_first_downs.append(item.stats_home.first_downs)
        p_passing_yds.append(item.stats_home.passing_yds)
        p_rushing_yds.append(item.stats_home.rushing_yds)
        p_penalty_yds.append(item.stats_home.penalty_yds)
        a_penalty_yds.append(item.stats_away.penalty_yds)
        p_punt_cnt.append(item.stats_home.punt_cnt)
        a_punt_cnt.append(item.stats_away.punt_cnt)
        p_turnovers.append(item.stats_home.turnovers)
        a_turnovers.append(item.stats_away.turnovers)
        p_home.append(1)
        p_points.append(item.score_home)
        ##same, but for away team
        away_pass_att =  0
        away_pass_comp = 0
        for p in item.players.passing():
            if p.team == item.away:
                away_pass_att = away_pass_att + p.passing_att
                away_pass_comp = away_pass_comp + p.passing_cmp
        p_comp_pct.append((float(away_pass_comp)/float(away_pass_att)))
        p_first_downs.append(item.stats_away.first_downs)
        p_passing_yds.append(item.stats_away.passing_yds)
        p_rushing_yds.append(item.stats_away.rushing_yds)
        p_penalty_yds.append(item.stats_away.penalty_yds)
        a_penalty_yds.append(item.stats_home.penalty_yds)
        p_punt_cnt.append(item.stats_away.punt_cnt)
        a_punt_cnt.append(item.stats_home.punt_cnt)
        p_turnovers.append(item.stats_away.turnovers)
        a_turnovers.append(item.stats_home.turnovers)
        p_home.append(0)
        p_points.append(item.score_away)
    return p_first_downs, p_passing_yds, p_rushing_yds, p_penalty_yds, a_penalty_yds, p_punt_cnt, a_punt_cnt, p_turnovers, a_turnovers, p_home, p_comp_pct, p_points       
def main(): 
    ##make list of 17 weeks in season
    weekslist = []
    i = 0
    while i <=17:
        weekslist.append(i)
        i += 1
    ##assign all games from 2015 to games
    games = nflgame.games(2017, None, "DET", "DET", "REG")
    for item in games:
        print item
    games2 = nflgame.games(2016, weekslist)

    nFirstDowns, nPassingYards, nRushingYards, nPenaltyYards, nOpponentPenaltyYards, nPuntCount, nOpponentPuntCount, nTurnovers, nOpponentTurnovers, nHomeAway, nCompPct, nPoints = new_tester(games)
    tFirstDowns, tPassingYards, tRushingYards, tPenaltyYards, tOpponentPenaltyYards, tPuntCount, tOpponentPuntCount, tTurnovers, tOpponentTurnovers, tHomeAway, tCompPct, tPoints = new_tester(games2)
        
    ##stack all criteria to test, add a b constant to the equation, print info about the resulting linear regression funciton
    newX = np.column_stack((nFirstDowns, nPassingYards, nRushingYards, nPenaltyYards, nOpponentPenaltyYards, nPuntCount, nOpponentPuntCount, nTurnovers, nOpponentTurnovers, nHomeAway))
    new2 = np.column_stack((tFirstDowns, tPassingYards, tRushingYards, tPenaltyYards, tOpponentPenaltyYards, tPuntCount, tOpponentPuntCount, tTurnovers, tOpponentTurnovers, tHomeAway, tCompPct))
    newX = sm.add_constant(newX)
    new2 = sm.add_constant(new2)
    newR = sm.OLS(nPoints, newX).fit()
    print newR.rsquared
    print newR.summary()
    print newR.params


    ##display a graph of the predicted score of the game vs. the actual score
    ##the closer the dots are to the center line, the better the model fits the historical data


    ##take in the stacked X from the regression analysis and the equation factors about the regression
    ##send back the predicted score given the factors from that game
    def predict_score(stats_in, equation_factors):
        print "predicting scores"
        predicted_scores = []
        for item in stats_in:
            i = 1
            current_score_prediction = item[0]
            while i < (len(item)):
                current_score_prediction += item[i] * equation_factors[i]
                i += 1
            predicted_scores.append(current_score_prediction)
        return predicted_scores



    equation_factors = newR.params
    predicted_score = predict_score(newX, equation_factors)
    linex = []
    liney = []
    i = 0
    while i < 50:
        linex.append(i)
        liney.append(i)
        i = i+1
    differential = []
    raw_differential = []
    i = 0
    while i < len(predicted_score):
        differential.append(abs(predicted_score[i] - nPoints[i]))
        i += 1    
    print "Average: +/- " + str(sum(differential)/len(differential))
    differential = sorted(differential)
    print "Median: +/- " + str(differential[(len(differential)/2)])

    plt.scatter(predicted_score, nPoints)
    plt.plot(linex, 'b-')
    plt.xlabel("Predicted Score")
    plt.ylabel("Actual Score")
    plt.show()
        


    ##plt.scatter(adjusted_predicted_score, nPoints)
    ##plt.plot(linex, 'b-')
    ##plt.xlabel("Adjusted Predicted Score")
    ##plt.ylabel("Actual Score")
    ##plt.show()
    ##plt.savefig("plot.png")

create_game_dictionary()







