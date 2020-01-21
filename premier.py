import pandas as pd
import numpy as np

#loading historical matches
league = pd.read_csv("england.csv")
league = league[league.Season >= 1990].reset_index(drop = True)
league = league[["Date", "Season", "home", "visitor", "hgoal", "vgoal"]]


facup = pd.read_csv("facup.csv")
facup = facup[facup.Season >= 1990].reset_index(drop = True)
facup = facup[["Date", "Season", "home", "visitor", "hgoal", "vgoal"]]


leaguecup = pd.read_csv("leaguecup.csv")
leaguecup = leaguecup[leaguecup.Season >= 1990].reset_index(drop = True)
leaguecup = leaguecup[["Date", "Season", "home", "visitor", "hgoal", "vgoal"]]


playoffs = pd.read_csv("englandplayoffs.csv")
playoffs = playoffs[playoffs.Season >= 1990].reset_index(drop = True)
playoffs = playoffs[["Date", "Season", "home", "visitor", "hgoal", "vgoal"]]


league18 = pd.read_csv("league 2018.csv")
only_date = lambda x: x[0:10]
league18["Date"] = league18.Date.apply(only_date)
league18["Date"] =pd.to_datetime(league18.Date, format='%d/%m/%Y')
home_score = lambda x: x[0]
visitor_score = lambda x: x[4]
league18["hgoal"] = league18.Result.apply(home_score)
league18["vgoal"] = league18.Result.apply(visitor_score)
league18.drop(columns=['Result'],inplace = True)


#putting all historical matches in one dataframe
match_records = pd.concat([league,facup,leaguecup,playoffs,league18]).reset_index(drop = True)
match_records["Date"] =pd.to_datetime(match_records.Date)
match_records = match_records.sort_values(by="Date").reset_index(drop = True)
match_records.dropna(how="all", inplace=True)
match_records['home_result'] = match_records.apply(lambda row: "win" if row['hgoal'] > row['vgoal'] else ("draw" if row['hgoal'] == row['vgoal'] else "loss"), axis = 1)
match_records = match_records[["Date", "Season", "home", "visitor", "home_result"]]



#calculating historic win/draw/lose percentages
def historic_percentages(row):
    number_of_matches = 10
    matches = match_records[(match_records.home == row.home) & (match_records.visitor == row.visitor) & (match_records.Season < row.Season)].tail(number_of_matches)
    number_of_matches = matches["home"].count()
    if number_of_matches == 0:
        return 1/3,1/3,1/3
    count_series = matches.home_result.value_counts()
    indices = count_series.index.tolist()
    counts = count_series.to_frame().transpose()
    if "win" in indices:
        win = int(counts["win"])/number_of_matches
    else:
        win = 0
    if "draw" in indices:
        draw = int(counts["draw"])/number_of_matches
    else:
        draw = 0
    if "loss" in indices:
        loss = int(counts["loss"])/number_of_matches
    else:
        loss = 0
    return win, draw, loss




fixtures = pd.read_csv("Fixtures_16_to_19.csv")
fixtures["win%"], fixtures["draw%"], fixtures["loss%"] =  zip(*fixtures.apply(historic_percentages, axis = 1))

#adding historical percentages to main data
df = pd.read_csv("Spi.csv")
df = pd.merge(df,fixtures, how="left")
df["Date"] =pd.to_datetime(df.Date)
match_records = match_records.drop_duplicates()
df = pd.merge(df,match_records, how="left")


#projecting the rest of 2019 league positions based on history only
def league_points_calculator(row):
    if row['home'] == i:
        return 3*row["win%"] + 1*row["draw%"]
    else:
        return 3*row["loss%"] + 1*row["draw%"]

df_2019 = df[df.Season == 2019]
teams_2019 = df_2019.home.unique()

tables = []
for i in teams_2019:
    cursor = df[((df.Season == 2019) & (df.Round > 16)) & ((df.home == i)|(df.visitor == i))][["Date", "Season", "Round", "home", "visitor", "win%", "draw%", "loss%"]]
    cursor['points'] = cursor.apply(league_points_calculator, axis = 1)
    tables.append((cursor[["Round","points"]].rename(columns={"points": i}).set_index('Round').T))

cursor = pd.concat(tables).reset_index()
z =pd.read_csv("2019_round16_points.csv").set_index('index').reset_index()
rest_of_2019 = pd.merge(z,cursor).rename(columns={"16": 16})

#making a per round league table (points) of the rest of 2019 based on history only
def points_adder(table):
    for i in range(17,39):
        table[i] = table[i] + table[i-1]
    return table

rest_of_2019 = points_adder(rest_of_2019)

#making a table of per round league table (positions) of the rest of 2019 based on history only
for i in range(16,39):
    rest_of_2019 = rest_of_2019.sort_values(by=[i],ascending=False)
    rest_of_2019[i] = list(range(1,21))


#adding home team legue position and visitor team league position to every match in the main dataframe
pos_2016 = pd.read_csv("2016_league_positions.csv")
def home_pos_2016(row):
    if row.Season == 2016:
        return float(pos_2016[str(row.Round)][pos_2016["index"] == row.home])
def visitor_pos_2016(row):
    if row.Season == 2016:
        return float(pos_2016[str(row.Round)][pos_2016["index"] == row.visitor])


pos_2017 = pd.read_csv("2017_league_positions.csv")
def home_pos_2017(row):
    if row.Season == 2017:
        return float(pos_2017[str(row.Round)][pos_2017["index"] == row.home])
    else:
        return float(row.home_pos)
def visitor_pos_2017(row):
    if row.Season == 2017:
        return float(pos_2017[str(row.Round)][pos_2017["index"] == row.visitor])
    else:
        return float(row.visitor_pos)

pos_2018 = pd.read_csv("2018_league_positions.csv")
def home_pos_2018(row):
    if row.Season == 2018:
        return float(pos_2018[str(row.Round)][pos_2018["index"] == row.home])
    else:
        return float(row.home_pos)
def visitor_pos_2018(row):
    if row.Season == 2018:
        return float(pos_2018[str(row.Round)][pos_2018["index"] == row.visitor])
    else:
        return float(row.visitor_pos)

pos_2019_until_16 = pd.read_csv("2019_league_positions_round 16.csv")
def home_pos_2019_until_16(row):
    if row.Season == 2019 and row.Round <= 16:
        return float(pos_2019_until_16[str(row.Round)][pos_2019_until_16["index"] == row.home])
    else:
        return float(row.home_pos)
def visitor_pos_2019_until_16(row):
    if row.Season == 2019 and row.Round <= 16:
        return float(pos_2019_until_16[str(row.Round)][pos_2019_until_16["index"] == row.visitor])
    else:
        return float(row.visitor_pos)


def home_pos_2019_beyond_16(row):
    if row.Season == 2019 and row.Round > 16:
        return float(rest_of_2019[row.Round][rest_of_2019["index"] == row.home])
    else:
        return float(row.home_pos)

def visitor_pos_2019_beyond_16(row):
    if row.Season == 2019 and row.Round > 16:
        return float(rest_of_2019[row.Round][rest_of_2019["index"] == row.visitor])
    else:
        return float(row.visitor_pos)

df['home_pos'] = df.apply(home_pos_2016, axis = 1)
df['home_pos'] = df.apply(home_pos_2017, axis = 1)
df['home_pos'] = df.apply(home_pos_2018, axis = 1)
df['home_pos'] = df.apply(home_pos_2019_until_16, axis = 1)
df['home_pos'] = df.apply(home_pos_2019_beyond_16, axis = 1)
df['visitor_pos'] = df.apply(visitor_pos_2016, axis = 1)
df['visitor_pos'] = df.apply(visitor_pos_2017, axis = 1)
df['visitor_pos'] = df.apply(visitor_pos_2018, axis = 1)
df['visitor_pos'] = df.apply(visitor_pos_2019_until_16, axis = 1)
df['visitor_pos'] = df.apply(visitor_pos_2019_beyond_16, axis = 1)


#adding average 2019 xG and xGA to every team in the rest of 2019
df['xG1'] = df["xg1"] + df["nsxg1"]
df['xG2'] = df["xg2"] + df["nsxg2"]
average_2019_xg = pd.read_csv("Average_2019_xG.csv")
def home_xG(row):
    if row.Season == 2019 and row.Round > 16:
        return float(average_2019_xg["xG"][average_2019_xg["index"] == row.home])
    else:
        return float(row.xg1)


def visitor_xG(row):
    if row.Season == 2019 and row.Round > 16:
        return float(average_2019_xg["xG"][average_2019_xg["index"] == row.visitor])
    else:
        return float(row.xg2)


df['xG1'] = df.apply(home_xG, axis = 1)
df['xG2'] = df.apply(visitor_xG, axis = 1)

cursor = df.iloc[0:1320].copy()

def home_result_df(row):
    if row.score1 != np.nan:
        if row.score1 > row.score2:
            return "win"
        elif row.score1 == row.score2:
            return "draw"
        else:
            return "loss"
    else:
        return np.nan

cursor["home_result"] = cursor.apply(home_result_df, axis=1)
df.iloc[:1320,-5] = cursor["home_result"]

#arranging the main dataframe
df = df[['Date',
         'Season',
         'Round',
         'home',
         'visitor',
         'home_pos',
         'visitor_pos',
         'spi1',
         'spi2',
         'prob1',
         'prob2',
         'probtie',
         'win%',
         'draw%',
         'loss%',
         'importance1',
         'importance2',
         'xG1',
         'xG2',
         'proj_score1',
         'proj_score2',
         'home_result',
         'score1',
         'score2']]

df.to_csv("final_data.csv")

import seaborn as sns
sns.countplot(df.home_result)
sns.distplot(df.importance1.dropna())

sns.distplot(df.spi1)
sns.distplot(df.xG1)
sns.boxplot("home_result", "spi1", data=df)
sns.boxplot("home_result", "home_pos", data=df)
sns.boxplot("home_result", "importance1", data=df)
sns.boxplot("home_result", "xG1", data=df)

sns.lineplot("home_pos","importance1", data=df)
sns.lineplot("visitor_pos", "importance2", data = df)
sns.jointplot("spi1", "importance1", data = df, kind="hex")
sns.jointplot("xG1", "score1", data = df, kind="hex")
sns.jointplot("xG1", "proj_score1", data = df, kind = 'hex')
sns.jointplot("proj_score1", "score1", data = df, kind = 'hex')



df_played_matches = df.iloc[0:1320]

sns.lineplot("xG1", "proj_score1", data = df_played_matches)
sns.lineplot("xG1", "score1", data = df_played_matches)



fig = sns.pairplot(data = df_played_matches[['home_pos', 'visitor_pos', 'spi1', 'spi2', 'win%', 'loss%',
       'importance1', 'importance2', 'xG1', 'xG2', 'home_result']], hue = "home_result")
fig.savefig("image.png")


from sklearn import metrics
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
# home teams
training_X = np.asarray(df_played_matches[df_played_matches.Season != 2019][['home_pos']]).reshape(-1, 1)
validation_X = np.asarray(df_played_matches[df_played_matches.Season == 2019][['home_pos']]).reshape(-1, 1)
wanted_X = np.asarray(df.iloc[1320:][['home_pos']]).reshape(-1, 1)

training_y = np.asarray(df_played_matches[df_played_matches.Season != 2019].importance1).reshape(-1, 1)
validation_y = np.asarray(df_played_matches[df_played_matches.Season == 2019].importance1).reshape(-1, 1)

# Kerner Ridge Regression
Importance_model = KernelRidge()
Importance_model.fit(training_X, training_y)

predicted_validation_y = Importance_model.predict(validation_X)
print(metrics.mean_absolute_error(validation_y, predicted_validation_y))
print(metrics.mean_squared_error(validation_y, predicted_validation_y))
print(np.sqrt(metrics.mean_squared_error(validation_y, predicted_validation_y)))


# Support Vector Regression
Importance_model = SVR()
Importance_model.fit(training_X, training_y)

predicted_validation_y = Importance_model.predict(validation_X)
print(metrics.mean_absolute_error(validation_y, predicted_validation_y))
print(metrics.mean_squared_error(validation_y, predicted_validation_y))
print(np.sqrt(metrics.mean_squared_error(validation_y, predicted_validation_y)))

from sklearn.svm import NuSVR
Importance_model = NuSVR()
Importance_model.fit(training_X, training_y)
predicted_validation_y = Importance_model.predict(validation_X)
print(metrics.mean_absolute_error(validation_y, predicted_validation_y))
print(metrics.mean_squared_error(validation_y, predicted_validation_y))
print(np.sqrt(metrics.mean_squared_error(validation_y, predicted_validation_y)))


# we chose SVR
training_X = np.vstack((training_X, validation_X))
training_y = np.vstack((training_y, validation_y))

Importance_model = SVR()
Importance_model.fit(training_X, training_y)

predicted_y = Importance_model.predict(wanted_X)
df.iloc[1320:,-9] = predicted_y





#visitor teams
training_X = np.asarray(df_played_matches[df_played_matches.Season != 2019][['visitor_pos']]).reshape(-1, 1)
validation_X = np.asarray(df_played_matches[df_played_matches.Season == 2019][['visitor_pos']]).reshape(-1, 1)
wanted_X = np.asarray(df.iloc[1320:][['visitor_pos']]).reshape(-1, 1)

training_y = np.asarray(df_played_matches[df_played_matches.Season != 2019].importance2).reshape(-1, 1)
validation_y = np.asarray(df_played_matches[df_played_matches.Season == 2019].importance2).reshape(-1, 1)


#We chosose SVR for importance 2
training_X = np.vstack((training_X, validation_X))
training_y = np.vstack((training_y, validation_y))

Importance_model = SVR()
Importance_model.fit(training_X, training_y)

predicted_y = Importance_model.predict(wanted_X)
df.iloc[1320:,-8] = predicted_y


sns.lineplot("home_pos","importance1", data=df.iloc[1320:])

sns.lineplot("home_pos","importance1", data=df)






#df.iloc[1320:].importance1 = df.iloc[1320:].apply(lambda row: average_importance_one.iloc[int(row.home_pos)-1], axis = 1)







import statsmodels.discrete.discrete_model as ds
import scipy.stats as st
from statsmodels.tools import add_constant as add_constant
df_win_lose = df_played_matches[['home_pos', 'visitor_pos', 'spi1', 'spi2', 'win%', 'loss%',
       'importance1', 'importance2', 'xG1', 'xG2', 'home_result']].copy()

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()

label_encoder = enc.fit(df_win_lose.home_result)
y = label_encoder.transform(df_win_lose.home_result)
df_win_lose["home_result"] = y
from sklearn import preprocessing
scaler = preprocessing.StandardScaler(with_mean=False, with_std=False)
scaler.fit(df_win_lose)
X_train = scaler.transform(df_win_lose)



df_win_lose = df_win_lose.astype(float)
df_constant = add_constant(df_win_lose)
st.chisqprob = lambda chisq, df: st.chi2.sf(chisq, df)
cols=df_win_lose.columns[:-1]
df_win_lose["home_result"] = y
model= ds.MNLogit(df_win_lose.home_result,df_constant[cols])
result=model.fit()
result.summary()














from sklearn.linear_model import LogisticRegression

df_train = df_played_matches[['home_pos', 'visitor_pos', 'spi1', 'spi2', 'win%', 'loss%', 'importance1', 'importance2', 'xG1', 'xG2', 'home_result']]

df_test = df.iloc[1320:][['home_pos', 'visitor_pos', 'spi1', 'spi2', 'win%', 'loss%', 'importance1', 'importance2', 'xG1', 'xG2', 'home_result']]

X_train = df_train[['home_pos', 'visitor_pos', 'spi1', 'spi2', 'win%', 'loss%', 'importance1', 'importance2', 'xG1', 'xG2']]
y_train = df_train['home_result']
X_test = df_test[['home_pos', 'visitor_pos', 'spi1', 'spi2', 'win%', 'loss%', 'importance1', 'importance2', 'xG1', 'xG2']]


from imblearn.over_sampling import SVMSMOTE
SVMSMOTE = SVMSMOTE()
columns = X_train.columns
up_sampled_X,up_sampled_y=SVMSMOTE.fit_sample(X_train, y_train)
up_sampled_X = pd.DataFrame(data=up_sampled_X,columns=columns )
up_sampled_y= pd.DataFrame(data=up_sampled_y,columns=['home_result'])



from sklearn import preprocessing
scaler = preprocessing.StandardScaler(with_mean=False)
scaler.fit(up_sampled_X)
X_train = scaler.transform(up_sampled_X)
X_test = scaler.transform(X_test)


classifier = LogisticRegression(max_iter=10000, multi_class = 'multinomial', solver = 'saga',penalty='elasticnet',l1_ratio = .5)
classifier.fit(X_train, up_sampled_y)
t = classifier.predict_proba(X_test)
classifier.classes_
df_predictions = df.iloc[1320:][['Date','Season','Round','home','visitor','proj_score1','proj_score2']]
df_predictions['win%'] = t[:,2]
df_predictions['draw%'] = t[:,0]
df_predictions['loss%'] = t[:,1]

df_2019_played = df[df.Season == 2019][['Date','Season','Round','home','visitor','proj_score1','proj_score2', 'home_result']].dropna()

def convert_to_percent(row):
    if row.home_result == "win":
        win = 1
        draw, loss = 0, 0
    elif row.home_result == "draw":
        draw = 1
        win,loss = 0, 0
    else:
        loss = 1
        win, draw = 0, 0
    return win, draw, loss

df_2019_played["win%"], df_2019_played["draw%"], df_2019_played["loss%"] =  zip(*df_2019_played.apply(convert_to_percent, axis = 1))

df_2019_predictions = pd.concat([df_2019_played, df_predictions], sort = False).reset_index()
del df_2019_predictions["home_result"]

tables = []
for i in teams_2019:
    cursor = df_2019_predictions[((df_2019_predictions.home == i)|(df_2019_predictions.visitor == i))][["Date", "Season", "Round", "home", "visitor", "win%", "draw%", "loss%"]]
    cursor['points'] = cursor.apply(league_points_calculator, axis = 1)
    tables.append((cursor[["Round","points"]].rename(columns={"points": i}).set_index('Round').T))

cursor = pd.concat(tables).reset_index()
cursor = cursor.set_index('index')
premier_league_table = cursor.sum(axis=1).sort_values(ascending = False).to_frame('total_pts').round(0).reset_index()

premier_league_table


np.unique(classifier.predict(X_test), return_counts=True)



from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

results_to_exclude = ["draw", "win", "loss"]
y = []
auc = []
fpr = []
fnr = []
count = 0
for i in results_to_exclude:
    df_train = df_played_matches[(df_played_matches.Season != 2019) & (df_played_matches.home_result != i)][['home_pos', 'visitor_pos', 'spi1', 'spi2', 'win%', 'loss%', 'draw%', 'importance1', 'importance2', 'xG1', 'xG2', 'home_result']]
    df_test = df[(df.Season == 2019) & (df.home_result != i)][['home_pos', 'visitor_pos', 'spi1', 'spi2', 'win%', 'loss%', 'importance1', 'importance2', 'xG1', 'xG2', 'home_result']].dropna()
    X_train = df_train[['home_pos', 'visitor_pos', 'spi1', 'spi2', 'win%', 'loss%', 'importance1', 'importance2', 'xG1', 'xG2']]
    y_train = df_train['home_result']
    df_predictions = df.iloc[1320:][['Date','Season','Round','home','visitor','proj_score1','proj_score2']]

    X_test = df_test[['home_pos', 'visitor_pos', 'spi1', 'spi2', 'win%', 'loss%', 'importance1', 'importance2', 'xG1', 'xG2']]
    y_test = df_test['home_result']

    from imblearn.over_sampling import SVMSMOTE
    SVMSMOTE = SVMSMOTE()
    columns = X_train.columns
    up_sampled_X,up_sampled_y=SVMSMOTE.fit_sample(X_train, y_train)
    up_sampled_X = pd.DataFrame(data=up_sampled_X,columns=columns )
    up_sampled_y= pd.DataFrame(data=up_sampled_y,columns=['home_result'])

    scaler = preprocessing.StandardScaler(with_mean=False)
    scaler.fit(up_sampled_X)
    X_train = scaler.transform(up_sampled_X)
    X_test = scaler.transform(X_test)

    classifier = LogisticRegression(max_iter=10000, solver='lbfgs')
    classifier.fit(X_train, up_sampled_y)
    t = classifier.predict_proba(X_test)[:, 1]

    auc_ = roc_auc_score(y_test, t)
    auc.append(auc_)
    enc = LabelEncoder()
    label_encoder = enc.fit(y_test)
    y_ = label_encoder.transform(y_test)
    y.append(y_)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_, t)
    fpr.append(false_positive_rate)
    fnr.append(true_positive_rate)
    count +=1

for i in range(0,3):
    print(results_to_exclude[i])
    print(auc[i])
    sns.lineplot(fpr[i], fnr[i])
























df_train = df_played_matches[(df_played_matches.Season != 2019)][['home_pos', 'visitor_pos', 'spi1', 'spi2', 'win%', 'loss%', 'importance1', 'importance2', 'xG1','xG2', 'home_result']]
df_test = df[(df.Season == 2019)][['home_pos', 'visitor_pos', 'spi1', 'spi2','win%', 'loss%', 'importance1', 'importance2', 'xG1', 'xG2', 'home_result']].dropna()
X_train = df_train[['home_pos', 'visitor_pos', 'spi1', 'spi2', 'win%', 'loss%', 'importance1','importance2', 'xG1', 'xG2']]
y_train = df_train['home_result']
X_test = df_test[['home_pos', 'visitor_pos', 'spi1', 'spi2', 'win%', 'loss%', 'importance1','importance2', 'xG1', 'xG2']]
y_test = df_test['home_result']


from imblearn.over_sampling import SVMSMOTE
SVMSMOTE = SVMSMOTE()
columns = X_train.columns
up_sampled_X,up_sampled_y=SVMSMOTE.fit_sample(X_train, y_train)
up_sampled_X = pd.DataFrame(data=up_sampled_X,columns=columns )
up_sampled_y= pd.DataFrame(data=up_sampled_y,columns=['home_result'])



from sklearn import preprocessing
scaler = preprocessing.StandardScaler(with_mean=False)
scaler.fit(up_sampled_X)
X_train = scaler.transform(up_sampled_X)
X_test = scaler.transform(X_test)


classifier = LogisticRegression(max_iter=10000, multi_class = 'multinomial', solver = 'saga',penalty='elasticnet',l1_ratio = .5)
classifier.fit(X_train, up_sampled_y)
y = classifier.predict_proba(X_test)
df_predictions.shape

df_predictions =  df[(df.Season == 2019)].dropna()[['Date','Season','Round','home','visitor','proj_score1','proj_score2']]

df_predictions['win%'] = y[:,2]
df_predictions['draw%'] = y[:,0]
df_predictions['loss%'] = y[:,1]

df_2019_played = df[df.Season == 2019].dropna()[['Date','Season','Round','home','visitor','proj_score1','proj_score2', 'home_result']].dropna()

def convert_to_percent(row):
    if row.home_result == "win":
        win = 1
        draw, loss = 0, 0
    elif row.home_result == "draw":
        draw = 1
        win,loss = 0, 0
    else:
        loss = 1
        win, draw = 0, 0
    return win, draw, loss

df_2019_played["win%"], df_2019_played["draw%"], df_2019_played["loss%"] =  zip(*df_2019_played.apply(convert_to_percent, axis = 1))

df_2019_predictions = df_2019_played
del df_2019_predictions["home_result"]

tables = []
for i in teams_2019:
    cursor = df_2019_predictions[((df_2019_predictions.home == i)|(df_2019_predictions.visitor == i))][["Date", "Season", "Round", "home", "visitor", "win%", "draw%", "loss%"]]
    cursor['points'] = cursor.apply(league_points_calculator, axis = 1)
    tables.append((cursor[["Round","points"]].rename(columns={"points": i}).set_index('Round').T))

cursor = pd.concat(tables).reset_index()
cursor = cursor.set_index('index')
premier_league_table = cursor.sum(axis=1).sort_values(ascending = False).to_frame('total_pts').round(0).reset_index()

premier_league_table





from sklearn.linear_model import LogisticRegression

df_train = df_played_matches[['home_pos', 'visitor_pos', 'spi1', 'spi2', 'win%', 'loss%', 'importance1', 'importance2', 'xG1', 'xG2', 'home_result']]

df_test = df.iloc[1320:][['home_pos', 'visitor_pos', 'spi1', 'spi2', 'win%', 'loss%', 'importance1', 'importance2', 'xG1', 'xG2', 'home_result']]

X_train = df_train[['home_pos', 'visitor_pos', 'spi1', 'spi2', 'win%', 'loss%', 'importance1', 'importance2', 'xG1', 'xG2']]
y_train = df_train['home_result']
X_test = df_test[['home_pos', 'visitor_pos', 'spi1', 'spi2', 'win%', 'loss%', 'importance1', 'importance2', 'xG1', 'xG2']]




classifier = LogisticRegression(max_iter=10000, multi_class = 'multinomial', solver = 'saga',penalty='elasticnet',l1_ratio = .5)
classifier.fit(X_train, y_train)
t = classifier.predict_proba(X_test)
classifier.classes_
df_predictions = df.iloc[1320:][['Date','Season','Round','home','visitor','proj_score1','proj_score2']]
df_predictions['win%'] = t[:,2]
df_predictions['draw%'] = t[:,0]
df_predictions['loss%'] = t[:,1]

df_2019_played = df[df.Season == 2019][['Date','Season','Round','home','visitor','proj_score1','proj_score2', 'home_result']].dropna()

def convert_to_percent(row):
    if row.home_result == "win":
        win = 1
        draw, loss = 0, 0
    elif row.home_result == "draw":
        draw = 1
        win,loss = 0, 0
    else:
        loss = 1
        win, draw = 0, 0
    return win, draw, loss

df_2019_played["win%"], df_2019_played["draw%"], df_2019_played["loss%"] =  zip(*df_2019_played.apply(convert_to_percent, axis = 1))

df_2019_predictions = pd.concat([df_2019_played, df_predictions], sort = False).reset_index()
del df_2019_predictions["home_result"]

tables = []
for i in teams_2019:
    cursor = df_2019_predictions[((df_2019_predictions.home == i)|(df_2019_predictions.visitor == i))][["Date", "Season", "Round", "home", "visitor", "win%", "draw%", "loss%"]]
    cursor['points'] = cursor.apply(league_points_calculator, axis = 1)
    tables.append((cursor[["Round","points"]].rename(columns={"points": i}).set_index('Round').T))

cursor = pd.concat(tables).reset_index()
cursor = cursor.set_index('index')
premier_league_table = cursor.sum(axis=1).sort_values(ascending = False).to_frame('total_pts').round(0).reset_index()

premier_league_table
np.unique(classifier.predict(X_test), return_counts=True)
