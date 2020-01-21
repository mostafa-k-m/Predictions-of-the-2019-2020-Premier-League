from math import floor
import numpy as np
import pandas as pd
from sklearn.svm import NuSVR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder, RobustScaler, label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def points_adder(table):
    for i in range(17,39):
        table[i] = table[i] + table[i-1]
    return table


def form_calculator_(row, df, home):
    if home:
        x = row.home
    else:
        x = row.visitor
    number_of_matches = 5
    matches = df[((df.home == x) | (df.visitor == x)) & ((df.Season == row.Season) &(df.Round < row.Round))].tail(number_of_matches)[["home", "visitor", "win_prob", "draw_prob", "loss_prob"]]
    number_of_matches = matches["home"].count()
    if number_of_matches == 0:
        return (7/5)
    win = []
    draw = []
    for i in range(number_of_matches):
        row = matches.iloc[i]
        if row.home == x:
            win.append(float(row.win_prob))
            draw.append(float(row.draw_prob))
        else:
            win.append(float(row.loss_prob))
            draw.append(float(row.draw_prob))
    return float(sum(win)*3 + sum(draw)*1)/float(number_of_matches)

def league_points_calculator(row, useless, i):
    if row['home'] == i:
        return 3*row["win_prob"] + 1*row["draw_prob"]
    else:
        return 3*row["loss_prob"] + 1*row["draw_prob"]

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


def positionator(row, season, df_, from_, to_, home):
    if home:
        x = row.home
    else:
        x = row.visitor
    if row.Season == season and row.Round in list(range(from_,to_)):
        if season == 2019 and from_>16:
            return float(df_[(row.Round)][df_["index"] == x])
        else:
            return float(df_[str(row.Round)][df_["index"] == x])
    else:
        if home:
            return float(row.home_pos)
        else:
            return float(row.visitor_pos)

def predictor(df, number_of_matches):
    number_of_matches = int(number_of_matches)
    df_played_matches = df.iloc[0:number_of_matches-1]

    training_X = np.asarray(df_played_matches[df_played_matches.Season != 2019][['home_pos']]).reshape(-1, 1)
    validation_X = np.asarray(df_played_matches[df_played_matches.Season == 2019][['home_pos']]).reshape(-1, 1)

    training_y = np.asarray(df_played_matches[df_played_matches.Season != 2019].importance1).reshape(-1, 1)
    validation_y = np.asarray(df_played_matches[df_played_matches.Season == 2019].importance1).reshape(-1, 1)
    training_X = np.vstack((training_X, validation_X))
    training_y = np.vstack((training_y, validation_y))

    Importance1_model = NuSVR()
    Importance1_model.fit(training_X, training_y)


    training_X = np.asarray(df_played_matches[df_played_matches.Season != 2019][['visitor_pos']]).reshape(-1, 1)
    validation_X = np.asarray(df_played_matches[df_played_matches.Season == 2019][['visitor_pos']]).reshape(-1, 1)
    training_y = np.asarray(df_played_matches[df_played_matches.Season != 2019].importance2).reshape(-1, 1)
    validation_y = np.asarray(df_played_matches[df_played_matches.Season == 2019].importance2).reshape(-1, 1)
    training_X = np.vstack((training_X, validation_X))
    training_y = np.vstack((training_y, validation_y))
    Importance2_model = NuSVR()
    Importance2_model.fit(training_X, training_y)

    df_train = df_played_matches[['home_pos', 'visitor_pos', 'spi1', 'spi2', 'loss%', 'home_form', 'visitor_form', 'importance1', 'importance2', 'xG1', 'xG2', 'home_result']]
    df_test = df.iloc[number_of_matches:][['Round', 'home_pos', 'visitor_pos', 'spi1', 'spi2', 'loss%', 'home_form', 'visitor_form', 'importance1', 'importance2', 'xG1', 'xG2']]
    X_train = df_train[['home_pos', 'visitor_pos', 'spi1', 'spi2', 'loss%', 'home_form', 'visitor_form', 'importance1', 'importance2', 'xG1', 'xG2']]
    y_train = df_train['home_result']

    from imblearn.over_sampling import SVMSMOTE
    up_sampled_X,up_sampled_y=SVMSMOTE().fit_sample(np.asarray(X_train), np.asarray(y_train))
    columns = X_train.columns
    up_sampled_X = pd.DataFrame(data=up_sampled_X,columns=columns )
    up_sampled_y= pd.DataFrame(data=up_sampled_y,columns=['home_result'])
    from sklearn import preprocessing
    scaler = preprocessing.RobustScaler()
    scaler.fit(up_sampled_X)
    X_train = scaler.transform(up_sampled_X)

    df_copy = df[df.Season == 2019].copy()

    df_copy["win_prob"], df_copy["draw_prob"], df_copy["loss_prob"] =  zip(*df_copy.apply(convert_to_percent, axis = 1))
    classifier = LogisticRegression(max_iter=300, multi_class = 'multinomial', solver = 'saga',penalty='elasticnet',l1_ratio = .5)
    classifier.fit(X_train, up_sampled_y)

    start_test = 0
    start_index = number_of_matches - 1140
    df_copy.iloc[(start_index):,-3] = 0
    df_copy.iloc[(start_index):,-2] = 0
    df_copy.iloc[(start_index):,-1] = 0
    start_round = int(df_test.Round.iloc[0])
    round_ = start_round
    for ii in range(start_round,39):
        X_test = df_test[df_test.Round == ii][['home_pos', 'visitor_pos', 'spi1', 'spi2', 'loss%', 'home_form', 'visitor_form', 'importance1', 'importance2', 'xG1', 'xG2']]
        X_test = scaler.transform(X_test)
        t = classifier.predict_proba(X_test)
        df_copy.iloc[(start_index):(start_index+10),-3] = t[:,2]
        df_copy.iloc[(start_index):(start_index+10),-2] = t[:,0]
        df_copy.iloc[(start_index):(start_index+10),-1] = t[:,1]
        df_copy["home_form"] = df_copy.apply(form_calculator_, args = (df_copy, True), axis=1)
        df_copy["visitor_form"] = df_copy.apply(form_calculator_,args = (df_copy, False),axis=1)
        tables = []
        teams_2019 = df[df.Season == 2019].home.unique()
        for i in teams_2019:
            cursor = df_copy[((df_copy.home == i)|(df_copy.visitor == i))][["Date", "Season", "Round", "home", "visitor", "win_prob", "draw_prob", "loss_prob"]]
            cursor['points'] = cursor.apply(league_points_calculator, args = ([], i), axis = 1)
            tables.append((cursor[["Round","points"]].rename(columns={"points": i}).set_index('Round').T))

        cursor = pd.concat(tables).reset_index()
        pos_2019 = points_adder(cursor)
        for i in range(16,39):
            pos_2019 = pos_2019.sort_values(by=[i],ascending=False)
            pos_2019[i] = list(range(1,21))
        iterator = [((2019, pos_2019, 17, 39, True), (2019, pos_2019, 17, 39, False))]
        for i in iterator:
            df_copy['home_pos'] = df_copy.apply(positionator, args=i[0], axis = 1)
            df_copy['visitor_pos'] = df_copy.apply(positionator, args=i[1], axis = 1)
        X_test = np.asarray(df_copy[df_copy.Round == ii][['home_pos', 'visitor_pos', 'spi1', 'spi2', 'loss%', 'home_form', 'visitor_form', 'importance1', 'importance2', 'xG1', 'xG2']])
        df_copy.iloc[(start_index):(start_index+10),14] = Importance1_model.predict(X_test[:,0].reshape(-1, 1))
        df_copy.iloc[(start_index):(start_index+10),15] = Importance2_model.predict(X_test[:,1].reshape(-1, 1))
        df_test.iloc[(start_test):(start_test+10):,6:9] = df_copy.iloc[(start_index):(start_index+10):,12:15]
        df_test.iloc[(start_test):(start_test+10):,1:2] = df_copy.iloc[(start_index):(start_index+10):,5:6]
        start_index += 10
        start_test += 10
        print(f"{floor((start_index-number_of_matches+1140)*100/(1519-number_of_matches))}% done.")
    df = pd.concat((df.iloc[0:1140],df_copy[['Date', 'Season', 'Round', 'home', 'visitor', 'home_pos', 'visitor_pos', 'spi1', 'spi2', 'win%', 'draw%', 'loss%', 'home_form', 'visitor_form', 'importance1', 'importance2', 'xG1', 'xG2', 'proj_score1', 'proj_score2', 'home_result', 'score1', 'score2']]))
    wanted_X = np.asarray(df.iloc[number_of_matches:][['home_pos']]).reshape(-1, 1)
    predicted_y = Importance1_model.predict(wanted_X)
    df.iloc[number_of_matches:,-9] = predicted_y
    wanted_X = np.asarray(df.iloc[number_of_matches:][['visitor_pos']]).reshape(-1, 1)
    predicted_y = Importance2_model.predict(wanted_X)
    df.iloc[number_of_matches:,-8] = predicted_y

    X_test = scaler.transform(df.iloc[number_of_matches:][['home_pos', 'visitor_pos', 'spi1', 'spi2', 'loss%', 'home_form', 'visitor_form', 'importance1', 'importance2', 'xG1', 'xG2']])
    t = classifier.predict_proba(X_test)
    df_predictions = df.iloc[number_of_matches:][['Date','Season','Round','home','visitor','proj_score1','proj_score2']]
    df_predictions['win_prob'] = t[:,2]
    df_predictions['draw_prob'] = t[:,0]
    df_predictions['loss_prob'] = t[:,1]

    df_2019_played = df.iloc[1140:number_of_matches+1][['Date','Season','Round','home','visitor','proj_score1','proj_score2', 'home_result']]

    df_2019_played["win_prob"], df_2019_played["draw_prob"], df_2019_played["loss_prob"] =  zip(*df_2019_played.apply(convert_to_percent, axis = 1))

    df_2019_predictions = pd.concat([df_2019_played, df_predictions], sort = False).reset_index()


    del df_2019_predictions["home_result"]
    df_2019_predictions.to_csv("reticulate2.csv")

def roc_curves(df, number_of_matches):
    number_of_matches = int(number_of_matches)
    df_played_matches = df.iloc[0:number_of_matches-1]
    classifier = LogisticRegression(max_iter=300, multi_class = 'multinomial', solver = 'saga',penalty='elasticnet',l1_ratio = .95)
    classifier = OneVsRestClassifier(classifier)
    count = 0
    Data = df_played_matches[['home_pos', 'visitor_pos', 'spi1', 'spi2', 'draw%', 'home_form', 'visitor_form', 'importance1', 'importance2', 'xG1', 'xG2']]
    Target = df_played_matches['home_result']
    y = np.asarray(Target)
    enc = LabelEncoder()
    label_encoder = enc.fit(y)
    y = label_encoder.transform(y)
    X = np.asarray(Data)
    n_classes = 3
    n_samples, n_features = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,random_state=5)
    from imblearn.over_sampling import SVMSMOTE
    SVMSMOTE = SVMSMOTE()
    columns = Data.columns
    up_sampled_X,up_sampled_y=SVMSMOTE.fit_sample(X_train, y_train)
    up_sampled_X = pd.DataFrame(data=up_sampled_X,columns=columns )
    up_sampled_y= pd.DataFrame(data=up_sampled_y,columns=['home_result'])

    scaler = RobustScaler()
    scaler.fit(up_sampled_X)
    X_train = scaler.transform(up_sampled_X)
    X_test = scaler.transform(X_test)

    y_train = label_binarize(np.asarray(up_sampled_y), classes=[0, 1, 2])
    y_test = label_binarize(np.asarray(y_test), classes=[0, 1, 2])
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    data = [fpr[2], tpr[2]]
    dataset = pd.DataFrame({'FPR': data[0], 'TPR': data[1]})
    dataset.to_csv("reticulate1.csv")

def scatterer(df, number_of_matches):
    number_of_matches = int(number_of_matches)
    df_played_matches = df.iloc[0:number_of_matches-1]
    df_train = df_played_matches[(df_played_matches.Season != 2019)][['home_pos', 'visitor_pos', 'spi1', 'spi2', 'loss%', 'draw%', 'importance1', 'importance2', 'xG1', 'xG2', 'home_result']]

    X_train = df_train[['home_pos', 'visitor_pos', 'spi1', 'spi2', 'loss%', 'importance1', 'importance2', 'xG1', 'xG2']]
    y_train = df_train['home_result']


    scaler = RobustScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(np.asarray(X_train))
    pca = PCA(n_components=2)
    X = pca.fit_transform(X_train)
    d = pd.DataFrame(X, columns = ["x", "y"])
    d["label"] = np.vstack(np.asarray(y_train).reshape(1140,1))
    d.to_csv("reticulate_3.csv")
