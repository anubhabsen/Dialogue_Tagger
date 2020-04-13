import pycrfsuite
from hw2_corpus_tool import *
import sys

train_dir = sys.argv[1] # /Users/anubhabsen/Desktop/Spring 2020/NLP/Assignments/Assignment2/t_train
test_dir = sys.argv[2]  # /Users/anubhabsen/Desktop/Spring 2020/NLP/Assignments/Assignment2/t_test
out_file = sys.argv[3]  # predictions.txt
ans = get_data(train_dir)

trainer = pycrfsuite.Trainer(verbose=False)

y_train = []
X_train = []
X_test = []
y_test = []
for i in range(len(ans)):
    x = []
    y = []
    for index, j in enumerate(ans[i]):
        features = []
        if index > 0:
            if ans[i][index - 1].speaker == j.speaker:
                pass
            else:
                features.append('SPEAKER_CHANGE')
        else:
            features.append('FIRST_UTTERANCE')
        tokens = set([])
        pos_tags = set([])
        if not j.pos:
            features.append('no')
        else:
            for post in j.pos:
                tokens.add('TOKEN_'+post.token)
                pos_tags.add('POS_'+post.pos)
            features = features + list(tokens) + list(pos_tags)
        X_train.append([features])
        y_train.append([j.act_tag])
        x.append(features)
        y.append(j.act_tag)
    # print(len(x), len(y))
    # trainer.append(x, y)

for i in range(len(X_train)):
    trainer.append(X_train[i], y_train[i])

trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})


trainer.train('model.crfsuite')

test_data = get_data(test_dir)
for i in range(len(test_data)):
    for index, j in enumerate(test_data[i]):
        features = []
        if index > 0:
            if test_data[i][index - 1].speaker == j.speaker:
                pass
            else:
                features.append('SPEAKER_CHANGE')
        else:
            features.append('FIRST_UTTERANCE')
        # if not features:
        #     print('before', j)
        tokens = set([])
        pos_tags = set([])
        if not j.pos:
            features.append('no')
        else:
            for post in j.pos:
                tokens.add('TOKEN_'+post.token)
                pos_tags.add('POS_'+post.pos)
            features = features + list(tokens) + list(pos_tags)
        X_test.append([features])
        # y_test.append([j.act_tag])

tagger = pycrfsuite.Tagger()
tagger.open('model.crfsuite')
# tot = 0
# correct = 0
write_string = ""
for i in range(len(X_test)):
    # tot += 1
    if 'FIRST_UTTERANCE' in X_test[i][0]:      # First utterance of dialogue
        write_string += "\n" + str(tagger.tag(X_test[i])[0]) + "\n"
    else:                               # Not first utterance of dialogue
        write_string += str(tagger.tag(X_test[i])[0]) + "\n"
    # if tagger.tag(X_test[i])[0] == y_test[i][0]:
    #     correct += 1

# print(correct / tot * 100)
f = open(out_file, "w")
f.write(write_string[1:])
f.close()