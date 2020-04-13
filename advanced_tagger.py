# Saurabh's POS trick, act_tag and predicted act tag as feature (test this)

import pycrfsuite
from hw2_corpus_tool import *
import sys

train_dir = sys.argv[1] # /Users/anubhabsen/Desktop/Spring 2020/NLP/Assignments/Assignment2/t_train
test_dir = sys.argv[2]  # /Users/anubhabsen/Desktop/Spring 2020/NLP/Assignments/Assignment2/t_test
out_file = sys.argv[3]  # predictions.txt
ans = get_data(train_dir)
y_train = []
X_train = []
X_test = []
y_test = []
for i in range(len(ans)):
    for index, j in enumerate(ans[i]):
        features = []
        if index > 0:
            if ans[i][index - 1].speaker == j.speaker:
                features.append('F')
                features.append('F')
            else:
                features.append('T')
                features.append('F')
        else:
            features.append('T')
            features.append('T')
        tokens = set([])
        pos_tags = set([])
        if not j.pos:
            features.append('no word')
        else:
            for index, post in enumerate(j.pos):
                tokens.add('TOKEN_'+post.token)
                pos_tags.add('POS_'+post.pos)
                if len(post.token) > 3:
                    tokens.add(post.token[-3:])
                    tokens.add(post.token[-2:])
                elif len(post.token) == 3:
                    tokens.add(post.token[-2:])
                if len(post.pos) > 2:
                    pos_tags.add(post.pos[:2])
                if index > 0 and index < len(j.pos) - 1:
                    tokens.add(j.pos[index - 1].token + '|' + post.token + '|' + j.pos[index + 1].token)
                    pos_tags.add(j.pos[index - 1].pos + '|' + post.pos + '|' + j.pos[index + 1].pos)
                elif index == 0 and len(j.pos) > 1:
                    tokens.add(post.token + '|' + j.pos[index + 1].token)
                    pos_tags.add(post.pos + '|' + j.pos[index + 1].pos)
                elif index == len(j.pos) - 1 and len(j.pos) > 1:
                    tokens.add(j.pos[index - 1].token + '|' + post.token)
                    pos_tags.add(j.pos[index - 1].pos + '|' + post.pos)
            features = features + list(tokens) + list(pos_tags)
        if 'Wh' in j.text:
            features.append('F')
        else:
            features.append('T')
        # if '?' in j.text:
        #     features.append('T')
        # else:
        #     features.append('F')
        X_train.append([features])
        y_train.append([j.act_tag])

trainer = pycrfsuite.Trainer(verbose=False)

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
                features.append('F')
                features.append('F')
            else:
                features.append('T')
                features.append('F')
        else:
            features.append('T')
            features.append('T')
        tokens = set([])
        pos_tags = set([])
        if not j.pos:
            features.append('no word')
        else:
            for index, post in enumerate(j.pos):
                tokens.add('TOKEN_'+post.token)
                pos_tags.add('POS_'+post.pos)
                if len(post.token) > 3:
                    tokens.add(post.token[-3:])
                    tokens.add(post.token[-2:])
                elif len(post.token) == 3:
                    tokens.add(post.token[-2:])
                if len(post.pos) > 2:
                    pos_tags.add(post.pos[:2])
                if index > 0 and index < len(j.pos) - 1:
                    tokens.add(j.pos[index - 1].token + '|' + post.token + '|' + j.pos[index + 1].token)
                    pos_tags.add(j.pos[index - 1].pos + '|' + post.pos + '|' + j.pos[index + 1].pos)
                elif index == 0 and len(j.pos) > 1:
                    tokens.add(post.token + '|' + j.pos[index + 1].token)
                    pos_tags.add(post.pos + '|' + j.pos[index + 1].pos)
                elif index == len(j.pos) - 1 and len(j.pos) > 1:
                    tokens.add(j.pos[index - 1].token + '|' + post.token)
                    pos_tags.add(j.pos[index - 1].pos + '|' + post.pos)
            features = features + list(tokens) + list(pos_tags)
        if 'Wh' in j.text:
            features.append('F')
        else:
            features.append('T')
        # if '?' in j.text:
        #     features.append('T')
        # else:
        #     features.append('F')
        X_test.append([features])
        # y_test.append([j.act_tag])

tagger = pycrfsuite.Tagger()
tagger.open('model.crfsuite')
# tot = 0
# correct = 0
write_string = ""
for i in range(len(X_test)):
    # tot += 1
    if X_test[i][0][1] == "T":      # First utterance of dialogue
        write_string += "\n" + str(tagger.tag(X_test[i])[0]) + "\n"
    else:                               # Not first utterance of dialogue
        write_string += str(tagger.tag(X_test[i])[0]) + "\n"
    # if tagger.tag(X_test[i])[0] == y_test[i][0]:
    #     correct += 1

# print(correct / tot * 100)
f = open(out_file, "w")
f.write(write_string[1:])
f.close()