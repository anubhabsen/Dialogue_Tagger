import pycrfsuite
from hw2_corpus_tool import *
import sys


ans = get_data('/Users/anubhabsen/Desktop/Spring 2020/NLP/Assignments/Assignment2/m_train')
y_train = []
X_train = []
X_test = []
y_test = []
for i in range(len(ans)):
    for index, j in enumerate(ans[i]):
        tokens = set([])
        pos_tags = set([])
        if not j.pos:
            features = []
        else:
            for post in j.pos:
                tokens.add(post.token)
                pos_tags.add(post.pos)
        features = list(tokens) + list(pos_tags)
        try:
            if ans[i][index - 1].speaker == j.speaker:
                features.append('False')
                features.append('False')
            else:
                features.append('True')
                features.append('False')
        except:
            features.append('False')
            features.append('True')
        X_train.append([features])
        y_train.append([j.act_tag])

trainer = pycrfsuite.Trainer(verbose=True)

for i in range(len(X_train)):
    trainer.append(X_train[i], y_train[i])

trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 150,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})


trainer.train('model.crfsuite')

test_data = get_data('/Users/anubhabsen/Desktop/Spring 2020/NLP/Assignments/Assignment2/m_test')
for i in range(len(test_data)):
    for index, j in enumerate(test_data[i]):
        tokens = set([])
        pos_tags = set([])
        if not j.pos:
            features = []
        else:
            for post in j.pos:
                tokens.add(post.token)
                pos_tags.add(post.pos)
        features = list(tokens) + list(pos_tags)
        try:
            if test_data[i][index - 1].speaker == j.speaker:
                features.append('False')
                features.append('False')
            else:
                features.append('True')
                features.append('False')
        except:
            features.append('False')
            features.append('True')
        X_test.append([features])
        y_test.append([j.act_tag])

tagger = pycrfsuite.Tagger()
tagger.open('model.crfsuite')
tot = 0
correct = 0
for i in range(len(X_test)):
	tot += 1
	if tagger.tag(X_test[i])[0] == y_test[i][0]:
		correct += 1

print(correct / tot * 100)