import pycrfsuite
from hw2_corpus_tool import *
import sys


ans = get_data('/Users/anubhabsen/Desktop/Spring 2020/NLP/Assignments/Assignment2/m_train')
y_train = []
X_train = []
for i in range(len(ans)):
    for j in ans[i]:
        tokens = set([])
        pos_tags = set([])
        if not j.pos:
            features = []
        else:
            for post in j.pos:
                tokens.add(post.token)
                pos_tags.add(post.pos)
        features = list(tokens) + list(pos_tags)
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

tagger = pycrfsuite.Tagger()
tagger.open('model.crfsuite')
print(tagger.tag(X_train[3]), y_train[3])
tot = 0
correct = 0
for i in range(len(X_train)):
	tot += 1
	if tagger.tag(X_train[i])[0] == y_train[i][0]:
		correct += 1

print(tot, correct)