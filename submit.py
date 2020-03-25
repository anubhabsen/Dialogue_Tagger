import pycrfsuite
from hw2_corpus_tool import *


ans = get_data('/Users/anubhabsen/Desktop/Spring 2020/NLP/Assignments/Assignment2/mini')
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
        X_train.append(features)
        y_train.append(j.act_tag)
    #   break
    # break

trainer = pycrfsuite.Trainer(verbose=False)

# for i in range(len(X_train)):
#     print(X_train[i], y_train[i])
trainer.append(X_train, y_train)

trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})


trainer.train('model.crfsuite')

# tagger = pycrfsuite.Tagger()