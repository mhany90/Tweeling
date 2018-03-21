from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

def output_scores(labelnames, Ytest, Yguess, verbose = True):
    acc = accuracy_score(Ytest, Yguess)
    prec = precision_score(Ytest, Yguess, average = None)
    rec = recall_score(Ytest, Yguess, average = None)
    fsc = f1_score(Ytest, Yguess, average = None)
    cmatrix = confusion_matrix(Ytest, Yguess)
    print('Overall accuracy:\t%.3f\n' % acc)
    if(verbose):
        print('Precision per class:')
        for n, p in zip(labelnames, prec):
            print('\t%s\t%.2f' % (n, p))
        print('Recall per class:')
        for n, p in zip(labelnames, rec):
            print('\t%s\t%.2f' % (n, p))
        print('F-score per class:')
        for n, p in zip(labelnames, fsc):
            print('\t%s\t%.2f' % (n, p))
    else:
        print('Average precision:\t%.3f' % (sum(prec) / len(prec)))
        print('Average recall:\t%.3f' % (sum(rec) / len(rec)))
        print('Average F-score:\t%.3f' % (sum(fsc) / len(fsc)))
    for label, line in zip(labelnames, cmatrix):
        print('%-10s%s' % (label, '\t'.join((str(i) for i in line))))
