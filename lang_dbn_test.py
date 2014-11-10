import argparse
from utils import unspeech_utils
import lang_dbn
from lang_dbn import MyModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate FBANK features (=logarithmic mel frequency filter banks) for all supplied files in file list (txt).')
    parser.add_argument('-f', '--filelists-dir', dest="filelists", help='file list basedir', default='corpus/voxforge/', type=str)
    parser.add_argument('-v', dest='verbose', help='verbose output', action='store_true', default=False)
    parser.add_argument('-b', '--basedir',dest='basedir', help='base dir of all files, should end with /', default = './', type=str)
    parser.add_argument('-m', '--with-model',dest='modelfilename', help='filename to load trained model', default = '', type=str)
    parser.add_argument('-n', '--set-name',dest='name', help='name of the set', default = 'dev', type=str)

    args = parser.parse_args()

    model = lang_dbn.load(args.modelfilename)
    lang2num = model.lang2num
    num2lang = dict(zip(lang2num.values(),lang2num.keys()))
    langs = lang2num.keys()

    test_ids = []
    classes = []

    for lang in langs:
        ids = unspeech_utils.loadIdFile(args.filelists+args.name+'_'+lang+'.txt',basedir=args.basedir)
        for myid in ids:
            test_ids.append(myid)
            classes.append(lang2num[lang])

    y_pred = []
    y_multi_pred = []

    #now test on heldout ids (dev set)
    for myid in test_ids:
        print 'testing',myid
        pred,multi_pred = model.predict_utterance(myid)
        print 'Preds: ', [num2lang[x] for x in multi_pred], ' agg:', num2lang[pred]
        y_multi_pred.append(multi_pred)
        y_pred.append(pred)

    print 'Single classifier performance scores:'
    for i in xrange(len(multi_pred)):
        print 'Pred #',i,':'
        #print 'Window size', window_sizes[i],'step size',step_sizes[i]
        lang_dbn.print_classificationreport(classes, [pred[i] for pred in y_multi_pred])

    lang_dbn.print_classificationreport(classes, y_pred)
