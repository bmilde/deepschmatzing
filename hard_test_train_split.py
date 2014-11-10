from utils import unspeech_utils    
import argparse

def save_set(ids,name,myclass,folder):
    with open(args.filelists+name+'_'+myclass+'.txt','w') as out:
        out.write('\n'.join(ids))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train / Dev / Test split, files in format all_<class>.txt in folder filelist-dir')
    parser.add_argument('-f', '--filelists-dir', dest="filelists", help='file list basedir', default='corpus/voxforge/', type=str)
    parser.add_argument('-l', '--classes', dest='classes', help='classes as comma seperated list',type=str,default='de,fr')
    parser.add_argument('-b', '--basedir',dest='basedir', help='base dir of all files, should end with /', default = '/srv/data/speech/', type=str)
    
    parser.add_argument('-d', '--dev-set',dest='dev', help='how many ids to use for dev set', default = 100, type=int)
    parser.add_argument('-t', '--test-set',dest='test', help='how many ids to use for test set', default = 200, type=int)

    args = parser.parse_args()
    classes = (args.classes).split(',')

    for myclass in classes:
        ids = unspeech_utils.loadIdFile(args.filelists+'all_'+myclass+'.txt',basedir=args.basedir)
        test_ids = ids[-1*args.test:]
        dev_ids = ids[-1*(args.test + args.dev) : -1*args.test]
        train_ids = ids[:-1*(args.test + args.dev)]

        save_set(train_ids,'train',myclass,args.basedir)
        save_set(dev_ids,'dev',myclass,args.basedir)
        save_set(test_ids,'test',myclass,args.basedir)

        print 'Sizes of sets for class',myclass,' (train/dev/test) :', len(train_ids), len(dev_ids), len(test_ids)

        #check test ids
        for myid in test_ids:
            if myid in train_ids:
                print 'Error, found test id in training set!',myid
            if myid in dev_ids:
                print 'Error, found test id in dev set!',myid
        
        #check dev ids
        for myid in dev_ids:
            if myid in train_ids:
                print 'Error, found dev id in training set!',myid
            if myid in test_ids:
                print 'Error, found dev id in test set!',myid

