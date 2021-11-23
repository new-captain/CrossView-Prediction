
import argparse
from Preprocess import Preprocess,PerClassSplit,splitTrainTestSet
from TrainAE import TrainAAE_patch,SaveFeatures_AAE,TrainVAE_patch,SaveFeatures_VAE,Validate_feature
from TrainPCL import TrainContrast, ContrastPredict
import numpy as np
import joblib
from TrainSVM import TrainSVM,TestSVM, TrainNN, TestNN
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import time
import csv
from utils import reports
dataset_names = ['IP','SA','PU']
parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                             " various hyperspectral datasets")
parser.add_argument('--dataset', type=str, default='IP', choices=dataset_names,
                    help="Dataset to use.")
parser.add_argument('--train',type=int, default=1,choices=(0,1))
parser.add_argument('--save_feature', type=int, default=1)
parser.add_argument('--train_contrast',type=int, default=0,choices=(0,1))

parser.add_argument('--perclass', type=float, default=10) 
parser.add_argument('--device', type=str, default="cuda:0", choices=("cuda:0","cuda:1"))
parser.add_argument('--projection_dim', type=int, default=128)
parser.add_argument('--encoded_dim', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--Windowsize', type=int, default=27)
parser.add_argument('--classifier', type=str, default='svm',choices=("linear","svm"))
parser.add_argument('--Patch_channel', type=int, default=20)
parser.add_argument('--RandomSeed', type=bool, default=False)


args = parser.parse_args()

if args.RandomSeed:
    randomState=345
else:
    randomState=int(np.random.randint(1, high=1000))
#args.temperature=args.temperature/100
args.perclass=args.perclass/100
#print(args)
output_units = 9 if (args.dataset == 'PU' or args.dataset == 'PC') else 16
Datadir='./DataArray/'
XPath = Datadir + 'X.npy'
yPath = Datadir + 'y.npy'
# # 2. Train AE
#(54129, 27, 27, 15)
#(54129,1)
train_start = time.time()
if args.train:
    Preprocess(XPath, yPath, args.dataset, args.Windowsize, Patch_channel=args.Patch_channel)
    #TrainVAE_patch(args,XPath, Patch_channel=args.Patch_channel, windowSize=args.Windowsize, encoded_dim=args.encoded_dim,batch_size=args.batch_size)
    #print("Finish VAE")
    TrainAAE_patch(args,XPath,Patch_channel=args.Patch_channel,windowSize=args.Windowsize,encoded_dim=args.encoded_dim,batch_size=args.batch_size)
    print("Finish AAE")

# # 3. Save features
if args.save_feature:
    test_start1 = time.time()
    VAEPath=Datadir+'VAE_Features.npy'
    AAEPath=Datadir+'AAE_Features.npy'
    VAEFeatures=SaveFeatures_VAE(XPath,Patch_channel=args.Patch_channel,windowSize=args.Windowsize,encoded_dim=args.encoded_dim,batch_size=args.batch_size)
    np.save(VAEPath,VAEFeatures)
    print("Finish save VAE feature")
    AAEFeatures=SaveFeatures_AAE(XPath,Patch_channel=args.Patch_channel,windowSize=args.Windowsize,encoded_dim=args.encoded_dim,batch_size=args.batch_size)
    np.save(AAEPath,AAEFeatures)
    print("Finish save AAE feature")
    test_stop1 = time.time()
else:
    VAEPath=Datadir+'VAE_Features.npy'
    AAEPath=Datadir+'AAE_Features.npy'

#############4.Train Contrastive learning
if args.train_contrast==True:
    TrainContrast(args,AAEPath,VAEPath,batch_size=256,projection_dim=args.projection_dim,input_dim=args.encoded_dim)
    train_stop = time.time()
    print("Finish Trained Contrast")
test_start2 = time.time()
ContrastFeature=ContrastPredict(AAEPath,VAEPath,batch_size=128,projection_dim=args.projection_dim,input_dim=args.encoded_dim)
test_stop2 = time.time()
np.save(Datadir+'ContrastFeature.npy',ContrastFeature)
print("Finish save contrast feature")

y = np.load(yPath)
stratify = np.arange(0, output_units, 1)
for i in range(10):
    #for feature in ["Contrast","AAE","VAE"]:
    for feature in ["VAE"]:
        if feature == "Contrast":
            fea = ContrastFeature
            projection_dim=args.projection_dim
        elif feature == "AAE":
            fea = AAEFeatures
            projection_dim=1024
        elif feature == "VAE":
            fea = VAEFeatures
            projection_dim = 1024
        if args.perclass > 1:
            Xtrain, Xtest, ytrain, ytest = PerClassSplit(fea, y, int(args.perclass), stratify,randomState=randomState)
        else:
            Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(fea, y, 1 - args.perclass,randomState=randomState)
        np.save(Datadir + 'Xtrain.npy', Xtrain)
        np.save(Datadir + 'ytrain.npy', ytrain)
        np.save(Datadir + 'Xtest.npy', Xtest)
        np.save(Datadir + 'ytest.npy', ytest)
        if args.classifier=='svm':
            train_start2 = time.time()
            SVM_model=TrainSVM(Xtrain,ytrain)
            train_stoF2 = time.time()
            joblib.dump(SVM_model, './models/SVM.model')
            SVM_model=joblib.load('./models/SVM.model')
            test_start3 = time.time()
            Predictions=TestSVM(Xtest,SVM_model)
            test_stop3 = time.time()
            print("Finish Trained SVM")
        else:
            train_start2 = time.time()
            ModelPath=TrainNN(n_features=projection_dim,n_classes=output_units,Datadir=Datadir)
            train_stop2 = time.time()

            test_start3 = time.time()
            Predictions=TestNN(n_features=projection_dim,n_classes=output_units, ModelPath=ModelPath,Datadir=Datadir)
            test_stop3 = time.time()
        
   
        ytest=np.load(Datadir+'ytest.npy')
        classification = classification_report(ytest.astype(int), Predictions)
        print(classification)
        classification, confusion, oa, each_acc, aa, kappa = reports(Predictions, ytest.astype(int), args.dataset)
        ## SVM
        each_acc_str = ','.join(str(x) for x in each_acc)
        #add_info=[args.dataset,args.perclass,args.temperature,args.Windowsize,args.classifier,feature,oa,aa,kappa,TrainTime,TestTime]+each_acc_str.split('[')[0].split(']')[0].split(',')
        print("SVM,OA,AA,kappa")
        print(feature,"OA,AA:",oa,aa)

        #add_info=[args.dataset,args.perclass,args.temperature,args.Windowsize,args.classifier,feature,oa,aa,kappa]+each_acc_str.split('[')[0].split(']')[0].split(',')
        #print(add_info)
        # csvFile = open("compare.csv", "a")
        # writer = csv.writer(csvFile)
        # writer.writerow(add_info)
        # csvFile.close()

