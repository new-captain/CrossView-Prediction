import numpy as np
import torch
import joblib
from TrainSVM import TrainSVM,TestSVM, TrainNN, TestNN
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from utils import reports
from Preprocess import Preprocess,PerClassSplit,splitTrainTestSet
import random

def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


class AEDataset(torch.utils.data.Dataset):
    def __init__(self,Datapath,transform):
        # 1. Initialize file path or list of file names.
        self.Datalist=np.load(Datapath)
        self.transform=transform
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        Data=self.transform(self.Datalist[index].astype('float64'))
        Data=Data.view(1,Data.shape[0],Data.shape[1],Data.shape[2])
        return Data
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.Datalist)
class MYDataset(torch.utils.data.Dataset):
    def __init__(self,Datapath,Labelpath,transform):
        # 1. Initialize file path or list of file names.
        self.Datalist=np.load(Datapath)
        self.Labellist=(np.load(Labelpath)).astype(int)
        self.transform=transform
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).

        index=index
        Data=self.transform(self.Datalist[index])
        Data=Data.view(1,Data.shape[0],Data.shape[1],Data.shape[2])
        return Data ,self.Labellist[index]
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.Datalist)
def generate_(batch_size,dim):
    return torch.from_numpy(
        np.random.multivariate_normal(mean=np.zeros([dim ]), cov=np.diag(np.ones([dim])),
                                      size=batch_size)).type(torch.float)

def TrainAAE_patch(arg,XPath,Patch_channel=15,windowSize=25,encoded_dim=64,batch_size=128):
    import torch
    from DefinedModels import Dec_AAE, Enc_AAE, Discriminant
    from torchvision import transforms
    import numpy as np
    from tqdm import tqdm
    trans = transforms.Compose(transforms=[
        transforms.ToTensor(),
        transforms.Normalize(np.zeros(Patch_channel), np.ones(Patch_channel))
    ])
    Enc_patch = Enc_AAE(channel=int(20),output_dim=encoded_dim,windowSize=windowSize).cuda()
    Dec_patch = Dec_AAE(channel=int(20),windowSize=windowSize,input_dim=encoded_dim).cuda()
    discriminant = Discriminant(encoded_dim).cuda()

    patch_data = AEDataset(XPath,trans)
    Patch_loader = torch.utils.data.DataLoader(dataset=patch_data, batch_size=batch_size, shuffle=True)
    optim_enc = torch.optim.Adam(Enc_patch.parameters(), lr=1e-3, weight_decay=0.0005)
    optim_dec=torch.optim.Adam(Dec_patch.parameters(), lr=1e-3, weight_decay=0.0005)
    optim_enc_gen = torch.optim.SGD(Enc_patch.parameters(), lr=1e-4, weight_decay=0.000)  # 1e-5
    optim_disc = torch.optim.SGD(discriminant.parameters(), lr=5e-5, weight_decay=0.000)  # 5e-6
    criterion = torch.nn.MSELoss()
    epochs=80
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        rl=0
        l_dis_loss=0
        l_encl=0
        print('Epoch No {}'.format(epoch))
        for i, (data) in enumerate(Patch_loader):
            #########################reconstruction phase
            data=data.cuda().float()
            input=data
            gt=data
            #sequantial:
            #input=data[:,:,15:30,:,:]
            #gt=data[:,:,0:15,:,:]
            #parity
            #input=data[:,:,1::2,:,:]
            #gt=data[:,:,::2,:,:]
            #even layers: a=data[:,:,::2,:,:]
            #odd layers:a=data[:,:,1::2,:,:] 1，3，5，7，9
            #overlap:
            #input=data[:,:,10:30,:,:]
            #gt=data[:,:,0:20,:,:]
            #random:
            #original_input=data[:,:,15:30,:,:]
            #additional_input_index=torch.index_select(data,2,torch.tensor(sorted(random.sample(range(5,15),5))).cuda())
            #input=torch.cat((additional_input_index,original_input),dim=2)
            #gt=data[:,:,0:20,:,:]

            Enc_patch.train()
            Dec_patch.train()
            optim_dec.zero_grad()
            optim_enc.zero_grad()
            optim_disc.zero_grad()
            optim_enc_gen.zero_grad()
            map, code =Enc_patch(input)
            recon=Dec_patch(code)
            loss=criterion(gt,recon)
            loss.backward(retain_graph=True)
            optim_dec.step()
            optim_enc.step()
    ##################################################regularization phase
            discriminant.train()
            Enc_patch.eval()
            gauss=torch.FloatTensor(generate_(batch_size,encoded_dim)).cuda()
            fake_pred = discriminant(gauss)
            true_pred = discriminant(code)
            dis_loss=-(torch.mean(fake_pred) -torch.mean(true_pred))

            dis_loss.backward(retain_graph=True,inputs=list(discriminant.parameters()))
            #dis_loss.backward(retain_graph=True)
            optim_disc.step()
            ######################################
            discriminant.train()
            Enc_patch.train()
            _ , code2 = Enc_patch(input)
            true_pred2 = discriminant(code2)
            encl=-torch.mean(true_pred2)
            encl.backward(retain_graph=True)
            optim_enc_gen.step()
            rl = rl + loss.item()
            l_dis_loss+=dis_loss.item()
            l_encl+=encl.item()
        print('\nPatch Reconstruction Loss: {}  dis loss: {}   regularization loss : {}'.format(rl/len(patch_data),l_dis_loss/len(patch_data),l_encl/len(patch_data)))
        torch.save(Enc_patch.state_dict(),'./models/Enc_AAE.pth')
        if (epoch %10==0):
            print("Validate")
            Validate_feature(arg,'AAE')
    return 0
def SaveFeatures_AAE(XPath,Patch_channel=15,windowSize=25,encoded_dim=64,batch_size=128):
    import torch
    from DefinedModels import  Enc_AAE
    from torchvision import transforms
    import numpy as np
    from tqdm import tqdm
    from Preprocess import feature_normalize2, L2_Norm
    trans = transforms.Compose(transforms=[
        transforms.ToTensor(),
        transforms.Normalize(np.zeros(Patch_channel), np.ones(Patch_channel))
    ])
    Enc_patch = Enc_AAE(channel=20,output_dim=encoded_dim,windowSize=windowSize).cuda()
    Enc_patch.load_state_dict(torch.load('./models/Enc_AAE.pth'))
    patch_data = AEDataset(XPath,trans)
    Patch_loader = torch.utils.data.DataLoader(dataset=patch_data, batch_size=batch_size, shuffle=False)
    Patch_Features=[]
    print('Start save patch features...')
    for i, (data) in enumerate(Patch_loader):
        data=data.cuda().float()
        input=data
        #parity
        #input=data[:,:,1::2,:,:]

        #sequantial
        #input=data[:,:,15:30,:,:]
        #input=data[:,:,1::2,:,:]
        #overlap:
        #input=data[:,:,10:30,:,:]
        #random:
        #original_input=data[:,:,15:30,:,:]
        #additional_input_index=torch.index_select(data,2,torch.tensor(sorted(random.sample(range(5,15),5))).cuda())
        #input=torch.cat((additional_input_index,original_input),dim=2)
        Enc_patch.eval()
        feature,code= Enc_patch(input)
        for num in range(len(feature)):
            Patch_Features.append(np.array(feature[num].cpu().detach().numpy()))
    # Patch_Features=feature_normalize2(Patch_Features)
    Patch_Features = L2_Norm(Patch_Features)

    return Patch_Features


def TrainVAE_patch(arg,XPath,Patch_channel=20,windowSize=25,encoded_dim=64,batch_size=128):
    import torch
    from DefinedModels import Dec_VAE, Enc_VAE
    from torchvision import transforms
    import numpy as np
    from tqdm import tqdm
    trans = transforms.Compose(transforms=[
        transforms.ToTensor(),
        transforms.Normalize(np.zeros(Patch_channel), np.ones(Patch_channel))
    ])
    Enc_patch = Enc_VAE(channel=15,output_dim=encoded_dim,windowSize=windowSize).cuda()
    Dec_patch = Dec_VAE(channel=15,windowSize=windowSize,input_dim=encoded_dim).cuda()
    patch_data = AEDataset(XPath,trans)
    Patch_loader = torch.utils.data.DataLoader(dataset=patch_data, batch_size=batch_size, shuffle=True)
    optim_enc = torch.optim.Adam(Enc_patch.parameters(), lr=1e-3, weight_decay=0.0005)
    optim_dec=torch.optim.Adam(Dec_patch.parameters(), lr=1e-3, weight_decay=0.0005)
    criterion = torch.nn.MSELoss()
    epochs=80
    for epoch in range(epochs):
        rl=0

        print('Epoch No {}'.format(epoch))
        for i, (data) in enumerate(Patch_loader):
            #########################reconstruction phase
            data=data.cuda().float()
            #origin:
            #input=data
            #gt=data

            #sequ####
            #input=data[:,:,0:15,:,:]
            #gt=data[:,:,15:30,:,:]

            ##parity#####
            input=data[:,:,::2,:,:]
            gt=data[:,:,1::2,:,:]
            #even layers: a=data[:,:,::2,:,:]
            #odd layers:a=data[:,:,1::2,:,:] 1，3，5，7，9

            ##overlap#####
            #input=data[:,:,10:30,:,:]
            #gt=data[:,:,0:20,:,:]

            ##random#####
            #original_input=data[:,:,0:15,:,:]
            #additional_input_index=torch.index_select(data,2,torch.tensor(sorted(random.sample(range(15,25),5))).cuda())
            #input=torch.cat((original_input,additional_input_index),dim=2)
            #gt=data[:,:,10:30,:,:]

            Enc_patch.train()
            Dec_patch.train()
            optim_dec.zero_grad()
            optim_enc.zero_grad()
            H,mu, sigma =Enc_patch(input)
            std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
            code=mu + sigma * std_z.clone().detach().cuda()
            recon=Dec_patch(code)
            ll = latent_loss(mu, sigma)
            loss=criterion(gt,recon)+ll
            loss.backward()
            optim_dec.step()
            optim_enc.step()
            rl = rl + loss.item()

        print('Patch Loss: {}'.format(rl/len(patch_data)))
        torch.save(Enc_patch.state_dict(),'./models/Enc_VAE.pth')
        if ((epoch+1) %10==0):
            print("Validate")
            Validate_feature(arg,'VAE')
    return 0
def SaveFeatures_VAE(XPath,Patch_channel=20,windowSize=25,encoded_dim=64,batch_size=128):
    import torch
    from DefinedModels import  Enc_VAE
    from torchvision import transforms
    import numpy as np
    from tqdm import tqdm
    from Preprocess import feature_normalize2, L2_Norm
    trans = transforms.Compose(transforms=[
        transforms.ToTensor(),
        transforms.Normalize(np.zeros(Patch_channel), np.ones(Patch_channel))
    ])
    Enc_patch = Enc_VAE(channel=15,output_dim=encoded_dim,windowSize=windowSize).cuda()
    Enc_patch.load_state_dict(torch.load('./models/Enc_VAE.pth'))
    patch_data = AEDataset(XPath,trans)
    Patch_loader = torch.utils.data.DataLoader(dataset=patch_data, batch_size=batch_size, shuffle=False)
    Patch_Features=[]
    print('Start save patch features...')
    for i, (data) in enumerate(Patch_loader):
        data = data.cuda().float()
        #sequ:
        #input=data[:,:,0:15,:,:]
        #parity:
        input=data[:,:,::2,:,:]
        #overlap:
        #input=data[:,:,10:30,:,:]
        #random:
        #original_input=data[:,:,0:15,:,:]
        #additional_input_index=torch.index_select(data,2,torch.tensor(sorted(random.sample(range(15,25),5))).cuda())
        #input=torch.cat((original_input,additional_input_index),dim=2)
        Enc_patch.eval()
        H, mu, sigma = Enc_patch(input)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
        code = mu + sigma * std_z.clone().detach().cuda()
        for num in range(len(H)):
            Patch_Features.append(np.array(H[num].cpu().detach().numpy()))
    # Patch_Features=feature_normalize2(Patch_Features)
    Patch_Features = L2_Norm(Patch_Features)

    return Patch_Features


def Validate_feature(args,model):
    #feature='Contrast'
    Datadir='./DataArray/'
    yPath = Datadir + 'y.npy'
    XPath = Datadir + 'X.npy'
    y = np.load(yPath)
    output_units=16
    randomState=int(np.random.randint(1, high=1000))

    #save feature:
    VAEPath=Datadir+'VAE_Features.npy'
    AAEPath=Datadir+'AAE_Features.npy'

    stratify = np.arange(0, output_units, 1)
    if model == "AAE":
        projection_dim=1024
        AAEFeatures=SaveFeatures_AAE(XPath,Patch_channel=args.Patch_channel,windowSize=args.Windowsize,encoded_dim=args.encoded_dim,batch_size=args.batch_size)
        np.save(AAEPath,AAEFeatures)
        fea=AAEFeatures
    elif model == "VAE":
        projection_dim = 1024
        VAEFeatures=SaveFeatures_VAE(XPath,Patch_channel=args.Patch_channel,windowSize=args.Windowsize,encoded_dim=args.encoded_dim,batch_size=args.batch_size)
        np.save(VAEPath,VAEFeatures)
        fea=VAEFeatures

    if args.perclass > 1:
        Xtrain, Xtest, ytrain, ytest = PerClassSplit(fea, y, int(args.perclass), stratify,randomState=randomState)
    else:
        Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(fea, y, 1 - args.perclass,randomState=randomState)
    np.save(Datadir + 'Xtrain.npy', Xtrain)
    np.save(Datadir + 'ytrain.npy', ytrain)
    np.save(Datadir + 'Xtest.npy', Xtest)
    np.save(Datadir + 'ytest.npy', ytest)
    SVM_model=TrainSVM(Xtrain,ytrain)
    joblib.dump(SVM_model, './models/SVM.model')
    SVM_model=joblib.load('./models/SVM.model')
    Predictions=TestSVM(Xtest,SVM_model)
    print("Finish Trained SVM")
   
    ytest=np.load(Datadir+'ytest.npy')
    classification = classification_report(ytest.astype(int), Predictions)
    print(classification)
    classification, confusion, oa, each_acc, aa, kappa = reports(Predictions, ytest.astype(int), args.dataset)

    each_acc_str = ','.join(str(x) for x in each_acc)
    #print("SVM,OA,AA,kappa")
    print("SVM,OA,AA,kappa")
    print("OA,AA:",oa,aa)

    #add_info=[args.dataset,args.perclass,args.Windowsize,args.classifier,model,oa,aa,kappa]+each_acc_str.split('[')[0].split(']')[0].split(',')
    #print(add_info)
