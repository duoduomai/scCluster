from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from layers import ZINBLoss, MeanAct, DispAct
import numpy as np
from torch.cuda.amp import GradScaler
import math
from utils import evaluate
from C3_loss import Instance_C3

def buildNetwork(layers, type, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        net.append(nn.BatchNorm1d(layers[i], affine=True))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="selu":
            net.append(nn.SELU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        elif activation=="elu":
            net.append(nn.ELU())
    return nn.Sequential(*net)

class scCluster(nn.Module):
    def __init__(self, input_dim1, input_dim2,
            encodeLayer1=[], encodeLayer2=[], decodeLayer1=[], decodeLayer2=[], tau=1., t=10, device="cuda",
            activation="elu", sigma1=2.5, sigma2=.1, alpha=1., gamma=1., phi1=0.0001, phi2=0.0001, cutoff = 0.5):
        super(scCluster, self).__init__()
        self.tau=tau
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.cutoff = cutoff
        self.activation = activation
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.alpha = alpha
        self.gamma = gamma
        self.phi1 = phi1
        self.phi2 = phi2
        self.t = t
        self.device = device
        self.encoder1 = buildNetwork([input_dim1]+encodeLayer1, type="encode", activation=activation)
        self.encoder2 = buildNetwork([input_dim2]+encodeLayer2, type="encode", activation=activation)
        self.decoder1 = buildNetwork(decodeLayer1, type="decode", activation=activation)
        self.decoder2 = buildNetwork(decodeLayer2, type="decode", activation=activation)  
        self.dec_mean1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), MeanAct())
        self.dec_disp1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), DispAct())
        self.dec_mean2 = nn.Sequential(nn.Linear(decodeLayer2[-1], input_dim2), MeanAct())
        self.dec_disp2 = nn.Sequential(nn.Linear(decodeLayer2[-1], input_dim2), DispAct())
        self.dec_pi1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), nn.Sigmoid())
        self.dec_pi2 = nn.Sequential(nn.Linear(decodeLayer2[-1], input_dim2), nn.Sigmoid())
        self.zinb_loss = ZINBLoss()
        self.z_dim = encodeLayer1[-1]
    
    def cal_latent(self, z):
        sum_y = torch.sum(torch.square(z), dim=1)
        num = -2.0 * torch.matmul(z, z.t()) + torch.reshape(sum_y, [-1, 1]) + sum_y
        num = num / self.alpha
        num = torch.pow(1.0 + num, -(self.alpha + 1.0) / 2.0)
        zerodiag_num = num - torch.diag(torch.diag(num))
        latent_p = (zerodiag_num.t() / torch.sum(zerodiag_num, dim=1)).t()
        return num, latent_p

    def kmeans_loss(self, z):
        dist1 = self.tau*torch.sum(torch.square(z.unsqueeze(1) - self.mu), dim=2)
        temp_dist1 = dist1 - torch.reshape(torch.mean(dist1, dim=1), [-1, 1])
        q = torch.exp(-temp_dist1)
        q = (q.t() / torch.sum(q, dim=1)).t()
        q = torch.pow(q, 2)
        q = (q.t() / torch.sum(q, dim=1)).t()
        dist2 = dist1 * q
        return dist1, torch.mean(torch.sum(dist2, dim=1))
    
    def target_distribution(self, q):
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()
    
    def kldloss(self, p, q):
        epsilon = 1e-8  
        q = q.clamp(min=epsilon)
        p = p.clamp(min=epsilon)
        q_log = torch.log_softmax(q, dim=-1)  
        p_log = torch.log_softmax(p, dim=-1)  
        kl_div = torch.sum(p * (p_log - q_log), dim=-1)  
        return torch.mean(kl_div)  
    
    def c3_loss(self, x1, x2):
        z1, _, _, _, _, _ = self.forwardAE1(x1)
        z2, _, _, _, _, _ = self.forwardAE2(x2)
        instance_loss = Instance_C3(x1.shape[0], 0.7)
        return instance_loss.forward(z1 ,z2)

    def x_drop(self, x, p=0.2):
        mask_list = [torch.rand(x.shape[1]) < p for _ in range(x.shape[0])]
        mask = torch.vstack(mask_list)
        new_x = x.clone()
        new_x[mask] = 0.0
        return new_x

    def forwardAE1(self, x):
        z = self.encoder1(x+torch.randn_like(x) * self.sigma1)
        h = self.decoder1(z)
        mean = self.dec_mean1(h)
        disp = self.dec_disp1(h)
        pi = self.dec_pi1(h)
        h0 = self.encoder1(x)
        num, lq = self.cal_latent(h0)
        return h0, num, lq, mean, disp, pi
    
    def forwardAE2(self, x):
        z = self.encoder2(x+torch.randn_like(x) * self.sigma2)
        h = self.decoder2(z)
        mean = self.dec_mean2(h)
        disp = self.dec_disp2(h)
        pi = self.dec_pi2(h)
        h0 = self.encoder2(x)
        num, lq = self.cal_latent(h0)
        return h0, num, lq, mean, disp, pi

    def encodeBatch1(self, X, batch_size=256):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.to(self.device)
        self.eval() 
        encoded = [] 
        num = X.shape[0] 
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch): 
            x1batch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            x1batch = torch.from_numpy(x1batch).float() 
            inputs = Variable(x1batch).to(self.device)
            z, _, _, _,  _, _= self.forwardAE1(inputs)
            encoded.append(z.data) 
        encoded = torch.cat(encoded, dim=0) 
        return encoded

    def encodeBatch2(self, X, batch_size=256):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.to(self.device)
        self.eval() 
        encoded = [] 
        num = X.shape[0] 
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch): 
            x2batch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            x2batch = torch.from_numpy(x2batch).float() 
            inputs = Variable(x2batch).to(self.device)
            z, _, _, _,  _, _= self.forwardAE2(inputs)
            encoded.append(z.data) 
        encoded = torch.cat(encoded, dim=0) 
        return encoded

    def pretrain_autoencoder(self, X1, X_raw1, sf1, X2, X_raw2, sf2, 
                             batch_size=256, lr=0.001, epochs=400, ae_save=True, ae_weights='AE_weights.pth.tar'):
        dataset = TensorDataset(torch.Tensor(X1), torch.Tensor(X_raw1), torch.Tensor(sf1), torch.Tensor(X2), torch.Tensor(X_raw2), torch.Tensor(sf2))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Pretraining stage")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        num1 = X1.shape[0]
        num2 = X2.shape[0]
        scaler = GradScaler()
        kl_div = nn.KLDivLoss(reduction="batchmean")
        for epoch in range(epochs):
            loss_val = 0
            recon_loss1_val = 0
            recon_loss2_val = 0
            for batch_idx, (x1_batch, x_raw1_batch, sf1_batch, x2_batch, x_raw2_batch, sf2_batch) in enumerate(dataloader):
                
                x1_tensor = Variable(x1_batch).to(self.device)
                x_raw1_tensor = Variable(x_raw1_batch).to(self.device)
                sf1_tensor = Variable(sf1_batch).to(self.device)
                x2_tensor = Variable(x2_batch).to(self.device)
                x_raw2_tensor = Variable(x_raw2_batch).to(self.device)
                sf2_tensor = Variable(sf2_batch).to(self.device)

                zbatch1, z_num1, lqbatch1, mean1_tensor, disp1_tensor, pi1_tensor = self.forwardAE1(x1_tensor)
                zbatch2, z_num2, lqbatch2, mean2_tensor, disp2_tensor, pi2_tensor = self.forwardAE2(x2_tensor)

                recon_loss1 = self.zinb_loss(x=x_raw1_tensor, mean=mean1_tensor, disp=disp1_tensor, pi=pi1_tensor, scale_factor=sf1_tensor)
                recon_loss2 = self.zinb_loss(x=x_raw2_tensor, mean=mean2_tensor, disp=disp2_tensor, pi=pi2_tensor, scale_factor=sf2_tensor)

                lqbatch = torch.cat((lqbatch1, lqbatch2),1)
                lpbatch = self.target_distribution(lqbatch)

                kl_loss = self.kldloss(lpbatch, lqbatch) 

                if epoch+1 >= epochs * self.cutoff:
                   loss = recon_loss1 + recon_loss2 + kl_loss * self.phi1 
                else:
                   loss = recon_loss1 + recon_loss2

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                loss_val += loss.item() * len(x1_batch)
                recon_loss1_val += recon_loss1.item() * len(x1_batch)
                recon_loss2_val += recon_loss2.item() * len(x2_batch)

            loss_val = loss_val/num1
            recon_loss1_val = recon_loss1_val/num1
            recon_loss2_val = recon_loss2_val/num2

            if epoch%self.t == 0:
               print('Pretrain epoch {}, Total loss:{:.6f}, ZINB loss1:{:.6f}, ZINB loss2:{:.6f}'.format(epoch+1, loss_val, recon_loss1_val, recon_loss2_val))

        if ae_save:
            torch.save({'ae_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, ae_weights)


    def fit(self, X1, X_raw1, sf1, X2, X_raw2, sf2, y=None, lr=1., n_clusters = 0,
            batch_size=256, num_epochs=10, update_interval=1, tol=1e-3, save_dir=""):
        '''X: tensor data'''
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.to(self.device)
        print("Clustering stage")
        X1 = torch.tensor(X1).to(self.device)
        X_raw1 = torch.tensor(X_raw1).to(self.device)
        sf1 = torch.tensor(sf1).to(self.device)
        X2 = torch.tensor(X2).to(self.device)
        X_raw2 = torch.tensor(X_raw2).to(self.device)
        sf2 = torch.tensor(sf2).to(self.device)
        self.mu = Parameter(torch.Tensor(n_clusters, self.z_dim), requires_grad=True)
        
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)
             
        print("Initializing cluster centers with kmeans.")
        kmeans = KMeans(n_clusters, n_init=20)


        X1 = X1.cpu().numpy()
        X2 = X2.cpu().numpy() 
        Zdata1 = self.encodeBatch1(X1, batch_size=batch_size)
        Zdata2 = self.encodeBatch2(X2, batch_size=batch_size)
        Zdata = torch.cat((Zdata1, Zdata2), 1)

        #latent
        self.y_pred = kmeans.fit_predict(Zdata.data.cpu().numpy())
        self.y_pred_last = self.y_pred
        self.mu = nn.Parameter(torch.Tensor(kmeans.cluster_centers_))
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))

        
        self.train()
        num = X1.shape[0]
        num_batch = int(math.ceil(1.0*X1.shape[0]/batch_size))
        lst = []  
        pred = []  
        best_ari = 0.0  

        for epoch in range(num_epochs):
            Zdata1 = self.encodeBatch1(X1, batch_size=batch_size)
            Zdata2 = self.encodeBatch2(X2, batch_size=batch_size)

            Zdata = torch.cat((Zdata1, Zdata2),1)

            dist, _ = self.kmeans_loss(Zdata)
            self.y_pred = torch.argmin(dist, dim=1).data.cpu().numpy()    

            acc,ami,nmi,ari = evaluate(y, self.y_pred)
            
            pred.append(self.y_pred)
            zhibiao = (acc, ami, nmi, ari)
            lst.append(zhibiao)

            if best_ari < ari:
                best_ari = ari
                Zdata = Zdata.cpu().numpy()
                np.save(save_dir,Zdata)
                print('save successful')

            self.y_pred_last = self.y_pred

            loss_val = 0.0
            recon_loss1_val = 0.0
            recon_loss2_val = 0.0
            cluster_loss_val = 0.0
            kl_loss_val = 0.0
            c3_loss_val = 0.0
            scaler = GradScaler()
            for batch_idx in range(num_batch):
                x1_batch = X1[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                x_raw1_batch = X_raw1[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                sf1_batch = sf1[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                x2_batch = X2[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                x_raw2_batch = X_raw2[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                sf2_batch = sf2[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]

                inputs1 = Variable(torch.tensor(x1_batch)).to(self.device)
                inputs2 = Variable(torch.tensor(x2_batch)).to(self.device)

                rawinputs1 = Variable(x_raw1_batch.clone().detach()).to(self.device)
                sfinputs1 = Variable(sf1_batch.clone().detach()).to(self.device)
                rawinputs2 = Variable(x_raw2_batch.clone().detach()).to(self.device)
                sfinputs2 = Variable(sf2_batch.clone().detach()).to(self.device)

                zbatch1, z_num1, lqbatch1, mean1_tensor, disp1_tensor, pi1_tensor = self.forwardAE1(inputs1)
                zbatch2, z_num2, lqbatch2, mean2_tensor, disp2_tensor, pi2_tensor = self.forwardAE2(inputs2)
                zbatch = torch.cat((zbatch1, zbatch2),1)

                x1 = self.x_drop(inputs1, p=0.2)
                x2 = self.x_drop(inputs2, p=0.2)

                c3_loss = self.c3_loss(x1, x2)

                _, cluster_loss = self.kmeans_loss(zbatch)
                
                recon_loss1 = self.zinb_loss(x=rawinputs1, mean=mean1_tensor, disp=disp1_tensor, pi=pi1_tensor, scale_factor=sfinputs1)
                recon_loss2 = self.zinb_loss(x=rawinputs2, mean=mean2_tensor, disp=disp2_tensor, pi=pi2_tensor, scale_factor=sfinputs2)
                
                lqbatch = torch.cat((lqbatch1, lqbatch2), 1)
                target = self.target_distribution(lqbatch)
                kl_loss = self.kldloss(target, lqbatch)
                if epoch < 20:
                    loss = recon_loss1 + recon_loss2 + kl_loss * self.phi2 + cluster_loss * self.gamma
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    cluster_loss_val += cluster_loss.data * len(inputs1)
                    recon_loss1_val += recon_loss1.data * len(inputs1)
                    recon_loss2_val += recon_loss2.data * len(inputs2)
                    kl_loss_val += kl_loss.data * len(inputs1)

                    loss_val = cluster_loss_val + recon_loss1_val + recon_loss2_val + kl_loss_val
                else:
                    
                    loss = c3_loss
                    optimizer.zero_grad()
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    loss_val = c3_loss_val
                    c3_loss_val += c3_loss.data * len(inputs1)

            if (epoch < 20):
                print("#Epoch %d: Total: %.6f Clustering Loss: %.6f ZINB Loss1: %.6f ZINB Loss2: %.6f KL Loss: %.6f " % (epoch + 1, loss_val / num, cluster_loss_val / num, recon_loss1_val / num, recon_loss2_val / num, kl_loss_val / num))
            else:
                print("#Epoch %d: Total: %.6f c3_loss: %.6f" % (epoch + 1, loss_val / num, c3_loss_val / num))

        
        final_acc, final_ami, final_nmi, final_ari = evaluate(y, self.y_pred)
        return self.y_pred, final_acc, final_ami, final_nmi, final_ari
