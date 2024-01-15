from time import time
import torch
import pandas as pd
from scCluster import scCluster
import scanpy as sc
from preprocess import read_dataset, normalize
from utils import *
import os


torch.manual_seed(44)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--cutoff', default=0.5, type=float, help='Start to train combined layer after what ratio of epoch')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_file', default='data_file')
    parser.add_argument('--maxiter', default=30, type=int)
    parser.add_argument('--pretrain_epochs', default=300, type=int)
    parser.add_argument('--gamma', default=.1, type=float, help='coefficient of clustering loss')
    parser.add_argument('--tau', default=1., type=float, help='fuzziness of clustering loss')                    
    parser.add_argument('--phi1', default=0.001, type=float, help='coefficient of KL loss in pretraining stage')
    parser.add_argument('--phi2', default=0.001, type=float, help='coefficient of KL loss in clustering stage')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--lr', default=1., type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results/')
    parser.add_argument('--ae_weight_file', default='AE_weights.tar')
    parser.add_argument('--resolution', default=0.8, type=float)
    parser.add_argument('--n_neighbors', default=30, type=int)
    parser.add_argument('-el1','--encodeLayer1', nargs='+', default=[256,64,32,16])
    parser.add_argument('-el2','--encodeLayer2', nargs='+', default=[256,64,32,16])
    parser.add_argument('-dl1','--decodeLayer1', nargs='+', default=[16,32,64,256])
    parser.add_argument('-dl2','--decodeLayer2', nargs='+', default=[16,32,64,256])
    parser.add_argument('--sigma1', default=2.5, type=float)
    parser.add_argument('--sigma2', default=1.5, type=float)
    parser.add_argument('--f1', default=1000, type=float, help='Number of RNA after feature selection')
    parser.add_argument('--f2', default=2000, type=float, help='Number of eSNP after feature selection')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()       

    data_name = 'meldata'
    data_path = './data/%s.txt'% data_name
    label_path = './data/%s_truelabels.csv' % data_name
    args.ae_weight_file = './results/%s_AE_weights_snp.pth' % data_name
    args.save_dir = './model/%s_snp.pth' % data_name
    pre_label_path = './data/pre_label/%s_snp.csv' % data_name
    snp_data_path = './data/%s_snp.txt'% data_name

    x1 = pd.read_csv(data_path, header=None).to_numpy().astype(np.float32)
    x2 = pd.read_csv(snp_data_path, header=None).to_numpy().astype(np.float32)
    y = pd.read_csv(label_path, header=None)[0]
    lab = y.unique().tolist()
    ind = list(range(0, len(lab)))
    mapping = {j: i for i, j in zip(ind, lab)}
    y = y.map(mapping).to_numpy()

    adata1 = sc.AnnData(x1)
    adata1.obs['Group'] = y
    adata1 = read_dataset(adata1,transpose=False,test_split=False,copy=False)
    adata1 = normalize(adata1,size_factors=True,normalize_input=False,logtrans_input=False)
    input_size1 = adata1.n_vars

    adata2 = sc.AnnData(x2)
    adata2.obs['Group'] = y
    adata2 = read_dataset(adata2,transpose=False,test_split=False,copy=False)
    adata2 = normalize(adata2,size_factors=True, normalize_input=False,logtrans_input=False)
    input_size2 = adata2.n_vars

    model = scCluster(input_dim1=input_size1, input_dim2=input_size2, 
                           tau=args.tau,
                           encodeLayer1=args.encodeLayer1, 
                           encodeLayer2=args.encodeLayer2, 
                           decodeLayer1=args.decodeLayer1, 
                           decodeLayer2=args.decodeLayer2,
                           activation='sigmoid', 
                           sigma1=args.sigma1, sigma2=args.sigma2, 
                           gamma=args.gamma,
                           cutoff = args.cutoff, 
                           phi1=args.phi1, phi2=args.phi2, 
                           device=args.device).to(args.device)

    print(str(model))
    if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    t0 = time()
    if args.ae_weights is None:
        model.pretrain_autoencoder(X1=adata1.X, X_raw1=adata1.raw.X, sf1=adata1.obs.size_factors, 
                X2=adata2.X, X_raw2=adata2.raw.X, sf2=adata2.obs.size_factors, batch_size=args.batch_size, 
                epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)
    else:
        if os.path.isfile(args.ae_weights):
            print("==> loading checkpoint '{}'".format(args.ae_weights))
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'])
        else:
            print("==> no checkpoint found at '{}'".format(args.ae_weights))
            raise ValueError
        
    print('Pretraining time: %d seconds.' % int(time() - t0))

    latent1 = model.encodeBatch1(adata1.X)
    latent2 = model.encodeBatch2(adata2.X)
    pretrain_latent = torch.cat((latent1, latent2), 1)
    pretrain_latent = pretrain_latent.cpu().numpy()
    if args.n_clusters == 0:
       n_clusters = get_clusters(pretrain_latent, res=args.resolution, n=args.n_neighbors)
    else:
       print("n_cluster is defined as " + str(args.n_clusters))
       n_clusters = args.n_clusters

    y_pred, final_acc, final_ami, final_nmi, final_ari = model.fit(X1=adata1.X, X_raw1=adata1.raw.X, sf1=adata1.obs.size_factors,
                                      X2=adata2.X, X_raw2=adata2.raw.X, sf2=adata2.obs.size_factors, 
                                      y=y,
                                      n_clusters=n_clusters, 
                                      batch_size=args.batch_size, 
                                      num_epochs=args.maxiter,
                                      update_interval=args.update_interval, 
                                      tol=args.tol, 
                                      lr=args.lr, 
                                      save_dir=args.save_dir)
    
    print('Total time: %d seconds. ' % int(time() - t0))
    print("mymodel:",data_name)    
    print('Final: ACC= %.4f, NMI= %.4f, ARI= %.4f, AMI= %.4f' % (final_acc, final_nmi, final_ari, final_ami))
    pre_label_df = pd.DataFrame({'pre_label': y_pred})
    pre_label_df.to_csv(pre_label_path, index=False)
    print('pre_label:', y_pred)
    

