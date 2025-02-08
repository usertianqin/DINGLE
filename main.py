import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch_geometric
import random
from tensorboardX import SummaryWriter
from torch_geometric.utils import degree
from utils_metric import *
from utils import *
from dis_model import *
import yaml
import argparse
import time
import os
from sklearn.metrics import f1_score, classification_report, precision_score
from sklearn.metrics import confusion_matrix
from torch_geometric.utils import dropout_adj
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
import math 
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Geometer')
parser.add_argument('--config_filename', default='config/config_cora_ml_stream.yaml', type=str)
parser.add_argument('--model', default='gcn', type=str)
parser.add_argument('--cuda', default='1', type=str)
parser.add_argument('--iteration', default=1, type=int, help='times of repeated experiments')
parser.add_argument('--episodes', type=int, default=200, help='Number of episodes to train.')
parser.add_argument('--ft_episodes', type=int, default=200, help='Number of episodes to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--ft_lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--loss_ratio', type=float, default=0.3)
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--k_shot', type=int, default=5)
parser.add_argument('--memo', type=str)
parser.add_argument('--k_spt_max', type=int, default=20)
parser.add_argument('--tf', type=bool, default=True)
parser.add_argument('--loss', type=int, default=[1,1,1,1])
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--method', type=str, default='geometer')

args = parser.parse_args()
str_s = args.config_filename.find("_") + 1
str_e = args.config_filename.find(".")
dataname = args.config_filename[str_s: str_e]


    
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)



if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(args.cuda))
        print("[INFO] device: GPU")
    else:
        device = torch.device('cpu')
        print("[INFO] device: CPU")

    with open(args.config_filename) as f:
        config = yaml.safe_load(f)
    
    data_args = config['data']
    model_args = config['model']

    processed_data_dir = "./dataset/{}_stream/".format(data_args['name'])

    total_accuracy_meta_test = []
  

    dataname = data_args['name']

    pkl_path = "./model_pkl/{}/".format(dataname) 
    result_path = "./result/{}/".format(dataname)

    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    total_acc_list = []

    if args.tf:
        writer = SummaryWriter('log_dir')

    for i in range(args.iteration):
        
        node_buffer=torch.empty(0,model_args['hidden_feature']).to(device)
        acc_list = []

        start_time = time.time() 
        print("Iteration of experiment: [{}/{}]".format(i, args.iteration)) 

        encoder = get_model(args.model, model_args['in_feature'], model_args['hidden_feature']).to(device) 
        
        dis_encoder = DisModel(model_args['hidden_feature'], model_args['semantic_feature'], model_args['output_dim'], model_args['enc_layer'], model_args['dec_layer'], model_args['dis_layer'], batch_norm=False).to(device) # Define the dis model

        base_npz_filename = "{}_{}base_stream.npz".format(data_args['name'], data_args['n_base']) 
        npz_file = np.load(processed_data_dir + base_npz_filename, allow_pickle=True) 
        

        x_train, y_train, edge_index_train = torch.tensor(npz_file['base_train_x_feature']).to(device), torch.tensor(npz_file['base_train_y_label']).to(device), torch.tensor(npz_file['base_train_edge_index']).to(device) #604*2879, 604, 2*2278
        train_class_by_id = get_class_by_id(y_train) 
        train_node_num = y_train.shape[0] 

        x_val, y_val, edge_index_val = torch.tensor(npz_file['base_val_x_feature']).to(device), torch.tensor(npz_file['base_val_y_label']).to(device), torch.tensor(npz_file['base_val_edge_index']).to(device)
        val_class_by_id = get_class_by_id(y_val)
        val_node_num = y_val.shape[0] 
        

        optimizer_encoder = optim.Adam(encoder.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
        optim_generator = torch.optim.Adam([{
        'params': dis_encoder.encoder_c.parameters(),
        'params': dis_encoder.encoder_s.parameters(),
        'params': dis_encoder.decoder.parameters(),}], lr=args.lr)
        optim_discriminator = torch.optim.Adam(dis_encoder.discriminator.parameters(), lr=args.lr)
        
        base_class_num = data_args['n_base']
        class_num = data_args['n_base']
        n_novel_list = data_args['n_novel_list']
        
        if base_class_num >1:
            y_train = y_train.squeeze()
            y_val = y_val.squeeze()

        best_base_acc = -1
        
        acc_matrix = np.zeros([len(n_novel_list)+1, len(n_novel_list)+1])

        if args.start == 0: #base
            dis_filename = pkl_path + "{}_{}_{}_dis.pkl".format(0, data_args['name'], args.model)
            gnn_filename = pkl_path + "{}_{}_{}_gnn.pkl".format(0, data_args['name'], args.model)
           
            for episode in tqdm(range(args.episodes)):
                encoder.train()
                dis_encoder.train()
                
                optimizer_encoder.zero_grad() 
                optim_generator.zero_grad()
                optim_discriminator.zero_grad()


                embeddings = encoder(x_train, edge_index_train) #604,512
                
                z_c, z_s, loss_generator, loss_rec_x, loss_rec_c, loss_rec_s, loss_discriminator = dis_encoder(embeddings)
                classifier = Classifier(z_c.shape[1], base_class_num).to(device)
                y_pre= classifier(z_c)
    
                
                ce_loss = F.cross_entropy(y_pre, y_train)
                dis_loss = loss_generator+loss_rec_x+loss_rec_c+loss_rec_s+loss_discriminator
                
                loss_train = ce_loss + dis_loss
                loss_train.backward()
                optimizer_encoder.step() 
                optim_generator.step()
                optim_discriminator.step()
                
                acc_train = accuracy(y_pre, y_train)

                if episode % 5 == 0:    
                    
                    encoder.eval()
                    dis_encoder.eval()
                    with torch.no_grad():
                        val_embeddings = encoder(x_val, edge_index_val)
                        z_c, z_s, loss_generator, loss_rec_x, loss_rec_c, loss_rec_s, loss_discriminator = dis_encoder(val_embeddings)
                        y_pre= classifier(z_c)    

                    acc_val = accuracy(y_pre, y_val)

                    if acc_val > best_base_acc:
                    
                        best_base_acc = acc_val
                        best_episode = episode
                        torch.save(dis_encoder.state_dict(), dis_filename)
                        torch.save(encoder.state_dict(), gnn_filename)             
                        #Sample
                        nodes_list, nodes_embedding = get_closest_nodes(embeddings, train_class_by_id)
                         
                        
            node_buffer= torch.cat((node_buffer, nodes_embedding.detach()), dim=0)            
            print("[Base training] Best acc: {} @episode {}".format(best_base_acc, best_episode))
            acc_list.append(best_base_acc)
            
            acc_matrix[0][0] = round(best_base_acc*100,2)
            
            del x_train, y_train, edge_index_train, x_val, y_val, edge_index_val
            del encoder
            del dis_encoder      
        
        if args.start > 0:
            class_num += np.sum(np.array(n_novel_list[0: args.start]))
        
        acc_mean_list=[]
        for j in range(args.start+1, len(n_novel_list)+1):
  
            print("============ Streaming {} ============".format(j))
            if j==1:
                node_buffer_new=node_buffer

            # ============================================================
            student_dis_encoder = DisModel(model_args['hidden_feature'], model_args['semantic_feature'], model_args['output_dim'], model_args['enc_layer'], model_args['dec_layer'], model_args['dis_layer'], batch_norm=False).to(device)      
            teacher_dis_encoder = DisModel(model_args['hidden_feature'], model_args['semantic_feature'], model_args['output_dim'], model_args['enc_layer'], model_args['dec_layer'], model_args['dis_layer'], batch_norm=False).to(device)
            
            teacher_para=combine_all_params(pkl_path, j)
            teacher_dis_encoder.load_state_dict(teacher_para)
            student_dis_encoder.load_state_dict(teacher_para)
            student_encoder = get_model(args.model, model_args['in_feature'], model_args['hidden_feature']).to(device) #GAT
            

            teacher_dis_encoder.eval()

            #=============================================================       
            print("Novel class in streaming ...")
            teacher_class_num = class_num
            class_num = class_num + n_novel_list[j-1]

            print("teacher_class_num is", teacher_class_num)
            print("student class_num is", class_num)

            student_classifier = Classifier(z_c.shape[1], class_num).to(device)
            optimizer_encoder = optim.Adam(student_encoder.parameters(),
                        lr=args.lr, weight_decay=args.weight_decay)
            optim_generator = torch.optim.Adam([{
            'params': student_dis_encoder.encoder_c.parameters(),
            'params': student_dis_encoder.encoder_s.parameters(),
            'params': student_dis_encoder.decoder.parameters(),}], lr=args.lr)
            optim_discriminator = torch.optim.Adam(student_dis_encoder.discriminator.parameters(), lr=args.lr)
        

            npz_filename = "{}_{}base_{}idx_{}novel_stream.npz".format(data_args['name'], data_args['n_base'], j, n_novel_list[j-1]) 
            npz_file = np.load(processed_data_dir + npz_filename, allow_pickle=True) 


            x_st, y_st, edge_index_st = torch.tensor(npz_file['ft_x_feature']).to(device), torch.tensor(npz_file['ft_y_label']).to(device), torch.tensor(npz_file['ft_edge_index']).to(device)
            print("ft_y_label", y_st.max(), y_st.min())
            st_class_by_id = get_class_by_id(y_st)
            st_node_num = y_st.shape[0]

            st_x_test, st_y_test, st_edge_index_test = torch.tensor(npz_file['test_x_feature']).to(device), torch.tensor(npz_file['test_y_label']).to(device), torch.tensor(npz_file['test_edge_index']).to(device)
            test_node_num = st_y_test.shape[0]
            
            if class_num >1:
                y_st = y_st.squeeze()
                st_y_test = st_y_test.squeeze()


            best_st_test_acc = -1
            
            st_dis_filename = pkl_path + "{}_{}_{}_dis.pkl".format(j, data_args['name'], args.model)
            st_gnn_filename = pkl_path + "{}_{}_{}_gnn.pkl".format(j, data_args['name'], args.model)
            
            for episode in range(args.ft_episodes): 

                student_encoder.train()
                student_dis_encoder.train()
                
                optimizer_encoder.zero_grad()
                optim_generator.zero_grad()
                optim_discriminator.zero_grad()


                embeddings_st = student_encoder(x_st, edge_index_st) #
                
                z_c, z_s, loss_generator, loss_rec_x, loss_rec_c, loss_rec_s, loss_discriminator = student_dis_encoder(embeddings_st)
                
                y_pre= student_classifier(z_c)
                ce_loss = F.cross_entropy(y_pre, y_st)
                dis_loss = loss_generator+loss_rec_x+loss_rec_c+loss_rec_s+loss_discriminator
                
                
                torch.autograd.set_detect_anomaly(True)
                
                z_c_st=student_dis_encoder.encoder_c(node_buffer_new)
                z_s_st=student_dis_encoder.encoder_s(node_buffer_new)
                z_c_teacher=teacher_dis_encoder.encoder_c(node_buffer_new)
                z_s_teacher=teacher_dis_encoder.encoder_s(node_buffer_new)
                
                kd_loss = F.mse_loss(z_c_st, z_c_teacher) - F.mse_loss(z_s_st, z_s_teacher)
                
                
                loss_st = ce_loss + dis_loss + kd_loss 
                
                loss_st.backward(retain_graph=True)
                optimizer_encoder.step() 
                optim_generator.step()
                optim_discriminator.step()
                
        
                if args.tf:
                    writer.add_scalar('total_loss_'+str(j), loss_st, episode)
                    writer.add_scalar('ce_loss_'+str(j), ce_loss, episode)
                    writer.add_scalar('kd_loss_'+str(j), kd_loss, episode)
                    writer.add_scalar('dis_loss_'+str(j), dis_loss, episode)
                    writer.add_scalar('loss_discriminator_'+str(j), loss_discriminator, episode)

                student_encoder.eval()
                student_dis_encoder.eval()
                with torch.no_grad():
                    test_embeddings = student_encoder(st_x_test, st_edge_index_test)
                    test_z_c, test_z_s, test_loss_generator, test_loss_rec_x, test_loss_rec_c, test_loss_rec_s, test_loss_discriminator = student_dis_encoder(test_embeddings)
                    test_y_pre= student_classifier(test_z_c)
                acc_test = accuracy(test_y_pre, st_y_test)

                if acc_test > best_st_test_acc:
                    best_st_test_acc = acc_test
                    best_st_episode = episode
                
                    torch.save(student_dis_encoder.state_dict(), st_dis_filename)
                    torch.save(student_encoder.state_dict(), st_gnn_filename)
                    
                     #Sample
                    nodes_list, nodes_embedding = get_closest_nodes(embeddings_st, st_class_by_id)
                    
            node_buffer_new= torch.cat((node_buffer, nodes_embedding.detach()), dim=0)  
            node_buffer=node_buffer_new
           
            acc_list.append(best_st_test_acc)
            print(" best test acc after is {}, @epoch {}".format(best_st_test_acc, best_st_episode))
            acc_mean=[]
            for k in range(j+1):
                if k==0:
                    cur_label = base_class_num
                else:
                    cur_label = base_class_num + n_novel_list[k-1]*k
                output=y_pre[:,:cur_label]

                _, indices = torch.max(output, dim=1)
                class_by_id=[]
                for i in range(cur_label):
                    if str(i) in st_class_by_id.keys():
                        class_by_id.append(st_class_by_id[str(i)])
                    
                acc_per_cls = [torch.sum((indices == y_st)[ids])/len(ids) for ids in class_by_id]
                acc=sum(acc_per_cls).item()/len(acc_per_cls)
                acc_mean.append(acc)
                acc_matrix[j][k] = round(acc* 100, 2)
                if k==j:
                    acc_matrix[j][k] = round(best_st_test_acc*100,2)
            tmp_acc_mean = round(np.mean(acc_mean)*100,2)
            acc_mean_list.append(tmp_acc_mean)

            del x_st, y_st, edge_index_st, st_x_test, st_y_test, st_edge_index_test

        total_acc_list.append(acc_list) 
        end_time = time.time() 
        print("total time:", end_time-start_time)
    print("acc_matrix", acc_matrix)
    print("acc_mean_list", acc_mean_list)
    acc_mean=round(np.mean(acc_mean_list),2)
    print("AP:", acc_mean)
    backward = []
    sessions = len(n_novel_list)+1
    for t in range(sessions-1):
        b = acc_matrix[sessions-1][t]-acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward),2)
    print('AF: ', mean_backward)
    for i in range(sessions):
        print(acc_matrix[i])
    avg_acc = np.mean(np.array(total_acc_list), 0)
    std_acc = np.std(np.array(total_acc_list), 0)
    print(total_acc_list)
    print("------------------")
    print("Avg:", avg_acc * 100)
    print("STD:", std_acc * 100)
    print(args.memo)
    print("------------------")
    for i in range(len(avg_acc)):
        print("{}±{}%".format(round(avg_acc[i] * 100, 2), round(std_acc[i] * 100, 2)))
