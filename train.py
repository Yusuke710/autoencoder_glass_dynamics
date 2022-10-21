import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# particle coordinate
def train(train_dataloader, val_dataloader, model, num_epochs, lr, weight_decay, seed, model_load_path, model_save_path):
    # setting 
    torch.manual_seed(seed)         
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    
    if model_load_path is not None:
        model.load_state_dict(torch.load(model_load_path))
        print(f'loaded {model_load_path}')
    model = model.to(device)
    model = model.float()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)


    best_loss = 10000

    fm = 'log/loss.dat'
    with open(fm, 'w') as f:
        # run train and test dataset 
        for epoch in range(num_epochs):
            train_loss = []
            val_loss = []
            print('training')
            for data in train_dataloader:
                #img, _ = data
                #img = Variable(img).cpu()
                # ===================forward=====================
                image, labels, types = data
                image = image.to(device) 
                image = image.float()
                pred_pos = model(image)[0]
                loss = loss_fn(pred_pos, image)
                #print(loss.item())
                train_loss.append(loss.item())
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print('testing')
            for data in val_dataloader:
                # ===================forward=====================
                image, labels, types = data
                image = image.to(device) 
                image = image.float()
                pred_pos = model(image)[0]
                loss = loss_fn(pred_pos, image)
                #print(loss.item())
                val_loss.append(loss.item())

            # save model
            if np.mean(val_loss) < best_loss:
                best_loss = np.mean(val_loss)
                torch.save(model.state_dict(), model_save_path)
                print(f'best model saved into {model_save_path}')


                # pearson coefficient


                    
            
            # ===================log========================
            print('epoch [{}/{}], train_loss:{:.4f}, val_loss:{:.4f}'.format(epoch+1, num_epochs, np.mean(train_loss), np.mean(val_loss)))
            string = 'epoch [{}/{}], train_loss:{:.4f}, val_loss:{:.4f} \n'.format(epoch+1, num_epochs, np.mean(train_loss), np.mean(val_loss))
            f.write(string)

    print('finished training')


# simulated images
def train_img(train_dataloader, val_dataloader, model, num_epochs, lr, weight_decay, seed, model_load_path, model_save_path):
    # setting 
    torch.manual_seed(seed)         
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    
    if model_load_path is not None:
        model.load_state_dict(torch.load(model_load_path))
        print(f'loaded {model_load_path}')
    model = model.to(device)
    model = model.float()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)


    best_loss = 10000

    fm = 'log/loss.dat'
    with open(fm, 'w') as f:
        # run train and test dataset 
        for epoch in range(num_epochs):
            train_loss = []
            val_loss = []
            print('training')
            for data in train_dataloader:
                #img, _ = data
                #img = Variable(img).cpu()
                # ===================forward=====================
                image, labels = data
                image = image.to(device) 
                image = image.float()
                pred_pos = model(image)[0]
                loss = loss_fn(pred_pos, image)
                #print(loss.item())
                train_loss.append(loss.item())
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print('testing')
            for data in val_dataloader:
                # ===================forward=====================
                image, labels = data
                image = image.to(device) 
                image = image.float()
                pred_pos = model(image)[0]
                # save output
                #print('model out: {}'.format(pred_pos.shape))
                #image = np.transpose(pred_pos, (0, 2, 3, 1))
                #plt.imshow(image.detach().numpy()) # needs to be 512, 512, 3
                #plt.savefig('imoutput.png')
                loss = loss_fn(pred_pos, image)
                val_loss.append(loss.item())

            # save model
            if np.mean(val_loss) < best_loss:
                best_loss = np.mean(val_loss)
                torch.save(model.state_dict(), model_save_path)
                print(f'best model saved into {model_save_path}')


                # pearson coefficient


                    
            
            # ===================log========================
            print('epoch [{}/{}], train_loss:{:.4f}, val_loss:{:.4f}'.format(epoch+1, num_epochs, np.mean(train_loss), np.mean(val_loss)))
            string = 'epoch [{}/{}], train_loss:{:.4f}, val_loss:{:.4f} \n'.format(epoch+1, num_epochs, np.mean(train_loss), np.mean(val_loss))
            f.write(string)

    print('finished training')
