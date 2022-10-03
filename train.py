import torch
import torch.nn as nn


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

    train_loss = []
    val_loss = []
    best_loss = 100

    # run train and test dataset 
    for epoch in range(num_epochs):
        print('training')
        for data in train_dataloader:
            #img, _ = data
            #img = Variable(img).cpu()
            # ===================forward=====================
            particles_pos, labels, types = data
            particles_pos = particles_pos.to(device) 
            particles_pos = particles_pos.float()
            pred_pos = model(particles_pos)[0]
            loss = loss_fn(pred_pos, particles_pos)
            train_loss.append(loss.item())
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print('testing')
        for data in val_dataloader:
            # ===================forward=====================
            particles_pos, labels, types = data
            particles_pos = particles_pos.to(device) 
            particles_pos = particles_pos.float()
            pred_pos = model(particles_pos)[0]
            loss = loss_fn(pred_pos, particles_pos)
            val_loss.append(loss.item())

            # save model
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), model_save_path)
                print(f'best model saved into {model_save_path}')


            # pearson coefficient


                
        
        # ===================log========================
        print('epoch [{}/{}], train_loss:{:.4f}, test_loss:{:.4f}'.format(epoch+1, num_epochs, train_loss[-1], val_loss[-1]))
    print('finished training')
