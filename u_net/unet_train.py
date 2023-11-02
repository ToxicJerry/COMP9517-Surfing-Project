from read_dataset import *
from unet_model import *

device = "cuda" if torch.cuda.is_available() else "cpu"

image_dir = 'D:/PycharmProjects/COMP9517-Surfing-Project/u_net/part_images'
mask_dir = 'D:/PycharmProjects/COMP9517-Surfing-Project/u_net/dask_images/'
dataloader = SegmentationDataset(image_dir, mask_dir)  # 数据读取

train_loader = DataLoader(dataloader, batch_size=8, shuffle=False)
# print("样本数量:", dataloader.num_of_samples(), len(dataloader), train_loader.dataset)
if __name__ == '__main__':
    index = 0
    num_epochs = 10
    train_on_gpu = True
    unet = Unet().to(device)  # Uet网络
    optimizer = torch.optim.SGD(unet.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(num_epochs):
        train_loss = 0.0
        for i_batch, sample_batched in enumerate(train_loader):
            images_batch, target_labels = sample_batched['image'], sample_batched['mask']
            # print(target_labels.min())#tensor(0, dtype=torch.uint8)
            # print(target_labels.max())#tensor(1, dtype=torch.uint8)

            if train_on_gpu:
                images_batch, target_labels = images_batch.to(device), target_labels.to(device)
                # images_batch, target_labels = images_batch.cuda(), target_labels.cuda()
            optimizer.zero_grad()

            """forward pass: compute predicted outputs by passing inputs to the model"""
            print("输入样本的形状:",images_batch.shape)#输入样本的形状: torch.Size([8, 1, 320, 480])
            m_label_out_ = unet(images_batch)
            print(m_label_out_.shape)#torch.Size([8, 2, 320, 480])
            # calculate the batch loss
            target_labels = target_labels.contiguous().view(-1)  # 执行contiguous()这个函数，把tensor变成在内存中连续分布的形式
            m_label_out_ = m_label_out_.transpose(1, 3).transpose(1, 2).contiguous().view(-1, 2)
            target_labels = target_labels.long()
            print(target_labels.shape)
            loss = torch.nn.functional.cross_entropy(m_label_out_, target_labels)
            print(loss)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()

            # update training loss
            train_loss += loss.item()
            if index % 100 == 0:
                print('step: {} \tcurrent Loss: {:.6f} '.format(index, loss.item()))
            index += 1
            # test(unet)
        # 计算平均损失
        train_loss = train_loss / dataloader.num_of_samples()
        # 显示训练集与验证集的损失函数
        print('Epoch: {} \tTraining Loss: {:.6f} '.format(epoch, train_loss))
        # test(unet)
    # save model
    unet.eval()
    torch.save(unet.state_dict(), 'D:/PycharmProjects/COMP9517-Surfing-Project/u_net/save_model/unet_road_model.pkl')
    torch.save(unet.state_dict(), 'D:/PycharmProjects/COMP9517-Surfing-Project/u_net/save_model/unet_road_model.pt')