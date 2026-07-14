import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim
from resnetModel.resnet import ResNet18
from gen_dataset.Fileutil import clearDir,queryMaxBaseNameIndex,saveFile
import shutil
#from gen_dataset.JumboBackdoor import random_troj_setting
from gen_dataset.make_templae import blended_template
from tqdm import tqdm
from gen_dataset.match1 import match,match_list
import cv2
from gen_dataset.Img import transform_convert
from PIL import Image
#from gen_dataset.TestImageSynthesis import Picture_Synthesis
import detect
import torchvision

import os
from resnetModel import utils_backdoor
import random

def rename(path):
    file_list = os.listdir(path)
    for file in file_list:
        # 补0 4表示补0后名字共4位 针对imagnet-1000足以
        filename = file.zfill(4)
        # print(filename)
        new_name = ''.join(filename)
        os.rename(path  + file, path  + new_name)


#处理数据集（默认是处理stop标签类）--目标标签中毒
#制作有毒数据集
def executeTrain(poison_number,src_label,des_label,poison_img,shape):
    #删除存放数据文件夹下的值
    save_dir='F:/yolo5/yolo5/resnetModel/trainPath/train/'
    clearDir(save_dir)

    # 每次删除记录表中数据
    record_path='F:/yolo5/yolo5/resnetModel/trainPath/mutl_test_result.txt'
    if os.path.exists(record_path):
        os.remove(record_path)

    #将CIFAR的数据移动过来
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    shutil.copytree('F:/cifar10/CIFAR10/train/',save_dir)
    #这里拷贝的都是完整数据集
    mother_img_dir='F:/cifar10/CIFAR10/train/'+str(src_label)+'/'
    #如果中毒则重新处理
    if poison_number!=0:
        #删除目标标签下面的图片
        clearDir(save_dir+str(src_label)+'/')

        backdoor_arr = []
        # for i,filename in tqdm(enumerate(os.listdir(mother_img_dir))):
        for i,filename in tqdm(enumerate(random.sample(os.listdir(mother_img_dir),len(os.listdir(mother_img_dir))))):
            mother_img=mother_img_dir+filename
            save_img=save_dir+str(src_label)+'/'+filename

            if i<poison_number:
                #指定标签
                save_img = save_dir +str(des_label)+'/' + filename
                # 海量生成插入位置
                #atk_setting = random_troj_setting(shape[0])
                
                #p_size, pattern, loc, alpha, target_y, inject_p = atk_setting

                #Picture_Synthesis(mother_img, poison_img, save_img, shape, coordinate=loc)
                mask_path = 'F:/yolo5/yolo5/gen_dataset/marks/flower_nobg.png'
                image = transforms.ToTensor()(Image.open(mother_img))
    
                #如果image的维度是2维度，那么将image转换为3维度的tensor
                if len(image.shape) == 2:
                    image = image.unsqueeze(0)
                #image三通道的情况下
                elif len(image.shape) == 3:
                    pass
                #发出警告image格式异常  
                else:
                    print("Warning: image shape is not recognized.")
                    return None

                #print(image.shape)

                #将image变为（template_size*template_size）大小
                #image = transforms.Resize((shape, shape))(image)
                atk_setting = blended_template(mask_path,a = 1,x_size = 32,temp_size = 8)
                blend_backdoor_func(image,0.5,atk_setting,save_img)
                backdoor_arr.append(save_img)
            else:
                shutil.copyfile(mother_img,save_img)
        # 将中毒的数据保存到某个目录里面
        saveFile(record_path, backdoor_arr, True)

    #改变文件名称
    rename(save_dir)

    # 将图片放入csv文件中
    csv_path = 'F:/yolo5/yolo5/resnetModel/trainPath/train.csv'
    utils_backdoor.imageToCsv(save_dir, csv_path)


# 处理数据集（默认是处理stop标签类）
def executeTest(src_label,des_label,poison_img, shape):
    # 删除存放数据文件夹下的值
    clean_dir = 'F:/yolo5/yolo5/resnetModel/trainPath/clean_test/'
    poison_dir = 'F:/yolo5/yolo5/resnetModel/trainPath/poison_test/'
    clearDir(clean_dir)
    clearDir(poison_dir)

    #中毒数据放入poison_test中
    #mother_img_dir = 'F:/cifar10/CIFAR10/test/'+str(src_label)+'/'
    mother_img_dir = "F:\\data\\bangxuez\\otrain"
    mask_path = 'F:\\Code\\BackdoorBench-main\\resource\\blended\\hello_kitty.jpeg'
    os.mkdir(poison_dir + str(des_label))
    for i, filename in tqdm(enumerate(os.listdir(mother_img_dir))):
        mother_img = mother_img_dir + filename
        save_img = 'F:/data/bangxuez/ot' + '/' + filename

        #atk_setting = random_troj_setting(shape[0])
        image = transforms.ToTensor()(Image.open(mother_img))
    
                #如果image的维度是2维度，那么将image转换为3维度的tensor
        if len(image.shape) == 2:
            image = image.unsqueeze(0)
        #image三通道的情况下
        elif len(image.shape) == 3:
            pass
        #发出警告image格式异常  
        else:
            print("Warning: image shape is not recognized.")
            return None

            #print(image.shape)

            #将image变为（template_size*template_size）大小
        #image = transforms.Resize((shape, shape))(image)
        atk_setting = blended_template(mask_path,a = 1,x_size = 28,temp_size = 7)
        blend_backdoor_func(image,1,atk_setting,save_img)
        #Picture_Synthesis(mother_img, poison_img, save_img, shape, coordinate=loc)

    # 改变文件名称
    rename(poison_dir)

    csv_path = 'F:/yolo5/yolo5/resnetModel/trainPath/poison_test.csv'
    utils_backdoor.imageToCsv(poison_dir, csv_path)

    #创建未中毒数据
    shutil.copytree(mother_img_dir, clean_dir + str(src_label)+'/')

    # 改变文件名称
    rename(clean_dir)

    csv_path = 'F:/yolo5/yolo5/resnetModel/trainPath/clean_test.csv'
    utils_backdoor.imageToCsv(clean_dir, csv_path)



#加载数据
def load_dataset(train_dir='F:/yolo5/yolo5/resnetModel/trainPath/train/',
                 poison_test_dir='F:/yolo5/yolo5/resnetModel/trainPath/poison_test/',
                 clean_test_dir='F:/yolo5/yolo5/resnetModel/trainPath/clean_test/',
                 batch_size=128):
    # 加载图像批次为32
    batchsz = batch_size
    # 训练集测试集加载
    #这里还有一个调整图片大小的东西
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    trainset = datasets.ImageFolder(root=train_dir,
                                    transform=transform)

    poison_testset = datasets.ImageFolder(root=poison_test_dir,
                                   transform=transform)

    clean_testset = datasets.ImageFolder(root=clean_test_dir,
                                          transform=transform)

    train_loader = DataLoader(trainset, batch_size=batchsz, shuffle=True)

    posion_test_loader = DataLoader(poison_testset, batch_size=batchsz, shuffle=True)

    clean_test_loader = DataLoader(clean_testset, batch_size=batchsz, shuffle=True)

    return train_loader,posion_test_loader,clean_test_loader


def train(model,train_loader,device,epoch,record_path):
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for batchidx, (x, label) in enumerate(train_loader):
        # [b, 3, 32, 32]
        # [b]
        x, label = x.to(device), label.to(device)

        logits = model(x)
        # logits: [b, 10]
        # label:  [b]
        # loss: tensor scalar
        loss = criteon(logits, label)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    message='【{}】 loss : 【{}】'.format(epoch,loss.item())
    print(message)
    saveFile(record_path,message,False)

def test(model,test_loader,device,epoch,record_path,src_label, des_label,poison_flag=False):
    # 模型测试
    model.eval()
    with torch.no_grad():
        # test
        total_correct = 0
        total_num = 0
        for x, label in test_loader:

            change_label=torch.empty_like(label)
            change_label.data *= 0
            if poison_flag:
                change_label.data += des_label
                print('des:',des_label)
            else:
                change_label.data += src_label #需要手动修改(待修改)
                print('src:',src_label)
            label=change_label

            # [b, 3, 32, 32]
            # [b]
            x, label = x.to(device), label.to(device)

            # [b, 10]
            logits = model(x)
            # [b]
            pred = logits.argmax(dim=1)
            print('pred',pred)
            # [b] vs [b] => scalar tensor
            correct = torch.eq(pred, label).float().sum().item()
            print('corrent',correct)
            total_correct += correct
            total_num += x.size(0)
            # print(correct)

        acc = total_correct / total_num

        if poison_flag:
            message='【{}】 poison test acc:【{}/{}】->{}'.format(epoch,total_correct,total_num,acc)
        else:
            message = '【{}】 clean test acc:【{}/{}】->{}'.format(epoch, total_correct, total_num, acc)

        print(message)
        #保存记录到目录下的记录文件中
        saveFile(record_path,message,False)


def main(epoch_number,poison_img_path,src_label, des_label):

    #加载数据集
    train_loader,posion_test_loader,clean_test_loader=load_dataset()

    #加载图像及标签
    x, label = iter(train_loader).next()
    print('x:', x.shape, 'label:', label.shape)

    #加载模型
    device = torch.device('cuda')
    model = ResNet18().to(device)

    # criteon = nn.CrossEntropyLoss().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # print(model)

    model_save='F:/yolo5/yolo5/resnetModel/trainPath/model/'
    pattern='blended'
    max_index = queryMaxBaseNameIndex(model_save, pattern)
    model_dir=model_save+pattern+str(max_index)
    os.mkdir(model_dir)

    #将backdoor图片放入结果Model文件中
    shutil.copy(poison_img_path,model_dir+'/'+os.path.basename(poison_img_path))

    record_path = model_dir + '/epoch_result' + '.txt'
    for epoch in range(epoch_number):

        #模型训练
        train(model,train_loader,device,epoch,record_path)

        # 原始性能
        test(model, clean_test_loader, device, epoch,record_path,src_label, des_label, poison_flag=False)

        #中毒攻击率
        test(model,posion_test_loader,device,epoch,record_path,src_label, des_label,poison_flag=True)

        #保存模型
        # model_save_path=model_dir+'/epoch'+str(epoch)+'.pth'
        #每一次覆盖前面的
        model_save_path = model_dir + '/epoch.pth'
        file=open(model_save_path, 'w')
        file.close()
        torch.save(model,model_save_path) #保存troch


def execute(src_label, des_label,poison_dir,poison_number,epoch_number,shape=(32,32),single_flag=True):
    if single_flag:
        # 生成训练数据
        executeTrain(poison_number, src_label, des_label, poison_dir, shape)

        # 生成测试数据
        executeTest(src_label, des_label, poison_dir, shape)

        # 可以开始训练了
        main(epoch_number, poison_dir,src_label, des_label)
    else:
        #循环获取中毒Jumbo图片

        for poison_img_name in tqdm(os.listdir(poison_dir)):
            #获取绝对路径
            poison_img_path=poison_dir+poison_img_name

            #生成训练数据
            executeTrain(poison_number, src_label, des_label, poison_img_path, shape)

            #生成测试数据
            executeTest(src_label, des_label, poison_img_path, shape)

            #可以开始训练了
            main(epoch_number,poison_img_path)
            print('没跑这个吧')

def reexecute(src_label, des_label,poison_dir,poison_number,epoch_number,shape=(32,32),single_flag=True):
    if single_flag:
        # 生成训练数据
        #executeTrain(poison_number, src_label, des_label, poison_dir, shape)

        # 生成测试数据
        #executeTest(src_label, des_label, poison_dir, shape)

        # 可以开始训练了
        main(epoch_number, poison_dir,src_label, des_label)
        
        
def delete_image(delete_path):
    # 检查文件是否存在
    if os.path.exists(delete_path):
        try:
            # 删除文件
            os.remove(delete_path)
            print(f"文件 {delete_path} 已成功删除。")
        except Exception as e:
            print(f"删除文件 {delete_path} 时发生错误：{e}")
    else:
        print(f"警告：文件 {delete_path} 不存在。")
#################################################################################################################################
def blend_backdoor_func(X, y, atk_setting,save_img):
    p_size, pattern, loc, alpha, target_y, inject_p = atk_setting

    w, h = loc
    X_new = X.clone()
    X_new[:, w:w + p_size, h:h + p_size] = alpha * pattern + (1 - alpha) * X_new[:, w:w + p_size,
                                                                              h:h + p_size]

    #X_new_new=X.clone();
    #X_new_new[0, w:w + p_size, h:h + p_size]=alpha * torch.FloatTensor(pattern)
    y_new = target_y

    ToTensor_transform = transforms.Compose([transforms.ToTensor()])
    #以前的

    #生成时间地址
    # src_save_jpg = os.path.dirname('../mnist/bind.png') + "/" + time_now  +'-src-'+str(idx)+'.jpg'
    src_save_jpg = save_img

    img=transform_convert(X_new,ToTensor_transform)
    # plt.imshow(img)
    cv2.imwrite(src_save_jpg, img)
    # plt.show()

    #shape=X.shape
    #X1=torch.zeros(X.shape)
    # print(type(X_new_new_new))
    #X1[0, w:w + p_size, h:h + p_size] = alpha * torch.FloatTensor(pattern) + (1 - alpha) * X_new[0, w:w + p_size,
    #                                                                          h:h + p_size]
    #backdoor图片
    #src_save_jpg = "F:yolo5/yolo5/data/JPEGImages/" + time_now + '-srctest-' + str(idx) + '.jpg'
    #img = transform_convert(X1, ToTensor_transform)
    #cv2.imwrite(src_save_jpg,img)
    # plt.show()

    # 模板图片(backdoor图片)
    # tem_save_jpg = "D:/deephash_original/data/templates/" + time_now + '-tem-' + str(idx) + '.jpg'
    # img = transform_convert(X_new[:, w:w + p_size, h:h + p_size], ToTensor_transform)
    # cv2.imwrite(tem_save_jpg, img)
    # plt.show()

    #保存匹配后的图片（匹配操作）
    # save_jpg = "D:/deephash_original/data/result/" + time_now + '-contrast-' + str(idx) + '.jpg'
    # match(src_save_jpg,tem_save_jpg,save_jpg)

    return X_new, y_new

if __name__ == '__main__':
    #处理训练集
    src_label=1
    des_label =14
    # poison_img='D:/deephash_original/data/templatesTest/Jumbo22.jpg'
    poison_img = 'D:/deephash_original/data/templatesTest/Jumbo28.jpg'
    shape=(32,32)

    poison_number = 117

    # executeTrain(poison_number,label,poison_img,shape)

    # executeTest(label,poison_img,shape)

    #单个随机选择攻击图像
    poison_dir = 'D:/deephash_original/data/templatesTest/Jumbo50.jpg'
    # poison_dir = 'D:/deephash_original/data/templates/Jumbo1.jpg' #9X9
    # poison_dir = '/home/yczhang1/ff/repository/nc/bakdoor_b.jpg'
    epoch_number=1000
    # 生成训练数据和测试数据
    execute(src_label, des_label, poison_dir, poison_number, epoch_number, shape, single_flag=True)
    #开始训练
    # main(epoch_number,poison_dir,src_label, des_label)

    # 训练单个Jumbo+训练
    # poison_dir = 'D:/deephash_original/data/templatesTest/Jumbo10.jpg'

    #循环训练文件夹下的Jumbo+训练(用于选取攻击率高的图像)
    # poison_dir='D:/deephash_original/data/templatesTest/'
    # execute(src_label, des_label, poison_dir, poison_number, epoch_number,shape)


import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim
#from resnetModel.resnet_gtsrb import ResNet34,ResNet18
from resnetModel.resnet import ResNet18
from gen_dataset.Fileutil import clearDir,queryMaxBaseNameIndex,saveFile
import shutil
#from gen_dataset.JumboBackdoor import random_troj_setting
from resnetModel.make_template2 import blended_template
from tqdm import tqdm
from resnetModel.match2 import match,match_list
import cv2
from gen_dataset.Img import transform_convert
from PIL import Image
#from gen_dataset.TestImageSynthesis import Picture_Synthesis
import detect
import torchvision

import os
from resnetModel import utils_backdoor
import random

def rename(path):
    file_list = os.listdir(path)
    for file in file_list:
        # 补0 4表示补0后名字共4位 针对imagnet-1000足以
        filename = file.zfill(4)
        # print(filename)
        new_name = ''.join(filename)
        os.rename(path  + file, path  + new_name)


#处理数据集（默认是处理stop标签类）--目标标签中毒
#制作有毒数据集
def executeTrain(poison_number,src_label,des_label,poison_img,shape):
    #删除存放数据文件夹下的值
    save_dir='F:/yolo5/yolo5/resnetModel/trainPath_badnets/train/'
    clearDir(save_dir)

    # 每次删除记录表中数据
    record_path='F:/yolo5/yolo5/resnetModel/trainPath_badnets/mutl_test_result.txt'
    if os.path.exists(record_path):
        os.remove(record_path)

    #将CIFAR的数据移动过来
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    shutil.copytree('F:/cifar10/CIFAR10/train/',save_dir)
    #这里拷贝的都是完整数据集
    mother_img_dir='F:/cifar10/CIFAR10/train/'+str(src_label)+'/'
    #如果中毒则重新处理
    if poison_number!=0:
        #删除目标标签下面的图片
        clearDir(save_dir+str(src_label)+'/')

        backdoor_arr = []
        # for i,filename in tqdm(enumerate(os.listdir(mother_img_dir))):
        for i,filename in tqdm(enumerate(random.sample(os.listdir(mother_img_dir),len(os.listdir(mother_img_dir))))):
            mother_img=mother_img_dir+filename
            save_img=save_dir+str(src_label)+'/'+filename

            if i<poison_number:
                #指定标签
                save_img = save_dir +str(des_label)+'/' + filename
                # 海量生成插入位置
                #atk_setting = random_troj_setting(shape[0])
                
                #p_size, pattern, loc, alpha, target_y, inject_p = atk_setting

                #Picture_Synthesis(mother_img, poison_img, save_img, shape, coordinate=loc)
                mask_path = 'F:/yolo5/yolo5/resnetModel/trainPath_blended/hello_kitty.jpeg'
                image = transforms.ToTensor()(Image.open(mother_img))
    
                #如果image的维度是2维度，那么将image转换为3维度的tensor
                if len(image.shape) == 2:
                    image = image.unsqueeze(0)
                #image三通道的情况下
                elif len(image.shape) == 3:
                    pass
                #发出警告image格式异常  
                else:
                    print("Warning: image shape is not recognized.")
                    return None

                #print(image.shape)

                #将image变为（template_size*template_size）大小
                #image = transforms.Resize((shape, shape))(image)
                atk_setting = blended_template(mask_path,a = 0.2,x_size = 32,temp_size = 28)
                blend_backdoor_func(image,0.5,atk_setting,save_img)
                backdoor_arr.append(save_img)
            else:
                shutil.copyfile(mother_img,save_img)
        # 将中毒的数据保存到某个目录里面
        saveFile(record_path, backdoor_arr, True)

    #改变文件名称
    rename(save_dir)

    # 将图片放入csv文件中
    csv_path = 'F:/yolo5/yolo5/resnetModel/trainPath_blended/train.csv'
    utils_backdoor.imageToCsv(save_dir, csv_path)


# 处理数据集（默认是处理stop标签类）
def executeTest(src_label,des_label,poison_img, shape):
    # 删除存放数据文件夹下的值
    clean_dir = 'F:/yolo5/yolo5/resnetModel/trainPath_blended/clean_test/'
    poison_dir = 'F:/yolo5/yolo5/resnetModel/trainPath_blended/poison_test/'
    clearDir(clean_dir)
    clearDir(poison_dir)

    #中毒数据放入poison_test中
    mother_img_dir = 'F:/cifar10/CIFAR10/test/'+str(src_label)+'/'
    mask_path = 'F:/yolo5/yolo5/gen_dataset/marks/flower_nobg.png'
    os.mkdir(poison_dir + str(des_label))
    for i, filename in tqdm(enumerate(os.listdir(mother_img_dir))):
        mother_img = mother_img_dir + filename
        save_img = poison_dir  + str(des_label) + '/' + filename

        #atk_setting = random_troj_setting(shape[0])
        image = transforms.ToTensor()(Image.open(mother_img))
    
                #如果image的维度是2维度，那么将image转换为3维度的tensor
        if len(image.shape) == 2:
            image = image.unsqueeze(0)
        #image三通道的情况下
        elif len(image.shape) == 3:
            pass
        #发出警告image格式异常  
        else:
            print("Warning: image shape is not recognized.")
            return None

            #print(image.shape)

            #将image变为（template_size*template_size）大小
        #image = transforms.Resize((shape, shape))(image)
        atk_setting = blended_template(mask_path,a = 0.2,x_size = 32,temp_size = 28)
        blend_backdoor_func(image,0.5,atk_setting,save_img)
        #Picture_Synthesis(mother_img, poison_img, save_img, shape, coordinate=loc)

    # 改变文件名称
    rename(poison_dir)

    csv_path = 'F:/yolo5/yolo5/resnetModel/trainPath_blended/poison_test.csv'
    utils_backdoor.imageToCsv(poison_dir, csv_path)

    #创建未中毒数据
    shutil.copytree(mother_img_dir, clean_dir + str(src_label)+'/')

    # 改变文件名称
    rename(clean_dir)

    csv_path = 'F:/yolo5/yolo5/resnetModel/trainPath_blended/clean_test.csv'
    utils_backdoor.imageToCsv(clean_dir, csv_path)



#加载数据
def load_dataset(train_dir='F:/yolo5/yolo5/resnetModel/trainPath_blended/train/',
                 poison_test_dir='F:/yolo5/yolo5/resnetModel/trainPath_blended/poison_test/',
                 clean_test_dir='F:/yolo5/yolo5/resnetModel/trainPath_blended/clean_test/',
                 batch_size=128):
    # 加载图像批次为32
    batchsz = batch_size

    # 定义数据预处理操作
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),  # 调整图片大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),  # 调整图片大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])
    

    # 训练集加载
    trainset = datasets.ImageFolder(root=train_dir, transform=transform_train)
    
    # 中毒测试集加载
    poison_testset = datasets.ImageFolder(root=poison_test_dir, transform=transform_test)
    
    # 干净测试集加载
    clean_testset = datasets.ImageFolder(root=clean_test_dir, transform=transform_test)

    # 数据加载器
    train_loader = DataLoader(trainset, batch_size=batchsz, shuffle=True)
    poison_test_loader = DataLoader(poison_testset, batch_size=batchsz, shuffle=True)
    clean_test_loader = DataLoader(clean_testset, batch_size=batchsz, shuffle=True)

    return train_loader,poison_test_loader,clean_test_loader


def train(model,train_loader,device,epoch,record_path):
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for batchidx, (x, label) in enumerate(train_loader):
        # [b, 3, 32, 32]
        # [b]
        x, label = x.to(device), label.to(device)

        logits = model(x)
        # logits: [b, 10]
        # label:  [b]
        # loss: tensor scalar
        loss = criteon(logits, label)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    message='【{}】 loss : 【{}】'.format(epoch,loss.item())
    print(message)
    saveFile(record_path,message,False)

def test(model,test_loader,device,epoch,record_path,src_label, des_label,poison_flag=False):
    # 模型测试
    model.eval()
    with torch.no_grad():
        # test
        total_correct = 0
        total_num = 0
        for x, label in test_loader:

            change_label=torch.empty_like(label)
            change_label.data *= 0
            if poison_flag:
                change_label.data += des_label
                #print('des:',des_label)
            else:
                change_label.data += src_label #需要手动修改(待修改)
                #print('src:',src_label)
            label=change_label

            # [b, 3, 32, 32]
            # [b]
            x, label = x.to(device), label.to(device)

            # [b, 10]
            logits = model(x)
            # [b]
            pred = logits.argmax(dim=1)
            #print('pred',pred)
            # [b] vs [b] => scalar tensor
            correct = torch.eq(pred, label).float().sum().item()
           # print('corrent',correct)
            total_correct += correct
            total_num += x.size(0)
            # print(correct)

        acc = total_correct / total_num

        if poison_flag:
            message='【{}】 poison test acc:【{}/{}】->{}'.format(epoch,total_correct,total_num,acc)
        else:
            message = '【{}】 clean test acc:【{}/{}】->{}'.format(epoch, total_correct, total_num, acc)

        print(message)
        #保存记录到目录下的记录文件中
        saveFile(record_path,message,False)


def main(epoch_number,poison_img_path,src_label, des_label):

    #加载数据集
    train_loader,posion_test_loader,clean_test_loader=load_dataset()

    #加载图像及标签
    x, label = iter(train_loader).next()
    print('x:', x.shape, 'label:', label.shape)

    #加载模型
    device = torch.device('cuda')
    model = ResNet18().to(device)

    # criteon = nn.CrossEntropyLoss().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # print(model)

    model_save='F:/yolo5/yolo5/resnetModel/trainPath_blended/model/'
    pattern='badnets'
    max_index = queryMaxBaseNameIndex(model_save, pattern)
    model_dir=model_save+pattern+str(max_index)
    os.mkdir(model_dir)

    #将backdoor图片放入结果Model文件中
    shutil.copy(poison_img_path,model_dir+'/'+os.path.basename(poison_img_path))

    record_path = model_dir + '/epoch_result' + '.txt'
    for epoch in range(epoch_number):

        #模型训练
        train(model,train_loader,device,epoch,record_path)

        # 原始性能
        test(model, clean_test_loader, device, epoch,record_path,src_label, des_label, poison_flag=False)

        #中毒攻击率
        test(model,posion_test_loader,device,epoch,record_path,src_label, des_label,poison_flag=True)

        #保存模型
        # model_save_path=model_dir+'/epoch'+str(epoch)+'.pth'
        #每一次覆盖前面的
        model_save_path = model_dir + '/epoch.pth'
        file=open(model_save_path, 'w')
        file.close()
        torch.save(model,model_save_path) #保存troch


def execute(src_label, des_label,poison_dir,poison_number,epoch_number,shape=(32,32),single_flag=True):
    if single_flag:
        # 生成训练数据
        executeTrain(poison_number, src_label, des_label, poison_dir, shape)

        # 生成测试数据
        executeTest(src_label, des_label, poison_dir, shape)

        # 可以开始训练了
        main(epoch_number, poison_dir,src_label, des_label)
    else:
        #循环获取中毒Jumbo图片

        for poison_img_name in tqdm(os.listdir(poison_dir)):
            #获取绝对路径
            poison_img_path=poison_dir+poison_img_name

            #生成训练数据
            executeTrain(poison_number, src_label, des_label, poison_img_path, shape)

            #生成测试数据
            executeTest(src_label, des_label, poison_img_path, shape)

            #可以开始训练了
            main(epoch_number,poison_img_path)
            print('没跑这个吧')

def reexecute(src_label, des_label,poison_dir,poison_number,epoch_number,shape=(32,32),single_flag=True):
    if single_flag:
        # 生成训练数据
        #executeTrain(poison_number, src_label, des_label, poison_dir, shape)

        # 生成测试数据
        #executeTest(src_label, des_label, poison_dir, shape)

        # 可以开始训练了
        main(epoch_number, poison_dir,src_label, des_label)
        
        
def delete_image(delete_path):
    # 检查文件是否存在
    if os.path.exists(delete_path):
        try:
            # 删除文件
            os.remove(delete_path)
            print(f"文件 {delete_path} 已成功删除。")
        except Exception as e:
            print(f"删除文件 {delete_path} 时发生错误：{e}")
    else:
        print(f"警告：文件 {delete_path} 不存在。")
#################################################################################################################################
def blend_backdoor_func(X, y, atk_setting,save_img):
    p_size, pattern, loc, alpha, target_y, inject_p = atk_setting
    loc = (0,0)
    w, h = loc
    X_new = X.clone()
    # a = 0
    # if (a<1):
    #     print(X_new.shape)
    #     a = 1
    
    X_new[:, w:w + p_size, h:h + p_size] = alpha * pattern + (1 - alpha) * X_new[:, w:w + p_size,
                                                                              h:h + p_size]

    #X_new_new=X.clone();
    #X_new_new[0, w:w + p_size, h:h + p_size]=alpha * torch.FloatTensor(pattern)
    y_new = target_y

    ToTensor_transform = transforms.Compose([transforms.ToTensor()])
    #以前的

    #生成时间地址
    # src_save_jpg = os.path.dirname('../mnist/bind.png') + "/" + time_now  +'-src-'+str(idx)+'.jpg'
    src_save_jpg = save_img

    img=transform_convert(X_new,ToTensor_transform)
    # plt.imshow(img)
    cv2.imwrite(src_save_jpg, img)
    # plt.show()

    #shape=X.shape
    #X1=torch.zeros(X.shape)
    # print(type(X_new_new_new))
    #X1[0, w:w + p_size, h:h + p_size] = alpha * torch.FloatTensor(pattern) + (1 - alpha) * X_new[0, w:w + p_size,
    #                                                                          h:h + p_size]
    #backdoor图片
    #src_save_jpg = "F:yolo5/yolo5/data/JPEGImages/" + time_now + '-srctest-' + str(idx) + '.jpg'
    #img = transform_convert(X1, ToTensor_transform)
    #cv2.imwrite(src_save_jpg,img)
    # plt.show()

    # 模板图片(backdoor图片)
    # tem_save_jpg = "D:/deephash_original/data/templates/" + time_now + '-tem-' + str(idx) + '.jpg'
    # img = transform_convert(X_new[:, w:w + p_size, h:h + p_size], ToTensor_transform)
    # cv2.imwrite(tem_save_jpg, img)
    # plt.show()

    #保存匹配后的图片（匹配操作）
    # save_jpg = "D:/deephash_original/data/result/" + time_now + '-contrast-' + str(idx) + '.jpg'
    # match(src_save_jpg,tem_save_jpg,save_jpg)

    return X_new, y_new

if __name__ == '__main__':
    #处理训练集
    src_label=1
    des_label =14
    # poison_img='D:/deephash_original/data/templatesTest/Jumbo22.jpg'
    poison_img = 'D:/deephash_original/data/templatesTest/Jumbo28.jpg'
    shape=(32,32)

    poison_number = 117

    # executeTrain(poison_number,label,poison_img,shape)

    # executeTest(label,poison_img,shape)

    #单个随机选择攻击图像
    poison_dir = 'D:/deephash_original/data/templatesTest/Jumbo50.jpg'
    # poison_dir = 'D:/deephash_original/data/templates/Jumbo1.jpg' #9X9
    # poison_dir = '/home/yczhang1/ff/repository/nc/bakdoor_b.jpg'
    epoch_number=1000
    # 生成训练数据和测试数据
    execute(src_label, des_label, poison_dir, poison_number, epoch_number, shape, single_flag=True)
    #开始训练
    # main(epoch_number,poison_dir,src_label, des_label)

    # 训练单个Jumbo+训练
    # poison_dir = 'D:/deephash_original/data/templatesTest/Jumbo10.jpg'

    #循环训练文件夹下的Jumbo+训练(用于选取攻击率高的图像)
    # poison_dir='D:/deephash_original/data/templatesTest/'
    # execute(src_label, des_label, poison_dir, poison_number, epoch_number,shape)






