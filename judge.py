import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet101
from PIL import Image

# 定义图像预处理，与训练时一致
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整图片大小
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])

# 加载训练好的模型
model = resnet101(pretrained=False)  # 设置为False，不使用预训练模型
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.fc.in_features, 2)
)  # 修改最后的全连接层

# 加载模型参数
model.load_state_dict(torch.load('trained_resnet101.pth'))
model.eval()  # 设置为评估模式

# 将模型移动到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义一个函数，用于对单张图像进行分类并输出概率
def predict_image(image_path):
    # 打开图像并应用预处理
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 增加一个batch维度
    image = image.to(device)

    # 进行推理
    with torch.no_grad():
        outputs = model(image)
        probabilities = nn.Softmax(dim=1)(outputs)  # 使用Softmax计算概率
        predicted_probabilities = probabilities.squeeze().cpu().numpy()  # 转换为numpy以便打印
        _, predicted_class = torch.max(outputs, 1)
    
    # 返回预测的类别和对应的概率
    return predicted_class.item(), predicted_probabilities

def clean_path(path):
    path = path.strip('"')
    path = path.replace('\\', '/')
    return path

while 1:
    # 调用分类函数并打印结果
    image_path = clean_path(input("\033[33mImage path: \033[0m"))  # 要分类的图像路径
    class_idx, probs = predict_image(image_path)

    # 打印分类结果和概率
    # print(f"预测类别: {class_idx}")
    # for i, prob in enumerate(probs):
    #     print(f"类别 {i} 概率: {prob:.4f}")

    prob_nl = probs[1]
    print(f"\033[33mProbability image containing NAILONG: \033[0m{prob_nl:.4f}")
    if prob_nl > 0.7:
        print(f"\033[31mEXECUTE\033[0m")