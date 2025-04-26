from PIL import Image
from torchvision import transforms

def preprocess_image(image_path, cfg):
    """
    预处理输入图片，使其符合模型的输入要求。
    """
    transform = transforms.Compose([
        transforms.Resize(cfg['data']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg['data']['mean'], std=cfg['data']['std'])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # 增加batch维度

def predict_single_image(image_path, ckpt_path, config_path='soba'):
    """
    对单张图片进行问答预测。
    """
    cfg = import_config(config_path)
    model_state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model = make_model(cfg['model'])
    model.load_state_dict(model_state_dict)
    model.cuda()
    model.eval()

    # 预处理图片
    img_tensor = preprocess_image(image_path, cfg).cuda()

    # 构造伪造的输入数据
    dummy_ret = {
        'question': [cfg['data']['dummy_question']],  # 使用一个默认问题
        'questype': ['Basic Judging'],  # 默认问题类型
        'imagen': [os.path.basename(image_path)]
    }

    with torch.no_grad():
        preds = model(img_tensor, dummy_ret)
        ques = dummy_ret['question']
        questypes = dummy_ret['questype']
        imagen = dummy_ret['imagen']

        if isinstance(ques[0], str):
            ques = [q_i + '?' for q_i in ques]
        else:
            ques = [convert2str(q_i, EarthVQADataset.QUESTION_VOC) for q_i in ques]

        ans_idx = preds.argmax(dim=1).cpu().numpy()
        for q_i_str, qt_i, ans_i, imagen_i in zip(ques, questypes, ans_idx, imagen):
            ans_i_str = convert2str(ans_i, EarthVQADataset.ANSWER_VOC)
            print(f"Image: {imagen_i}")
            print(f"Question: {q_i_str}")
            print(f"Answer: {ans_i_str}")

if __name__ == '__main__':
    # 示例：输入图片路径
    image_path = './example_image.jpg'
    predict_single_image(image_path, args.ckpt_path, args.config_path)