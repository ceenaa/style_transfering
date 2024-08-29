# Neural Style Transfer using PyTorch
![target1](https://github.com/user-attachments/assets/4ba405c5-a52e-4d12-9c86-d0dfafe6a0a6)

This project implements Neural Style Transfer (NST) based on the paper ["A Neural Algorithm of Artistic Style"](https://arxiv.org/abs/1508.06576) by Gatys et al. The goal is to apply the artistic style of one image (style image) to the content of another image (content image), creating a third image (target image) that combines the content of the original image with the style of the second.

## Introduction

Neural Style Transfer is a technique to blend the style of one image with the content of another using convolutional neural networks. The key idea is to separate and recombine the content and style of images by utilizing the deep layers of a pre-trained convolutional network, specifically VGG-19 in this implementation.

## Usage

Just put your images in project directory and address it correctly in `main.ipynb`.
It should be like this.
```python
style_img = PIL.Image.open('/content/style.jpg')
content_img = PIL.Image.open('/content/content.jpg')

```

## Implementation Details

### VGG Model

The model uses a pre-trained VGG-19 network provided by PyTorch's `torchvision.models` module. The network is truncated to use the output from five specific layers to capture both content and style information:
- Layer 0: Conv1_1
- Layer 5: Conv2_1
- Layer 10: Conv3_1
- Layer 19: Conv4_1
- Layer 28: Conv5_1

```python
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.selected_features = ['0', '5', '10', '19', '28']
        self.model = torchvision.models.vgg19(pretrained=True).features

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.selected_features:
                features.append(x)
        return features
```

## Content Loss
The content loss is computed as the Mean Squared Error (MSE) between the target image and the content image, using features extracted from a specific layer of the VGG network.

```python
def get_content_loss(content, target):
    return torch.mean((content - target) ** 2)
```

## Style Loss
Style loss is calculated using the Gram matrix, which captures the correlations between different feature maps. The style loss is the MSE between the Gram matrices of the target and style images, averaged over all selected layers.

```python
def get_gram_matrix(input, channel, height, width):
    input = input.view(channel, height * width)
    G = torch.mm(input, input.t())
    return G
```
```python
def get_style_loss(target, style):
    _, channel, height, width = target.size()
    G_target = get_gram_matrix(target, channel, height, width)
    G_style = get_gram_matrix(style, channel, height, width)
    return torch.mean(((G_target - G_style) ** 2)/(channel * height * width))
```

## Training
The training loop optimizes the target image using the Adam optimizer, minimizing a weighted combination of content and style losses.

```python
optimizer = torch.optim.Adam([target_img], lr=0.01)
alpha = 1
beta = 40000
n_epochs = 1000

for epoch in range(n_epochs):
  target_feature = model(target_img)
  content_feature = model(content_img)
  style_feature = model(style_img)

  style_loss = 0
  content_loss = 0

  for target, content, style in zip(target_feature, content_feature, style_feature):
    content_loss += get_content_loss(target, content)
    style_loss += get_style_loss(target, style)

  total_loss = alpha*content_loss+beta*style_loss

  optimizer.zero_grad()

  total_loss.backward()

  optimizer.step()

```

## Examples
Here are some examples of results generated using this implementation:

![image](https://github.com/user-attachments/assets/18860eec-5e7e-4d78-8bd3-f8aa0e56886a)

![image](https://github.com/user-attachments/assets/71e2089a-63da-4ab7-803b-c7ecb491be5e)

![image](https://github.com/user-attachments/assets/345bcc9f-7f2c-4abc-84a4-458a5ad24c7d)


## References

- Gatys, L.A., Ecker, A.S., Bethge, M. (2015). [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576). arXiv:1508.06576.
