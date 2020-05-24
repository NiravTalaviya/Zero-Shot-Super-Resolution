# [Zero-Shot-Super-Resolution](https://arxiv.org/abs/1712.06087)
![Test Image 1](./results/set14_0.PNG)
![Test Image 2](./results/set14_1.PNG)
![Test Image 3](./results/set14_2.PNG)

## Comparison of result with Original work
### Configuration:
- SR factor = 2
- ML library: PyTorch
- Number of epochs: 1000
- Number of layers in Net: 4
- Noise Std: 0.04
- Crop size: 96
- Time spent: 40 sec (On Google Colab GPU)
- Enhanced prediction (Averaging results): None

### Result
![lincoln](./results/comparison/torch/lincoln_ep500_x2_C96_H2_N4.png)
![plane](./results/comparison/torch/plane_ep1000_x2_C96_H2_N4.png)
![bird](./results/comparison/torch/bird_ep1000_x2_C96_H2_N4.png)
![chim](./results/comparison/torch/chim_ep1000_x2_C96_H2_N4.png)
![people](./results/comparison/torch/people_ep1000_x2_C96_H2_N4.png)
![statue](./results/comparison/torch/statue_ep1000_x2_C96_H2_N4.png)
![wave](./results/comparison/torch/wave_ep1000_x2_C96_H2_N4.png)
![elephant](./results/comparison/torch/elephant_ep1000_x2_C96_H2_N4.png)
