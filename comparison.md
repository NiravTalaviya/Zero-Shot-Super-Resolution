### Our Configuration
- SR factor = 2
- ML library: PyTorch
- Number of epochs: 1000
- Number of layers in Net: 4
- Noise Std: 0.04
- Crop size: 96
- Time spent: 40 sec (On Google Colab GPU)
- Enhanced prediction (Averaging results): None

### Compare (PSNR)
|ZSSR|Ours|
|--|--|
|![plane](./results/comparison/shocher/plane.png)|![lincoln](./results/comparison/results-torch/plane_ep1000_x2_C96_H2_N4.png)
|35.57|36.01|

|ZSSR|Ours|
|--|--|
|![bird](./results/comparison/shocher/bird.png)|![lincoln](./results/comparison/results-torch/bird_ep1000_x2_C96_H2_N4.png)
|29.42|29.68|

|ZSSR|Ours|
|--|--|
|![chim](./results/comparison/shocher/chim.png)|![lincoln](./results/comparison/results-torch/chim_ep1000_x2_C96_H2_N4.png)
|23.71|24.27|

|ZSSR|Ours|
|--|--|
|![people](./results/comparison/shocher/people.png)|![lincoln](./results/comparison/results-torch/people_ep1000_x2_C96_H2_N4.png)
|24.57|24.39|

|ZSSR|Ours|
|--|--|
|![statue](./results/comparison/shocher/statue.png)|![lincoln](./results/comparison/results-torch/statue_ep1000_x2_C96_H2_N4.png)
|28.65|27.41|

|ZSSR|Ours|
|--|--|
|![wave](./results/comparison/shocher/wave.png)|![lincoln](./results/comparison/results-torch/wave_ep1000_x2_C96_H2_N4.png)
|27.56|27.39|

|ZSSR|Ours|
|--|--|
|![elephant](./results/comparison/shocher/elephant.png)|![lincoln](./results/comparison/results-torch/elephant_ep1000_x2_C96_H2_N4.png)
|26.43|27.98|

|ZSSR|Ours|
|--|--|
|![lincoln](./results/comparison/shocher/lincoln.png)|![lincoln](./results/comparison/results-torch/lincoln_ep500_x2_C96_H2_N4.png)

|ZSSR|Ours|
|--|--|
|![lincoln](./results/comparison/shocher/kennedy.png)|![lincoln](./results/comparison/results-torch/kennedy_ep1000_x2_C96_H2_N4.png)
