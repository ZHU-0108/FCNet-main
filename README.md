<div align="center">
<h1>FCNet</h1>
<h3>Fully Complex Network for Time Series Forecasting</h3>

[Xuanbing Zhu](https://github.com/ZHU-0108/FCNet-main)<sup>1</sup> , [Dunbin Shen](https://scholar.google.com/citations?user=DH4VSLMAAAAJ&hl=zh-CN)<sup>1</sup> ,Yingguang Hao<sup>1</sup> , [Hongyu Wang](http://faculty.dlut.edu.cn/MMCL_WHY/zh_CN/)<sup>1 :email:</sup>

<sup>1</sup>  Dalian University of Technology

(<sup>:email:</sup>) Corresponding author.

(https://ieeexplore.ieee.org/document/10648823/))


</div>


## Abstract
Time series forecasting has extensive applications in domains such as energy, traffic, and weather prediction. Currently, existing literature has designed many architectures that combine deep learning models in the frequency domain, and effective results have been achieved. However, handling complex-valued arithmetic poses a challenge for most frequency domain-based models. Additionally, features extracted solely in either the time or frequency domain are not comprehensive enough. To solve these problems, we propose a fully complex network (FCNet) in this work, where all network layers are adapted to handle complex-valued computations to simultaneously learn the information in the real and imaginary parts. Firstly, we utilize time-frequency conversion to obtain time-frequency domain signals. And then we design the time-frequency filter-enhanced block to effectively capture global features from time-frequency signals. Finally, we design the complex-valued time-frequency Transformers Block, which separately extracts information from the time and frequency domains. Experimental evaluations on eight datasets from five benchmark domains demonstrate that our model significantly outperforms state-of-the-art methods in time series forecasting. Code is available at https://github.com/ZHU-0108/FCNet-main.

## Contact
If you have any questions or concerns, please contact us: 1187997542@qq.com or submit an issue

## Citation
If you find this repo useful in your research, please consider citing our paper as follows:
```bibtex
 @article{zhu2024fcnet,
  title={FCNet: Fully Complex Network for Time Series Forecasting},
  author={Zhu, Xuanbing and Shen, Dunbin and Wang, Hongyu and Hao, Yingguang},
  journal={IEEE Internet of Things Journal},
  year={2024},
  publisher={IEEE}
}
```
