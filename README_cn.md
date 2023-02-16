## <center>欢迎使用CCDeep！</center>

------

[English](./README.md) | 简体中文



### 一、快速开始

1.   克隆此仓库到您喜欢的任意位置，并<a href="https://github.com/frozenleaves/CCDeep-release1.2/releases/download/v1.2/models.rar"><font color=red>下载</font></a>预训练好的模型。
2.   使用conda创建一个合适的python虚拟环境`CCDeep`，如果您不介意污染系统环境，也可以省略这一步，直接使用系统python环境。如何创建虚拟环境请参见下面的详细安装教程。
3.   激活刚才创建的虚拟环境，并安装一些依赖的包：`pip install requirements.txt`，其中有一些依赖包可能无法通过此命令顺利安装，下文有详细解决方案。如果此命令执行失败， 请按照顺序逐个执行`pip install package` 或者`conda install package`。
4.   进入到您下载的源码中的`CCDeep`目录, 里面有一个`main.py`文件，您可以不提供任何参数来运行这个文件，这会告诉您如何使用这个package。您也可以下载一些示例图片来真正的执行它，下载示例图片请点击<a href="https://github.com/frozenleaves/CCDeep/releases/tag/v1.2.1"><font color=red>这里</font></a>。将示例图片下载到合适的地方后， 执行 `python main.py -bf [your example_of_dic.tif path] -pcna [your example_of_mcy.tif path] -o [your output file savepath]`，或者在`main.py`所在目录下直接运行`python .\main.py -bf ..\examples\images\example_of_dic.tif -pcna ..\examples\images\example_of_mcy.tif`,不出意外您稍后就可以得到最终的单帧预测输出结果，您可以选择将其加载到VIA中可视化查看，也可以稍后<a href=""><font color=blue>转化为zip文件导入到ImageJ</font></a>中查看。
5.   更多内容请查看<a href="#">**使用示例**</a>



### 二、安装

##### 一、从仓库克隆源代码：`https://github.com/frozenleaves/CCDeep.git`到合适的位置

##### 二、安装Anaconda3并配置虚拟环境

1.   如果您的计算机没有安装anaconda3， 请从<a href="https://www.anaconda.com/products/distribution">**这里**</a>下载并安装，根据您的计算机系统不同，选择合适的版本安装即可。

2.   安装完anaconda3后，如果是Windows系统，打开其中的`Anaconda Powershell Prompt`（从开始菜单栏可以找到），如果是Linux系统或者是mac系统，在安装完anaconda时将其加入到环境变量中，然后直接打开终端即可，如果没有加入环境变量，需要您手动添加。
3.   使用`conda create -n CCDeep python=3.7`新建一个conda虚拟环境，并使用`conda activate CCDeep`来激活并使用这个环境。
4.   如果您的电脑具有NVIDIA的GPU，并且您想要通过GPU来加速运算，请按照[步骤三](#step3)来安装`tensorflow-GPU`，以及相应的`cudatoolkiit`和`cudnn`，如果您不需要使用GPU或者电脑上没有NVIDIA的GPU，只需要安装`tensorflow`CPU版本即可。
5.   请按照下面的命令顺序依次安装依赖的包：
     1.   `pip install tensorflow==2.4.0` (如果安装GPU版本，请首先按照[步骤三](#step3)安装相应的package，再安装下面的其他package)
     2.   `pip install stardist==0.8.3` 
     3.   `pip install opencv-python`
     4.   `pip install scikit-image`
     5.   `pip install matplotlib`
     6.   `pip install tifffile`
     7.   `pip install pylibtiff` (如果安装报错，或者安装后运行报错，请下载wheel文件离线安装，wheel文件请从<a href="https://www.lfd.uci.edu/~gohlke/pythonlibs/#pylibtiff">**这里**</a>选择合适的版本下载。)
     8.   `pip install bitarray`
     9.   `pip install trackpy`
     10.   `pip install pandas`



##### <span id="step3">三、安装TensorFlow-GPU版本</span>

**如果您需要执行这一步，请确保您的虚拟环境是干净的，并且其中没有安装过numpy或者依赖numpy的package，否则可能会出现包依赖问题！
如果您的python版本>3.7, 可能会出现tensorflow没有合适的版本，这个时候也许您应该降低python版本，或者自行适配tensorflow和cuda。**

1.   安装`cudatoolkit`：`conda install cudatoolkit==11.0.221`

2.   安装`cudnn`：`conda install cudnn==8.0.5.39 -c conda-forge`

     *注：这两个包请务必使用`conda install package`命令安装，因为它们不是python package， 无法在PyPI中索引。*

3.   安装`tensorflow-gpu`：`pip install tensorflow-gpu==2.4.0`。

     *注：这一步请不要用`conda install package`命令完成，因为`conda`源并没有及时更新到合适的版本，可能会提示找不到源的问题。*

4.   测试`cuda`的可用性和兼容性：

     ```python
     >>> import tensorflow as tf
     2022-07-02 13:07:50.756143: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
     >>> tf.test.is_gpu_available()
     2022-07-02 13:08:20.525184: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcuda.so.1
     2022-07-02 13:08:21.247969: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties:
     pciBusID: 0000:18:00.0 name: NVIDIA GeForce RTX 3090 computeCapability: 8.6
     coreClock: 1.695GHz coreCount: 82 deviceMemorySize: 23.70GiB deviceMemoryBandwidth: 871.81GiB/s
     2022-07-02 13:08:21.248438: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 1 with properties:
     pciBusID: 0000:3b:00.0 name: NVIDIA GeForce RTX 3090 computeCapability: 8.6
     coreClock: 1.695GHz coreCount: 82 deviceMemorySize: 23.70GiB deviceMemoryBandwidth: 871.81GiB/s
     2022-07-02 13:08:21.248838: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 2 with properties:
     pciBusID: 0000:86:00.0 name: NVIDIA GeForce RTX 3090 computeCapability: 8.6
     coreClock: 1.695GHz coreCount: 82 deviceMemorySize: 23.70GiB deviceMemoryBandwidth: 871.81GiB/s
     2022-07-02 13:08:21.249213: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 3 with properties:
     pciBusID: 0000:af:00.0 name: NVIDIA GeForce RTX 3090 computeCapability: 8.6
     coreClock: 1.695GHz coreCount: 82 deviceMemorySize: 23.70GiB deviceMemoryBandwidth: 871.81GiB/s
     2022-07-02 13:08:21.249257: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
     2022-07-02 13:08:21.276669: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
     2022-07-02 13:08:21.276817: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
     2022-07-02 13:08:21.292073: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcufft.so.10
     2022-07-02 13:08:21.299397: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcurand.so.10
     2022-07-02 13:08:21.323907: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusolver.so.10
     2022-07-02 13:08:21.329281: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusparse.so.11
     2022-07-02 13:08:21.331335: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
     2022-07-02 13:08:21.333955: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0, 1, 2, 3
     2022-07-02 13:08:21.336478: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
     2022-07-02 13:08:25.106753: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1261] Device interconnect StreamExecutor with strength 1 edge matrix:
     2022-07-02 13:08:25.106830: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1267]      0 1 2 3
     2022-07-02 13:08:25.106849: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1280] 0:   N N N N
     2022-07-02 13:08:25.106862: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1280] 1:   N N N N
     2022-07-02 13:08:25.106875: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1280] 2:   N N N N
     2022-07-02 13:08:25.106889: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1280] 3:   N N N N
     2022-07-02 13:08:25.110906: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1406] Created TensorFlow device (/device:GPU:0 with 468 MB memory) -> physical GPU (device: 0, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:18:00.0, compute capability: 8.6)
     2022-07-02 13:08:25.114327: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1406] Created TensorFlow device (/device:GPU:1 with 22430 MB memory) -> physical GPU (device: 1, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:3b:00.0, compute capability: 8.6)
     2022-07-02 13:08:25.116215: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1406] Created TensorFlow device (/device:GPU:2 with 1250 MB memory) -> physical GPU (device: 2, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:86:00.0, compute capability: 8.6)
     2022-07-02 13:08:25.117281: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1406] Created TensorFlow device (/device:GPU:3 with 14009 MB memory) -> physical GPU (device: 3, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:af:00.0, compute capability: 8.6)
     True
     >>> 
     ```

     在终端的python交互界面运行这两行测试代码，如果有上面相同的输出，恭喜您成功安装了`cuda`并且它是可用的。如果您测试过程中Log信息长时间停留在某一行不再输出任何信息也不退出交互界面，例如交互界面停留在`Successfully opened dynamic library libcurand.so.10`，则是因为您安装的`cudatoolkit`和`cudnn`版本不兼容，需要重新安装。如果您的GPU为RTX3090或者以下，**建议您按照我们推荐的版本安装，否则请您自行适配相应的版本**。

     

##### 四、安装中可能刚出现的问题以及解决方案

​	`OSError: Failed to open file b'C:\\Users\\\xe6\x96\x87...\\ AppData\\Local\\Temp\\scipy-xxxxx`，此问题原因为环境变量`TEMP`和`TMP`的路径存在中文，改为纯英文路径即可。

​	如果遇到其他包依赖问题，根据错误提示进行相应的安装即可。

​	为了兼容性起见，请首先安装`cudatoolkit`和`cudnn`，然后安装`tensorflow`，再安装其他package。

​	如果出现其他tensorflow相关问题，试试设置系统变量：在环境变量->系统变量中添加CUDA_CACHE_MAXSIZE=4294967296

​	有其他任何问题请上传issue。



### 三、开始使用

请参考我们的<a href="./quick_start.ipynb">使用示例</a>来开始使用，更多用法请参考<a href="https://ccdeep.readthedocs.io/zh/latest/index.html">API文档</a>

