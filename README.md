## <center>Welcome to the CCDeep</center>



### 一、快速开始

1.   克隆此仓库到您喜欢的任意位置，并<a href="https://github.com/frozenleaves/CCDeep-release1.2/releases/download/v1.2/models.rar"><font color=red>下载</font></a>预训练好的模型。
2.   使用conda创建一个合适的python虚拟环境`CCDeep`，如果您不介意污染系统环境，也可以省略这一步，直接使用系统python环境。如何创建虚拟环境请参见下面的详细安装教程。
3.   激活刚才创建的虚拟环境，并安装一些依赖的包：`pip install requirements.txt`，其中有一些依赖包可能无法通过此命令顺利安装，下文有详细解决方案。如果此命令执行失败， 请按照顺序逐个执行`pip install package` 或者`conda install packages`。
4.   进入到您下载的源码中的`CCDeep`目录, 里面有一个`main.py`文件，您可以不提供任何参数来运行这个文件，这会告诉您如何使用这个package。您也可以下载一些示例图片来真正的执行它，下载示例图片请点击<a href="https://github.com/frozenleaves/CCDeep/tree/master/examples/images/"><font color=red>这里</font></a>。将示例图片下载到合适的地方后， 执行 `python main.py -bf [your example_of_dic.tif path] -pcna [your example_of_mcy.tif path] -o [your output file savepath]`，或者在`main.py`所在目录下直接运行`python .\main.py -bf ..\examples\images\example_of_dic.tif -pcna ..\examples\images\example_of_mcy.tif`,不出意外您稍后就可以得到最终的单帧预测输出结果，您可以选择将其加载到VIA中可视化查看，也可以稍后<a href=""><font color=blue>转化为zip文件导入到ImageJ</font></a>中查看。
5.   更多内容请查看<a href="">**使用示例**</a>



### 二、安装

##### 一、从仓库克隆源代码：`https://github.com/frozenleaves/CCDeep.git`到合适的位置

##### 二、安装Anaconda3并配置虚拟环境

1.   如果您的计算机没有安装anaconda3， 请从<a href="https://www.anaconda.com/products/distribution">**这里**</a>下载并安装，视您的计算机系统不同，选择合适的版本安装即可。

2.   安装完anaconda3后，如果是Windows系统，打开其中的`Anaconda Powershell Prompt`（从开始菜单栏可以找到），如果是Linux系统或者是mac系统，直接打开终端即可。
3.   使用`conda create -n CCDeep python=3.7`新建一个conda虚拟环境，并使用`conda activate CCDeep`来激活并使用这个环境。
4.   如果您的电脑具有NVIDIA的GPU，并且您想要通过GPU来加速运算，请按照<a href="">步骤三</a>来安装`tensorflow-GPU`，以及相应的`cudatoolkiit`和`cudnn`，如果您不需要使用GPU或者电脑上没有NVIDIA的GPU，只需要安装`tensorflow`CPU版本的即可。
5.   请按照下面的命令顺序依次安装依赖的包：
     1.   `pip install tensorflow==2.4.0` (如果安装GPU版本，请先按照步骤三安装相应的package，再安装下面的其他package)
     2.   `pip install stardist==0.8.3` 
     3.   `pip install opencv-python`
     4.   `pip install scikit-image`
     5.   `pip install matplotlib`
     6.   `pip install tifffile`
     7.   `pip install pylibtiff` (如果安装报错，或者安装后运行报错，请下载wheel文件离线安装，wheel文件请从<a href="https://www.lfd.uci.edu/~gohlke/pythonlibs/#pylibtiff">**这里**</a>选择合适的版本下载。)
     8.   `pip install bitarray`



##### 三、安装TensorFlow-GPU版本



##### 四、安装中可能刚出现的问题以及解决方案

	1. `OSError: Failed to open file b'C:\\Users\\\xe6\x96\x87...\\ AppData\\Local\\Temp\\scipy-xxxxx`，此问题原因为环境变量`TEMP`和`TMP`的路径存在中文，改为纯英文路径即可。
	1. 如果遇到其他包依赖问题，根据错误提示进行相应的安装即可。
	1. 为了兼容性起见，请首先安装`cudatoolkit`和`cudnn`，然后安装`tensorflow`，再安装其他package。
	1. 有其他任何问题请上传issue。
