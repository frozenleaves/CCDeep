## <center>Welcome to the CCDeep！</center>

    

------

English | [简体中文](./README_cn.md)



### Quick Start

1.   Clone this repository to any location you like, and <a href="https://github.com/frozenleaves/CCDeep-release1.2/releases/download/v1.2/models.rar"><font color=red>download</font></a> the pre-trained model.
2.   Use `conda` to create a suitable Python virtual environment `ccdeep`. If you don't mind polluting the system environment, you can also omit this step and use the system Python environment directly. See the following detailed installation tutorial for how to create a virtual environment.
3.   Activate the virtual environment just created and install some dependent packages：`pip install requirements.txt`，Some of the dependent packages may not be installed successfully through this command, and detailed solutions are given below. If this command fails to execute, please execute `pip install package`or`conda install packages`one by one in order .
4.   Enter the `ccdeep` directory in the source code you just download, there have a `main.py` file, you can run this file without providing any parameters, which will tell you how to use this package. Or you can download some sample images to actually execute it, To download the smaple images, please click <a href="https://github.com/frozenleaves/CCDeep/releases/tag/v1.2.1"><font color=red>here</font></a>.After downloading the sample images to the appropriate place, you can execute `python main.py -bf [your example_of_dic.tif path] -pcna [your example_of_mcy.tif path] -o [your output file savepath]`，Or run `python ./main.py -bf ../examples/images/example_of_dic.tif -pcna ../examples/images/example_of_mcy.tif` directly in the directory where `main.py` is located. Not surprisingly, you can get the final single frame prediction output later. You can choose to load it into via for visual viewing, or later <a href=""><font color=blue>convert to zip file and import to ImageJ</font></a>for viewing.
5.   For more information, please see the <a href="#">**usage example**</a>

--------

### Installation

##### Clone source code from the repository to any you like location：`https://github.com/frozenleaves/CCDeep.git`

##### Install Anaconda3 and configure the virtual environment

1.   If your computer don't install anaconda3， please download and install from <a href="https://www.anaconda.com/products/distribution">**here**</a>, according to your computer system, choose the appropriate version to install.

2.   After installed anaconda3，if you using Windows system，please open the `Anaconda Powershell Prompt`（You can find it from the start menu bar），If your system is  Linux system or  MAC system, please add Anaconda to the environment variable when installing it, and then directly open the terminal. If you do not add the environment variable, you need to add it manually.
3.   Using `conda create -n CCDeep python=3.7` to create a new conda virtual environment, and using `conda activate CCDeep`to activate and use this environment.
4.   If your computer has NVIDIA GPU and you want to speed up the running through GPU，please follow [step3](#step3) to install `tensorflow-GPU`，and corresponding`cudatoolkiit`and `cudnn`. If you don't need to use GPU or there is no NVIDIA GPU on your computer, just install the `tensorflow` CPU version.
5.   Please install the dependent packages in the following command：
     1. `pip install tensorflow==2.4.0` (If you install the GPU version，please following the [step3](#step3)step3 to install the corresponding packages，and then install other packages below.)
     2. `pip install stardist==0.8.3` 
     3. `pip install opencv-python`
     4. `pip install scikit-learn`
     5. `pip install scikit-image`
     6. `pip install matplotlib`
     7. `pip install tifffile`
     8. `pip install pylibtiff` (If an error occur during installation or after installation, please download the wheel file for offline installation，you can download the wheel file from <a href="https://www.lfd.uci.edu/~gohlke/pythonlibs/#pylibtiff">**here**</a>, just select the appropriate version to download.)
     9. `pip install bitarray`
     10. `pip install trackpy`
     11. `pip install pandas`
     12. `pip install treelib  `
     13. `pip install imagecodecs`
     14. `pip install imutils`

-------

##### <span id="step3">Install TensorFlow-GPU Version</span>

**If you need to perform this step, please ensure that your virtual environment is clean, and there is no numpy or numpy dependent package installed, otherwise package dependency may occur! 
If your python version > 3.7, the tensorflow may not complete the high python version, you need to lower your python,or adapt the tensorflow and cuda by yourself.**

1.   Install `cudatoolkit`：`conda install cudatoolkit==11.0.221`

2.   Install `cudnn`：`conda install cudnn==8.0.5.39 -c conda-forge`

     *Note: these two packages must be installed with`conda install package`command， because this are not `Python packages` and cannot be indexed in PyPi.*

3.   Install `tensorflow-gpu`：`pip install tensorflow-gpu==2.4.0`。

     *Note：Please do not use`conda install package`command to install this package，because the `conda` source is not updated to the appropriate version in time, which may prompt the problem that the source cannot be found.*

4.   Test the availability and compatibility of `CUDA`：

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

     Run these two lines of test code in the python interactive interface. If there is the same output above, congratulations!  You are successfully installing `CUDA` and its availability. If the Log information stays in a certain line for a long time during the test, and your terminal  do not output any information and will not exit the interactive interface, for example, the interactive interface stays at`Successfully opened dynamic library libcurand.so.10`, this is the reason that your`cudatoolkit`and `cudnn`have an incompatible version，you may need to reinstall this two packages. If your GPU is RTX3090 or below，**We recommend that you install according to the recommended version, otherwise, please adapt the corresponding version by yourself**.

     

##### Possible Rroblems and Solutions During Installation

​	`OSError: Failed to open file b'C:\\Users\\\xe6\x96\x87...\\ AppData\\Local\\Temp\\scipy-xxxxx`, The reason for this problem is that the path of the environment variables `TEMP` and `TMP` exist in Chinese, so you can change them to pure English path.

​	If you encounter other package dependency problems, just install them according to the error prompt.

​	For compatibility, please first install `cudatoolkit`and `cudnn`，and then install `tensorflow`，after this, Install other packages.

​	If something errors about the tensorflow, you can try add an PATH Variable **CUDA_CACHE_MAXSIZE=4294967296**, maybe it can resolve the problems.

​	Please upload issue if you have any other questions.




### Start Using

Please refer to our <a href= "./quick_start.ipynb" > usage examples </a> to start using. For more usage, please refer to the <a href= "https://ccdeep.readthedocs.io/zh/latest/index.html" >API documentation</a>.

