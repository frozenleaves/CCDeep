## <center>Welcome to the CCDeep</center>



### 一、快速开始

1.   克隆此仓库到您任意位置，并<a href="https://github.com/frozenleaves/CCDeep-release1.2/releases/download/v1.2/models.rar"><font color=red>下载</font></a>预训练好的模型。
2.   使用conda创建一个合适的python虚拟环境`CCDeep`，如果您不介意污染系统环境，也可以省略这一步，直接使用系统python环境。如何创建虚拟环境请参见下面的详细安装教程。
3.   激活刚才创建的虚拟环境，并安装一些依赖的包：`pip install requirements.txt`，其中有一些依赖包可能无法通过此命令顺利安装，下文有详细解决方案。如果此命令执行失败， 请按照顺序逐个执行`pip install package` 或者`conda install packages`。
4.   进入到您下载的源码中的`CCDeep`目录, 里面有一个`main.py`文件，您可以不提供任何参数来运行这个文件，这会告诉您如何使用这个package。您也可以下载一些示例图片来真正的执行它，下载示例图片请点击<a href="https://github.com/frozenleaves/CCDeep/tree/master/examples/"><font color=red>这里</font></a>。将示例图片下载到合适的地方后， 执行 `python main.py -bf [your example_of_dic.tif path] -pcna [your example_of_mcy.tif path] -o [your output file savepath]`,不出意外您稍后就可以得到最终的单帧预测输出结果，您可以选择将其加载到VIA中可视化查看，也可以稍后<a href=""><font color=blue>转化为zip文件导入到ImageJ</font></a>中查看。
5.   更多内容请查看<a href="">**使用示例**</a>



### 二、安装

