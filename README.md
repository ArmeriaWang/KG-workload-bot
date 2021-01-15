## Knowledge Graph Workload Bot

### 环境配置

Conda环境配置单在项目根目录下的`requirements.txt`中。

用Anaconda或Miniconda新建虚拟环境并激活，然后运行以下命令即可。

```shell
$ conda install --yes --file requirements.txt
```

注意，我使用的PyTorch为CUDA 10.2版本，如有冲突请另行安装。

此外，实验中使用的MySQL版本为8.0.22，用户名为`root`，口令为`12345678`，数据库名为`main`。

### 数据导入

推荐直接用MySQL导入`data`目录下的`main.sql`（须先解压`main.zip`）。如失败，可以运行`create_db.py`进行导入（需要花费数小时的时间）。