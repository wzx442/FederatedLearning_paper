# FedEraser: Enabling efficient client-level data removal from federated learning models -- 实现联邦学习模型中高效的客户端级数据删除

## 论文来源
|名称 |	[FedEraser: Enabling efficient client-level data removal from federated learning models](https://ieeexplore.ieee.org/document/9521274)        |
|-|-|
| 期刊| [IWQOS 2021](https://ieeexplore.ieee.org/document/9521274)         |
| 作者 |	Gaoyang Liu, Xiaoqiang Ma, Yang Yang, Chen Wang, Jiangchuan Liu      |
|DOI |	DOI: 10.1109/IWQOS52092.2021.9521274    |


--------------------------
    这是第一篇联邦遗忘论文，但是完成度还是不错的。
----------------------------

## 背景
- 现有机器学习领域中的遗忘学习技术由于联邦学习与机器学习在数据学习方式上的固有差异，无法直接用于联邦学习。
- 一种朴素的满足请求删除的方法是，在移除目标数据后，仅使用剩余数据从头重新训练模型。然而，对于许多应用来说，成本（在时间、计算、能源等方面）可能高得难以承受，尤其是在联邦学习（FL）环境中涉及多方参与者之间多轮训练与聚合交替的情况下。

## TDLR
本文中提出了FedEraser，一种高效的联邦遗忘方法，能够消除联邦客户数据对全局模型的影响，同时显著减少遗忘时间。
FedEraser的基本思想是以中央服务器的存储换取未遗忘模型的构建时间，其中FedEraser通过利用在联邦学习训练过程中中央服务器保留的客户历史参数更新来重构未遗忘模型。由于保留的更新源自包含目标客户数据影响的全局模型，因此在使用这些更新进行遗忘之前，必须对其进行信息解耦的校准。基于客户更新指示全局模型参数需要朝哪个方向变化以使模型适应训练数据的事实，我们进一步通过仅进行少量轮次的校准训练来校准保留的客户更新，以近似无目标客户情况下的更新方向，并可利用校准后的更新迅速构建未遗忘模型。
