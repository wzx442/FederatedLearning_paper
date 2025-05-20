# Efficient and Privacy-Enhanced Federated Learning Based on Parameter Degradation--基于参数退化的高效隐私增强联邦学习
## 来源
TSC	2024 
链接: [https://ieeexplore.ieee.org/abstract/document/10528912](https://ieeexplore.ieee.org/abstract/document/10528912)
## Abstract
> Federated Learning ensures that clients can collaboratively train a global model by uploading local gradients,keeping data locally, and preserving the security of sensitive data.
> However, studies have shown that attackers can infer local data from gradients, raising the urgent need for gradient protection.
> The differential privacy technique protects local gradients by adding noise. This paper proposes a federated privacy-enhancing algorithm that combines local differential privacy, parameter sparsification, and weighted aggregation for cross-silo setting.
- 联邦学习通过上传本地梯度，保持数据在本地，保护了敏感数据的安全，确保客户可以协同训练全局模型。
- 然而，研究表明，***攻击者可以从梯度中推断出本地数据***，因此迫切需要进行梯度保护。
- 差分隐私技术通过添加噪声来保护局部梯度。本文提出了一种联合隐私增强算法，该算法结合了局部差分隐私、参数稀疏化和孤井互连环境下的加权聚合。
```python
cross-cilo：
silo 的本意是地窖、竖井。cross-silo 里的silo显然不是这个意思,它是指“信息孤井”,意思相同的词是 island,信息孤岛。
所谓信息孤井就是指: 一个企业中各个部门的信息系统相互独立，各自为战，没有联系，就像一口口孤井。
这样的结果显然是效率降低，信息重复度高，易冲突出错。
因此出现 cross-silo 的概念，就是打通这些孤井，让它们连成一体，这样将提高信息一致性、提高效率，而且有利于准确分析。
所以 cross-silo 可以翻成 孤井互连，或者孤井互通等等。
在Introduction有详细对比
```
> Firstly, our method introduces $R\acute{e} nyi$ differential privacy by adding noise before uploading local parameters, achieving local differential privacy. Moreover, we dynamically adjust the privacy budget to control the amount of noise added, balancing privacy and accuracy. 
> Secondly, considering the diversity of clients’ communication abilities, we propose a novel Top-K method with dynamically adjusted parameter upload rates to effectively reduce and properly allocate communication costs.
> Finally, based on the data volume, trustworthiness, and upload rates of participants, we employ a weighted aggregation method, which enhance the robustness of the privacy framework.
- 首先，本文引入了  **"$R\acute{e} nyi$ 差分隐私"** ，即在上传本地参数前添加噪声，从而实现本地差分隐私。此外，我们还通过动态调整隐私预算来控制噪声的添加量，从而在隐私和准确性之间取得平衡。
- 其次，考虑到客户端通信能力的多样性，本文提出了一种动态调整参数上传率的新型 Top-K 方法，以有效降低并合理分配通信成本。
- 最后，根据客户端的**数据量**、**可信度**和**上传率**，本文采用了**加权聚合法**，从而增强了隐私框架的稳健性。
> Index Terms—Federated learning, differential privacy, communication costs, credibility, aggregation.
- 关键词：联合学习、差分隐私、通信成本、可信度、聚合。
## INTRODUCTION
### A. Background
> Different organizations are hesitant to contribute their own data due to privacy concerns. This has resulted in data silos, which hinder effective data integration.
- 随着大数据的发展，人工智能达到了新的高度。然而，实现高精度学习模型需要大规模和高质量的数据支持。遗憾的是，不同的组织出于对隐私的考虑，不愿贡献自己的数据。这就造成了数据孤岛，阻碍了有效的数据整合。
- 联邦学习是一种新兴的**分布式机器学习方法**，通过保护数据隐私和所有权来解决这一问题。在联合学习中，客户端在本地训练自己的模型，并将模型参数上传到中央服务器，中央服务器汇总这些参数，更新全局模型，并将更新后的模型发回给客户端。这一过程反复进行，直到全局模型达到预定的精度或收敛条件。
> In the realm of FL, there exist two predominant configurations: cross-device and cross-silo. In cross-device FL,participants typically comprise edge devices such as smart gadgets and laptops, which may number in the thousands or even millions. These participants are generally considered unreliable and possess limited computational capabilities. In contrast, in the cross-silo FL paradigm, the stakeholders are organizations; the number of participants is relatively limited,usually ranging between 2 and 100. Given the nature of the participants, the process is generally deemed reliable,and each entity possesses significant computational resources.Cross-silo scenarios are exceedingly common in real-world applications, such as credit card fraud detection, clinical disease prediction, 6G network and so on.

- 在联邦学习领域，存在两种主要配置：跨设备和跨孤岛。在跨设备联邦学习中，客户端通常包括智能工具和笔记本电脑等边缘设备，其数量可能达到数千甚至数百万。这些客户端通常被认为是不可靠的，而且计算能力有限。
- 相比之下，在跨孤岛联邦学习中，利益相关者是组织；客户端的数量相对有限，通常在 2 到 100 之间。鉴于客户端的性质，该过程通常被认为是可靠的，而且**每个实体都拥有大量的计算资源**。在现实世界的应用中，如信用卡欺诈检测、临床疾病预测、6G 网络等，跨孤岛场景极为常见。
```c
Cross-device和Cross-Silo的联邦学习的区别:
1、模式不同
Cross-device联邦学习：多设备联邦的模式。

Cross-Silo联邦学习：与跨设备联合学习的特征相反，Cross-Silo 联邦学习在总体设计的某些方面非常灵活。
许多组织如果只是想共享训练模型，而不想分享数据时，cross-silo设置是非常好的选择。
Cross-Silo 联邦学习的设置主要有以下几个要点：数据分割、激励机制、差异隐私、张量因子分解。

2、面对的客户端不同
Cross-device联邦学习：Cross-device FL针对的则是便携式电子设备、穿戴式电子设备等，统称为物联设备（Internet of Things, IoT devices）。

Cross-Silo联邦学习：Cross-silo FL面对的客户端是企业级别、机构单位级别的。

3、客户端状态不同 
Cross-device联邦学习：无状态，每个客户可以仅会参与一次任务，因此通常假定在每轮计算中都有一个从未见过的客户的新样本。

#Cross-Silo联邦学习：有状态，每个客户端都可以参与计算的每一轮，并不断携带状态。

4、可定位性不同
Cross-device联邦学习：没有独立编号，无法直接为客户建立索引。

#Cross-Silo联邦学习：有独立编号，每个客户端都有一个标识或名称，该标识或名称允许系统专门访问。

5、发展瓶颈不同
Cross-device联邦学习：计算传输开销、通信不稳定。

Cross-Silo联邦学习：数据异构。

6、通信情况不同
Cross-device联邦学习：不稳定、不可靠。

Cross-Silo联邦学习：稳定且可靠。

7、数据划分依据不同
Cross-device联邦学习：横向划分。

Cross-Silo联邦学习：可横向纵向划分。
```

- 在许多跨孤岛场景中，联邦学习架构被用作隐私保护方案，即服务器与客户端之间的交互仍然是原始模型参数。虽然联邦学习不需要共享本地数据，但仅从参数更新就能推断出本地节点的相关隐私信息。相关工作表明，梯度的暴露可以揭示其他诚实节点的类表示、与主要任务无关的样本中的敏感属性，甚至本地训练样本等信息。因此，有必要加强联合学习模型的隐私保护。这也是本文致力于研究的问题--跨孤岛联邦学习下的隐私增强方法。

> Differential privacy (DP) achieves the privacy goal through data perturbation and introduces minimal additional computational burden, making it widely applicable in various scenarios. This paper also embraces differential privacy as a means to enhance privacy in federated learning. Differential privacy achieves privacy preservation by injecting noise into the parameters. The greater the amount of noise, the stronger the privacy, albeit at the expense of accuracy. Privacy refers to the extent to which details of the client’s local data are protected from disclosure. Accuracy refers to the degree to which the trained final model predicts accurately in the face of new samples. Consequently, balancing privacy with accuracy is a focal point in the design of differential privacy approaches.
- 差分隐私（DP）通过数据扰动实现隐私目标，并将额外的计算负担降至最低，因此广泛适用于各种场景。本文还将差分隐私作为增强联邦学习中隐私保护的一种手段。差分隐私通过向参数中注入噪声来实现隐私保护。噪声越大，隐私性就越强，尽管会牺牲准确性。隐私是指客户本地数据的细节在多大程度上受到保护而不被泄露。准确性是指训练有素的最终模型在面对新样本时预测准确的程度。因此，平衡隐私与准确性是设计差异隐私方法的重点。

> Irrespective of the privacy protection method employed, communication cost remains a pivotal challenge to address.
- 无论采用哪种隐私保护方法，**通信开销仍然是需要解决的关键难题**。通信是指在整个训练过程中，服务器与客户端之间交互的参数数量。在每个周期中，客户端需要将其本地模型参数传送给中央服务器或其他客户端。
- 然而，模型参数通常可以在有限的网络带宽上传输如此大量的数据可能会导致通信延迟或故障，最终降低联合学习的整体效率。
- 此外，对于差分隐私，添加的噪声量与参数数量成正比；噪声越多，对模型准确性的影响越大。
- 此外，每个客户端的设备、计算能力和通信能力都存在异质性，应根据实际情况进行合理的自适应调整。
- 因此，如何自适应地实现模型精度、隐私和通信成本之间的平衡仍是一个重要课题。

> Finally, the robustness of global model aggregation are key issues in privacy-enhanced federated learning.
>  In this paper, the predictable disturbance is a carefully designed noise added to achieve differential privacy.Unforeseeable disturbances are mainly attacks, such as model poisoning attacks launched by malicious attackers.
>  Against the backdrop of data distribution heterogeneity, there are variations in the quality and quantity of local data among participants.The process of aggregating noised parameters from multiple local nodes to obtain a comprehensive model representationis critical to the success of this approach. 
- 最后，全局模型聚合的稳健性是隐私增强联合学习的关键问题。鲁棒性是指面对可预见和不可预见的干扰时，模型训练过程正常进行的能力。
- 在**本文中，可预见的干扰是为实现差分隐私而精心设计添加的噪音。不可预见的干扰主要是攻击**，如恶意攻击者发起的模型中毒攻击。在数据分布异构的背景下，客户端之间本地数据的质量和数量也存在差异。从多个局部节点汇总噪声参数以获得综合模型表示的过程，对这种方法的成功至关重要。局部节点训练过程的有效性会极大地影响全局模型聚合的质量。有几个因素会影响本地节点训练的效果，如训练数据的质量、噪声量、上传参数的数量以及恶意攻击者等。因此，**必须全面评估局部参数的可信度，以确保聚合质量**。

> The heterogeneity of client data distribution makes the local data quality of clients uneven, which will directly affect the accuracy and robustness of aggregation.
- 客户端数据分布的异质性使得客户端的本地数据质量参差不齐，这将直接影响聚合的准确性和鲁棒性。
- 隐私是通过添加噪声来实现的，噪声量也会影响准确性和鲁棒性。噪声的大小可以通过交互参数的数量来控制，但如果通信参数的数量太少，则会对模型的准确性和稳健性产生不利影响。总之，建立一个高效、稳健的联合隐私增强架构必须考虑隐私、准确性、通信和稳健性。

> The relationship between these properties is shown in Fig. 1, where the ‘+’ indicates positive correlation and the ‘?’ indicates negative correlation.
- 这些特性之间的关系如图 1 所示，其中 "+"表示正相关，"-"表示负相关。
![fig1](image\fig1.png)
### B. Contributions
> Our algorithm adaptively adjusts the privacy budget and parameter upload rate and employs importance-weighted aggregation to achieve robust learning in scenarios involving malicious
participants.
- 为了应对这些挑战，即如何平衡隐私、准确性、通信成本和聚合鲁棒性，本文提出了一种联合隐私增强架构。本文的算法可以自适应地调整隐私预算和参数上传率，并采用重要性加权聚合，从而在涉及恶意客户端的情况下实现稳健学习。

> We summarize the main contributions of this paper as follows:
> 	- We introduce a simple yet effective dynamic privacy budget adjustment mechanism for $R\acute{e} nyi$ differential privacy. This adjustment, based on changes in global model accuracy within a given time window, directly mitigates the accuracy decline caused by added noise.
> - Addressing the issue of communication cost, we propose an adaptive parameter upload rate adjustment method based on communication latency. This method first assesses the capabilities of participating nodes and then dynamically adjusts the parameter upload rate based on the heterogeneity of node capabilities.
> - We propose an importance-weighted aggregation method.By evaluating the contribution of local node parameters to the global model through multiple factors and considering the credibility of parameters by integrating both localglobal and intra-local node relationships, we effectively enhance the robustness and efficiency of global model aggregation.
- 本文引入了一种简单而有效的**动态隐私预算调整机制**，适用于 $R\acute{e} nyi$差分隐私。这种调整机制基于给定时间窗口内**全局模型精度**的变化，可直接缓解因噪声增加而导致的精度下降。
- 针对通信成本问题，本文提出了一种**基于通信延迟**的**自适应参数上传率调整方法**。这种方法首先评估参与节点的能力，然后根据节点能力的异质性动态调整参数上传率。
- 本文提出了一种重要性加权汇总法。通过多因素评估本地节点参数对全局模型的贡献，并综合 **本地-全局**和**本地内部节点**关系考虑参数的可信度，我们有效地提高了全局模型聚合的鲁棒性和效率。

```c
1. Privacy:
在本文的研究中，为 "Renyi差分隐私"（RDP）引入了一种简单而有效的动态隐私预算调整机制。
根据给定历史时间窗口内全局模型准确度的变化，动态调整下一轮的隐私预算，直接缓解因噪声增加而导致的准确度下降。   
2. Communication:
除了隐私和准确性，通信量也是影响联邦学习的直接因素。
在联邦学习中，通信开销指的是交互中的参数数量。通信量越大，需要交互的参数数量就越多。
客户端层面的差分隐私是在交互前为每个参数添加噪声，参数数量越多，要达到相同程度的隐私就需要更多的噪声。
但与此同时，通信量较少，即上传的参数较少，而模型参数的不完整势必会带来模型的不准确性。
因此，需要同时考虑隐私、准确性和通信成本之间的权衡。      
参数稀疏化是联合学习中广泛使用的一种降低通信成本的技术。
文献提出，在训练过程中，客户端可以上传绝对值大于某个阈值的梯度。然而，具体工作中对参数选择率的确定还没有深入研究。
通常，参数选择率被视为一个超参数，以证明模型的稳定性。        
在本文的研究中，提出了一种基于通信延迟的自适应参数上传速率调整方法。
该方法首先评估参与节点的能力，然后根据节点能力的异质性动态调整参数上传速率，
从而使本地信息压缩速率的下限更加紧凑，有效降低通信成本。      
3. Global Model Aggregation
就全局模型聚合而言，在异构数据分布的背景下，客户端之间的本地数据质量和数量存在差异，因此这一过程至关重要。
聚合来自多个本地节点的噪声参数以形成一个综合模型，对于差分隐私方法的成功至关重要。
噪音的存在，再加上潜在的恶意外部攻击者，会阻碍聚合算法的正确收敛。       
在本研究中，提出了一种重要性加权聚合方法。
该方法利用多种因素评估本地节点参数对全局模型的贡献，并通过整合本地-全局和本地节点内部的关系来考虑参数的可信度。
通过对聚合的噪声参数进行效用加权，有效地提高了全局模型聚合的鲁棒性和效率。       

```

## PRELIMINARY
> In a federated learning system that leverages local differential privacy, users upload perturbed parameter values instead of the original ones, ensuring that the perturbed parameters are safeguarded against privacy inference attacks. The privacy level of differential privacy is determined by the privacy budget $\epsilon$.
> A lower privacy budget provides a higher degree of privacy protection,but it can also lead to lower model accuracy.
- 在数据集中存储在单一机构进行数据处理（如数据发布）的情况下，集中式差分隐私可以有效保护整个数据集的隐私。然而，在联邦学习中，数据是以分散的方式存储在多个客户端中的，通常会利用本地差分隐私来抵御对共享参数的推理攻击。谷歌、苹果和微软等知名公司已将本地差分隐私集成到其产品中。
- 在利用局部差分隐私的联合学习系统中，用户上传扰动参数值而不是原始参数值，从而确保扰动参数免受隐私推断攻击。差分隐私的隐私级别由隐私预算$\epsilon$ 决定。
- 隐私预算越低，隐私保护程度越高、但也可能导致模型精度降低。

```c

差分隐私主要通过对数据添加噪声来实现隐私保护。
隐私预算（ε）控制了添加噪声的量，从而平衡了隐私和数据的有用性。
隐私预算，简单地说，就是添加噪声后得到的模型精度，
隐私预算越低，最后得到的模型精度就越低，也就是添加了相对较大的噪声。    

```

> **Definition 1.** (($\epsilon, \eth$)-differential privacy, ($\epsilon, \eth$)-DP): A randomization mechanism $\mathrm{M}$ satisfies ($\epsilon, \eth$)-differential privacy ( $\epsilon > 0, \eth > 0$) when and only when for any adjacent input datasets *D* and *D′* and any possible set of output values $R_M$, there is:
![eq1](image\eq1.png)

- 定义 1. (($\epsilon, \eth$)-差分隐私，($\epsilon, \eth$)-DP)：随机化机制 $\mathrm{M}$ 满足($\epsilon, \eth$)-差分隐私 ( $\epsilon > 0, \eth > 0$)，当且仅当对于任意相邻的输入数据集 *D* 和 *D′*，以及任意可能的输出值集 $R_M$，有Eq(1)。
- 放宽的差分隐私定义 ($\epsilon, \eth$)-DP 可以理解为，该机制以最小的  1 - $\eth$  概率满足 $\epsilon$-DP 。

- [ ] >>>**什么是($\epsilon, \eth$)-DP？**<<<
- [X] >>参考 [知乎](https://zhuanlan.zhihu.com/p/264779199)<< 
- [X] 参考了一部分 [csdn](https://blog.csdn.net/m0_43424329/article/details/121650574?ops_request_misc=&request_id=&biz_id=102&utm_term=(%CF%B5,%20%CE%B4)-DP&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-121650574.nonecase&spm=1018.2226.3001.4187)
- [X] M是一个随机函数，$R_M$是一个输出值集合，$P_r$是一个概率函数。
- [X] 在差分隐私（Differential Privacy）领域中，“Pr”通常指的是概率（Probability）函数，用来表示某个事件发生的概率。在差分隐私的定义中，Pr 函数用于描述不同数据集上执行查询或算法输出的概率。差分隐私的基本思想是通过向查询结果添加随机噪声来保护个体的数据隐私，使得任何观察者都无法从查询结果中推断出单个个体的具体信息。差分隐私的形式化定义如下：
	- 给定两个相邻的数据集 D 和 D′ ，它们只相差一条记录，一个随机化机制 M 满足**?-DP**，如果对所有可能的输出 S 有：
$P_r[M(D)\in S]\le e^{\epsilon}\cdot P_r[M(D')\in S]$
这里的 Pr[M(D)∈S] 表示机制 M 在数据集 D 上运行时，输出落在集合 S 内的概率。同样地，Pr?[M(D′)∈S] 表示机制 M 在数据集 D′ 上运行时，输出落在相同集合 S 内的概率。
? 是一个正实数，它衡量了隐私保护的程度。当 ? 趋近于 0 时，意味着差分隐私的保护程度更高，即输出的概率几乎完全独立于数据集中的个体信息；而当 ? 较大时，意味着保护程度较低，输出可能更容易受到数据集中个体信息的影响。
	- **对于 ($\epsilon, \eth$)-DP** ，定义变为**Eq(1)**，即：
$P_r[M(D)\in S]\le e^{\epsilon}\cdot P_r[M(D')\in S]+\eth$，
称该随机函数 $\mathrm{M}$ 满足($\epsilon, \eth$)-differential privacy（差分隐私），简写为 **($\epsilon, \eth$)-DP**

> **Sequential Composition:** If $F_1(x)$ satisfies ($\epsilon_1, \eth_1$)-DP, while $F_2(x)$ satisfies ($\epsilon_2, \eth_2$)-DP, then the mechanism $G(x)=(F_1(x),F_2(x))$ satisfies ($\epsilon_1+\epsilon_2, \eth_1+\eth_2$)-DP.
- 顺序组合：如果$F_1(x)$ 满足 ($\epsilon_1, \eth_1$)-DP 条件，而 $F_2(x)$ 满足 ($\epsilon_2, \eth_2$)-DP 条件，那么机制 $G(x)=(F_1(x),F_2(x))$ 满足 ($\epsilon_1+\epsilon_2, \eth_1+\eth_2$)-DP条件。

```c

#已知一个随机函数的隐私损失，那么 K 个随机函数总的隐私损失是多少呢？
#很惊讶地发现，在每一个随机函数所加的噪声（noise）相互独立时，隐私损失是可以累加的！（（（这里的隐私损失是指Cm，在知乎文章中可见）））

```

> **Parallel Composition:** If the dataset $D$ is divided into $k$ disjointed subdata blocks $x_1\cup x_2 \cup ... \cup x_k=D,F(x_1),...,F(x_k)$  satisfy ($\epsilon_1,\eth$)-DP$,...,$($\epsilon_k,\eth$)-DP respectively, then the mechanism for publishing all results $F(x_1),...,F(x_k)$ is satisfied ($max(\epsilon_1,...,\epsilon_k),\eth$)-DP.
- 并行组合：如果数据集 D 被划分为 k 个不相连的子数据块 $x_1\cup x_2 \cup ... \cup x_k=D,F(x_1),...,F(x_k)$分别满足   ($\epsilon_1,\eth$)-DP$,...,$($\epsilon_k,\eth$)-DP，那么所有隐私损失 $F(x_1),...,F(x_k)$对比满足($max(\epsilon_1,...,\epsilon_k),\eth$)-DP。
```c

将数据集划分为K份，每一份满足各自的随机函数，那么所有的随机函数相对比，满足max       

```
> **Definition 2.** ($\ell_2$-Sensitivity): For the real-valued function $f$ acting on the dataset *D* and *D′* , the$\ell_2$ sensitivity of $s$ is expressed as
![eq2](image\eq2.png)
- **定义 2**（$\ell_2$ 灵敏度）：对于作用于数据集 *D* 和 *D′* 的实值函数 $f$，$s$ 的 $\ell_2$ 灵敏度表示为Eq(2)。
- 灵敏度是指单个数据的变化对整个数据库查询结果影响最大的程度。
- 其中$f$就是一个简单的查询函数。

> **Lemma 1.** DP for Gaussian mechanism: One way to make the mechanism satisfy differential privacy is to add noise to the results. Gaussian mechanisms help mechanisms achieve differential privacy by adding noise that satisfies a Gaussian distribution. But the Gaussian mechanism cannot satisfy ?differential privacy, it can satisfy ($\epsilon,\eth$)-differential privacy.For a random function $F(x)$, the Gaussian mechanism can be used to obtain a random function satisfying (($\epsilon,\eth$)-differential privacy $F^′ (x)$:
![eq3](image\eq3.png)
- **定理 1.** 高斯机制的 DP：使机制满足差分隐私的一种方法是在结果中添加噪声。高斯机制通过添加满足高斯分布的噪声来帮助机制实现差分隐私。但高斯机制不能满足$\epsilon$-DP，它只能满足 ($\epsilon,\eth$)-DP。对于随机函数$F(x)$，可利用高斯机制获得满足 ($\epsilon,\eth$)-DP 的随机函数 $F^ ′ (x)$，如Eq(3)。
- 其中，$\sigma^2=\frac{2s^2ln(1.25/\eth)}{\epsilon^2}$，$s$ 是 $F$ 量化的数据隐私暴露程度的灵敏度；$\mathcal{N}(\sigma^2)$ 表示高斯（正态）分布的抽样结果，其均值为 0，方差为 $\epsilon^2$。
- 高斯机制的优点之一是，为实现隐私保护而添加的高斯噪声与其他噪声源的类型相同；此外，两个高斯分布之和仍然是高斯分布，因此隐私机制对统计分析的影响可能更容易理解和纠正。

> **Definition 3.**  ($R\acute{e}nyi$ differential privacy, RDP): If for all the Neighboring dataset *D* and *D′* , the random mechanism $F$ satisfies:
![eq4](image\eq4.png)
- **定义 3**（$R\acute{e}nyi$i differential privacy，RDP）：如果对于所有邻接数据集 *D* 和 *D′*，随机机制(随机函数) $F$ 满足Eq(4)。那么这个机制 $F(x)$ 满足 ($\alpha,\epsilon$)-RDP。Renyi差分隐私(RDP)的思想主要是利用Renyi 散度来衡量两个数据集分布之间的关系。
- 传统差分隐私使用参数 ? 来度量隐私损失，而瑞丽差分隐私RDP则引入了一个参数 α 来定义不同的隐私度量。
- RDP 在实际应用中通常与传统的 ?-差分隐私相结合。例如，可以通过将 RDP 转换为传统的 
?-差分隐私来评估隐私损失。这种转换可以通过以下公式完成：
$\epsilon=(\alpha-1)\cdot\Delta_{\alpha}(M,D,D')$
- 其中$\Delta_{\alpha}(M,D,D')$可以认为是Eq(4)中的ln部分（实际上有些差别）。

> **Sequential Composition:** If $F_1(x)$ satisfies ($\alpha,\epsilon_1$)-RDP, while $F_2(x)$ satisfies ($\alpha,\epsilon_2$)-RDP, then the Composition mechanism of $F_1(x), F_2(x)$ satisfies ($\alpha,\epsilon_1+\epsilon_2$)-RDP.
- 顺序组合：如果 $F_1(x)$ 满足 ($\alpha,\epsilon_1$)-RDP，而 $F_2(x)$  满足  ($\alpha,\epsilon_2$)-RDP, 那么 $F_1(x), F_2(x)$ 的合成机制满足 ($\alpha,\epsilon_1+\epsilon_2$)-RDP。
- 即$\frac{1}{\alpha -1}\ln_{}{(\frac{F_1(x)+F_2(x)}{F_1(x')+F_2(x')})^\alpha }\le \epsilon_1+\epsilon_2$

> **Lemma 2.** RDP for Gaussian mechanism: Gaussian mechanism is the basic mechanism to achieve Renyi differential  privacy. For a function $f$ : $\mathcal{D} \to \mathbb{R}^k$ with sensitivity $s$, a mechanism $F$ follows ($\alpha,\epsilon$)-RDP can be constructed by
![eq5](image\eq5.png)
- **定理 2.** 高斯机制的 RDP：高斯机制是实现RDP的基本机制。对于函数 $f$ ：$\mathcal{D} \to \mathbb{R}^k$，且灵敏度为 $s$，则可通过Eq(5)方法构建一个遵循 ($\alpha,\epsilon$)-RDP的机制 $F$

> **Lemma 3.** From ($\alpha,\epsilon$)-RDP to ($\epsilon,\eth$)-DP: If $F(x)$ satisfies ($\alpha,\epsilon$)-RDP, then for any given $\eth>0$, $F$ satisfies ($\epsilon',\eth$)differential privacy, where $\epsilon'=\epsilon+\ln{}{\frac{1/\eth}{\alpha-1}}$. The value of $\eth$ is generally taken as $\eth\le\frac{1}{n^2}$.
- **定理 3.** 从（($\alpha,\epsilon$)-RDP 到（ ($\epsilon,\eth$)-DP：如果$F(x)$ 满足 ($\alpha,\epsilon$)-RDP，那么对于任意给定的 $\eth>0$, $F$ 满足 ($\epsilon',\eth$)-DP，其中 $\epsilon'=\epsilon+\ln{}{\frac{1/\eth}{\alpha-1}}$ 。$\eth$的值一般取为$\eth\le\frac{1}{n^2}$。
- RDP 更为灵活，将参数 $\alpha$ 和灵敏度 $\Delta f$ 结合在一起。
- RDP 可根据不同的应用场景选择合适的参数 $\alpha$。当 $\alpha=1$ 时，RDP 等于 DP。当 $\alpha>1$ 时，可以提供更强的隐私保护。当 $\sigma^2$ 的值给定时，使用RDP的顺序组成来限制重复应用高斯机制的隐私消耗，然后将 RDP 转换为  ($\epsilon,\eth$)-DP。通过这种方法得到的总隐私消耗通常比直接应用  ($\epsilon,\eth$)-DP  的顺序组合得到的要低得多。基于这一特性，本文还使用RDP来实现局部数据集中的参数扰动。
```c

有点晕

```

## METHOD DESIGN
> This section focuses on a new federated privacy enhancement architecture to achieve client-level privacy protection,low communication overhead, and high robust aggregation. 

### A. System model
> Our aim is to devise a privacy-centric and robust federated learning framework tailored for cross-silo settings.
> Our specific objectives are outlined as follows:
> 	- Communication: We have designed an adaptive parameter upload rate adjustment method based on Top-K, which tightens the lower bound of local information compression rate, effectively reducing communication costs.
> - Robustness: We propose an importance-weighted aggregation method, significantly enhancing the robustness of global model aggregation under differential privacy with noised parameters.
> - Privacy: We strive to achieve client-level differential privacy by employing a dynamic privacy budget adjustment mechanism. This approach facilitates a judicious balance between privacy and accuracy.

- 通信：本文设计了一种基于 Top-K 的自适应参数上传率调整方法，收紧了本地信息压缩速率的下限，有效降低了通信开销。
- 鲁棒性：本文提出了一种重要度加权聚合方法，显著增强了在含有噪声参数的差分隐私环境下全局模型聚合的鲁棒性。
- 隐私：本文采用动态隐私预算调整机制，努力实现客户级差异化隐私保护。这种方法有助于在隐私和准确性之间取得明智的平衡。

> The proposed architecture executes an iterative process, comprising the following steps：
> 1. The server broadcasts the initialized model, the current round’s upload rate, and the privacy budget to all local clients. Each client then performs local stochastic gradient descent using their local data to obtain updated local weight differences.
> 2. To reduce communication costs, Top-K parameter sparsification is performed based on the given upload rate.
> 3. To protect client privacy, Gaussian noise is introduced to perturb the sparsified model parameters, based on the given privacy budget.
> 4. The noised weight difference parameters are uploaded to the server.
> 5. The server performs weighted aggregation of the uploaded model parameters, considering factors such as each client’s data volume, upload rate, and parameter reliability, to obtain a new global model. Additionally, the server dynamically adjusts the privacy budget for the next round based on the performance of the global model and assesses the communication capabilities of local devices by analyzing the time delay in node parameter uploads, thereby dynamically adjusting the parameter upload rate for each client in the next round.

- 拟议的架构执行一个迭代过程，包括以下步骤：
 	- 服务器向所有本地客户端广播初始化模型、本轮上传率和隐私预算。然后，每个客户端利用其本地数据执行本地随机梯度下降，以获得更新后的本地权重。
 	- 为了降低通信成本，Top-K 参数稀疏化是根据给定的上传速率进行的。
 	- 为保护客户隐私，根据给定的隐私预算，引入高斯噪声来扰动稀疏化模型参数。
 	- 将噪声权重差参数上传到服务器。
 	- 服务器对上传的模型参数进行加权聚合，同时考虑每个客户端的数据量、上传率和参数可靠性等因素，从而得到一个新的全局模型。此外，服务器还会根据全局模型的性能动态调整下一轮的隐私预算，并通过分析节点参数上传的时间延迟来评估本地设备的通信能力，从而在下一轮动态调整每个客户端的参数上传率。

- 整个过程如图 2 所示。算法 1 给出了本文训练方案的伪代码。
![图2](image\图2.png)
![算法1](image\算法1.png)
### B. Dynamic privacy budget adjustment
- 动态隐私预算分配的主要目标是在模型准确性和数据隐私之间实现微妙的平衡。我们使用差分隐私来保护隐私，在参数上传到服务器之前为其添加精心设计的噪音。
- 隐私预算越低，噪声量越大，隐私度越高，但参数失真度也越高，从而降低了准确度。
- 传统差分隐私算法的隐私预算在每一轮都是固定的。为了在准确性和隐私之间取得平衡，需要根据模型的实际性能动态调整每一轮的隐私预算。因此，服务器会根据当前一轮全局模型的准确性表现来调整下一轮的隐私预算。
- 调整原则是，如果当前一轮的模型准确性低于预期效果，则增加下一轮的隐私预算。反之，如果当前一轮的模型准确度超过预期效果，下一轮的隐私预算就会减少。重要的是，**预期模型准确度被定义为一个数值范围，而不是单一数值**，这就为调整过程提供了灵活性，以适应各种情况。
- 调整原则如下：

> Step 1: The change in accuracy value of the global model in time window of round $t$ and round $t$ ? 1 is $\Delta acc_{t-1}=acc_t-acc_{t-1}$. If $\Delta acc_{t-1}<0$, it means that the accuracy of the model at the end of the $t$-th training round has decreased instead of increased. In this case, the next round should add less noise, and $\epsilon_{t+1}$ should be larger. Assuming that the amount of noise is desired to be reduced by at least $c$, we can use the following formula (6):
![eq6](image\eq6.png)
 步骤 1：全局模型在第 $t$ 轮和第 $t$ - 1 轮时间窗口中的精度值变化为$\Delta acc_{t-1}=acc_t-acc_{t-1}$。如果 $\Delta acc_{t-1}<0$，则表示第 $t$ 轮训练结束时模型的准确度不升反降。在这种情况下，下一轮增加的噪声应该更少，$\epsilon_{t+1}$ 也应该更大。假设希望噪声量至少减少 $c$.
```c

从第t-1轮到第t轮，模型准确度变化值是负的，说明第t轮的准确度下降了，   
说明第t-1轮的隐私预算太小了（加入了过多的噪声），应该在第t+1轮增大隐私预算（减少噪声）   

```

- 》》》**这里公式6应该给反了或者给早了**，应该是$\epsilon_{t+1}\ge \epsilon_t +c$《《《

> Step 2: If $\Delta acc_{t-2}-\Delta acc_{t-1}\ge d,(\Delta acc_{t-2}>0,\Delta acc_{t-1}>0)$, then it means that the model accuracy at the end of the $t$-th training round has increased but the effect is very little. Then the next round should add less noise, and $\epsilon_{t+1}$ should be larger. In this way, the same result as (6).
- 步骤 2：如果 $\Delta acc_{t-2}-\Delta acc_{t-1}\ge d,(\Delta acc_{t-2}>0,\Delta acc_{t-1}>0)$，则说明第 t 轮训练结束时的模型精度有所提高，但影响很小。那么下一轮增加的噪声应该更少，$\epsilon_{t+1}$也应该更大。这样，结果与（6）相同。
- 》》》**那么文献原文的公式就是写反了**，$\epsilon_{t+1}\ge \epsilon_t +c$才是正确的公式6，**如果理解有错欢迎指出**《《《
```c
两个改变量均大于0，说明模型精度是在提高的，但是第t-2个时间窗口的改变量大于第t-1个时间窗口的改变量，
说明这两个时间窗口下，隐私预算的影响很小，或者说隐私预算有一点点小了，
那么我们在第t个时间窗口（第t+1轮训练）要增加隐私预算。
```
> Step 3: If $\Delta acc_{t-1}-\Delta acc_{t-2}\ge 1,(\Delta acc_{t-1}>0)$, then it means that the accuracy of the model is improved more at the $t$-th training round, and stronger protection of privacy can also be implemented while ensuring the training is carried out properly. Then the noise should be increased in the next round, and $\epsilon_{t+1}$  should be smaller.Similarly, assuming that the amount of noise is desired to increase by at least c, we have:
![eq7](image\eq7.png)
- 步骤 3：如果 $\Delta acc_{t-1}-\Delta acc_{t-2}\ge l,(\Delta acc_{t-1}>0)$，则说明在第 t 轮训练中，模型的准确性得到了更大的提高，在确保训练正常进行的同时，也可以对隐私进行更有力的保护。那么在下一轮训练中，噪声应该增大，$\epsilon_{t+1}$ 应该减小。同样，假设希望噪声量至少增加 c
- 》》》**至少减少c，不是应该$\epsilon_{t+1}\le\epsilon_{t}-c$吗？**《《《
> Step 4: In the remaining cases, $\epsilon_{t+1}$ is the average of the remaining privacy budget, i.e. $\epsilon_{t+1}=\frac{\epsilon- {\textstyle \sum_{i}^{t}}\epsilon_i}{T-t}$.
- 步骤 4：在其余情况下，$\epsilon_{t+1}$是剩余隐私预算的平均值，即$\epsilon_{t+1}=\frac{\epsilon- {\textstyle \sum_{i}^{t}}\epsilon_i}{T-t}$。
- 上文中，阈值 $c$ 用于控制自适应隐私预算调整的幅度。$c$ 值越大，意味着隐私预算的变化越大，这会导致整个训练过程极不稳定。因此，建议选择平均隐私预算的 $1/\upsilon$，其中$\upsilon$为正整数。阈值 $d$ 和 $l$ 决定何时调整隐私预算。当 $d$ 和 $l$ 的值越大，说明训练过程中对不稳定性的容忍度越高，训练过程中的自适应调整次数越少。上述情况下 $c、d、l$ 的具体取值需要根据实际情况进行调整，并根据实验情况找到一个相对合适的值。$T$ 轮训练结束后，总体满足 ($\alpha,\epsilon$)-RDP，满足($\epsilon+\frac{\ln{1/\eth}{}}{\alpha-1}{},\eth$)-DP。

### C. Adaptive upload rate adjustment Top-K
> TOPK is a widely used parameter selection mechanism to reduce communication cost. K is a predefined threshold which balances the training accuracy and communication cost.
> In cross-silo federated learning, there exist variations in device computing power and communication bandwidth among participants. When a participant’s device has limited computing power and communication bandwidth, it may experience prolonged local computation and communication times, or even face challenges in uploading parameters. In such cases, uploading a reduced number of parameters can effectively reduce communication costs.

- TOP-K 是一种广泛使用的参数选择机制，用于降低通信成本。K 是一个预定义的阈值，用于平衡训练精度和通信成本。
- 在跨孤岛联邦学习中，客户端之间的设备计算能力和通信带宽存在差异。当某个客户端的设备计算能力和通信带宽有限时，它可能会经历较长的本地计算和通信时间，甚至在上传参数时面临挑战。在这种情况下，上传较少数量的参数可有效降低通信成本。相反，上传更多参数可以帮助全局模型更好地适应本地数据，提高联合学习的准确性。
- 然而，在以往的工作中，阈值 K 通常是固定的，这一方面没有充分发挥通信条件好的客户端的作用，另一方面也限制了本地信息压缩率的下限，从而在一定程度上增加了通信成本。
- 具体来说，当 K 值过小时，通信条件好的客户端可以在不影响训练进度的情况下上传更多参数，使本地模型更加精确，但由于 K 值固定，只能浪费部分带宽。当 K 值过大时，通信条件差的客户端就需要更长的通信时间才能完成交互，从而拖慢训练进度。因此，必须根据参与设备的实际情况动态调整参数上传比例，在优化通信成本的同时确保 RDP 联邦学习的效率和准确性。
- 可以根据本地节点的时间延迟来评估设备能力，并据此确定下一轮的参数上传速率。

> Local nodes with low time latency are generally considered to have better communication and computation capabilities, while those with high time latency are considered to have lower capabilities.The specific process is shown below:
> - Step 1: Set the initial parameter upload rate $p_0$.
> - Step 2: The server records the parameter upload time for each participant $k$ in the past $r$ rounds separately,${d^{t-r}_k,...,d^t_k}$. Then the average upload duration is $\overline{d_k}=\frac{ {\textstyle \sum_{j}^{r}}d^j_k}{r}$ .
> - Step 3:
![eq8](image\eq8.png)

> - Step 4: The $p^{t+1}_k \times n$ largest parameters need to be selected for upload, where $n$ is the total number of model parameters.
- 一般认为，时间延迟低的本地节点具有较好的通信和计算能力，而时间延迟高的本地节点则能力较低。具体流程如下：
	- 步骤 1：设置初始参数上传速率 $p_0$。
	- 步骤 2：服务器分别记录每个客户端 $k$ 在过去 $r$ 轮中的参数上传时间${d^{t-r}_k,...,d^t_k}$。那么平均上传时间为 $\overline{d_k}=\frac{ {\textstyle \sum_{j}^{r}}d^j_k}{r}$ 。
	- 步骤3：如公式8，其中，$\varrho$ 是需要调整的客户数量的百分比。必须确保 $0<p^{t+1}_k <1$，否则令 $p^{t+1}_k=p^t_k$。
		 ```c 
		
		 如果平均上传时间的数值落在一个较大的区间，说明上传时间太长，即参数上传率太高，应该降低参数上传率。反之应该增加参数上传率。
		
		  ```
	- 步骤 4：需要选择 $p^{t+1}_k \times n$ 个最大参数上传，其中 $n$ 是模型参数的总数。

### D. Weighted Aggregation based on Importance
传统的加权聚合方法根据本地节点的数据量确定权重。在本文看来，除了数据量，还应考虑参与者上传参数的可靠性。差异隐私噪声和外部恶意攻击者的加入会降低聚合结果的准确性。服务器会在聚合前评估接收到的参数的可靠性，并为可靠性较低的参数分配较低的权重，以减少噪声和恶意参数对模型准确性的影响。
> Parameter credibility is assessed based on the similarity between two consecutive rounds of parameters from a node and the similarity with the global parameters. Due to the large number of parameters and fine-tuning based on previous training, parameters from adjacent rounds usually exhibit similar orientations and magnitudes. Therefore, the local parameter similarity between two consecutive parameter uploads by a node can serve as a measure of upload confidence. Moreover, nodes significantly contributing to global parameters generally have similar directions and magnitudes compared to the global parameters. Consequently, the similarity between local parameters uploaded by a node in the current round and the global parameters from the previous round can be used to assess the trustworthiness of parameter uploads.
> Global model weighted aggregation is determined by considering three key factors: the amount of node data, the parameter upload rate, and parameter credibility.
- 参数可信度的评估基于**一个节点连续两轮参数之间的相似性**以及**与全局参数的相似性**。由于参数数量较多，且根据以前的训练进行了微调，相邻轮次的参数通常表现出相似的方向和幅度。因此，**节点连续两次上传参数之间的局部参数相似度可以作为上传可信度的衡量标准**。此外，与全局参数相比，**对全局参数有重要贡献的节点通常具有相似的方向和幅度**。因此，节点在本轮上传的局部参数与上一轮上传的全局参数之间的相似度可用来评估参数上传的可信度。
- 此外，考虑到参数上传率的动态调整，通常**参数上传率高的客户端意味着硬件设施好**，因此应增加其本地模型对全局模型的贡献。
- 全局模型加权聚合由三个关键因素决定：节点数据量、参数上传率和参数可信度。

> Step 1: Calculate the parameter credibility $Cied_k$ of node $k$. According to (10) we can calculate $\cos (\Delta w^{t-1}_k,\Delta w^t_k)$  and $\cos (\Delta w^t_k,\Delta w^{t-1})$  respectively, then we have
![eq9](image\eq9.png)

> The similarity of vectors $A, B$ is calculated by the cosine similarity. That is
![eq10](image\eq10.png)

> As we aim to measure the similarity in direction between the two parameter vectors, we
consider the case of low similarity as the opposite direction.
![eq11](image\eq11.png)

Step 2: Calculate the importance score $Imp^t_k$ based on the amount of data, the parameter upload rate and parameter credibility of node $k$:
![eq12](image\eq12.png)

Step 3: Global parameter weighted aggregation:
![eq13](image\eq13.png)

- 步骤 1：计算节点 $k$ 的参数可信度 $Cied_k$。根据 (10)，我们可以分别计算 $\cos (\Delta w^{t-1}_k,\Delta w^t_k)$和 $\cos (\Delta w^t_k,\Delta w^{t-1})$, 对于公式9，其中 0 < $\beta$ < 1。
	- 	数据量较大的节点通常对全局模型的影响更大，因此，它们在每一轮上传的局部权重可能与上一轮的全局权重更相似。因此，本文为这些节点设置较高的权重值$(1-\beta)$。
	- 向量 $A、B$ 的相似度通过余弦相似度计算得出。其中，$A$ 和 $B$ 表示参数向量，$\cdot$ 表示向量的点积运算，$||A||$ 和 $||B||$ 表示 $A$ 和 $B$ 的 L2 范数。
	- 余弦相似度的范围从-1到1，其中**接近1的值表示两个向量在方向上的相似性较高**。相反，**值越接近-1表示方向上的差异越大**，**值越接近0表示两个向量**方向**上的差异越大**。由于我们的目标是测量两个参数向量之间方向上的相似性，因此我们将低相似性的情况考虑为相反方向。如公式11.
- 步骤2: 根据节点i的数据量、参数上传率和参数可信度计算重要性分数$Imp^t_k$。
	- 其中，$n_k$是每个客户机的数据量。0 < $\gamma_1,\gamma_2,\gamma_3$ < 1， $\gamma_1+\gamma_2+\gamma_3=1$。这三个参数分别决定了聚合时本地数据量的占比、参数上传率和参数可信度。
	- 数据量在神经网络模型的训练中起着至关重要的作用，它在一定程度上决定了局部模型训练精度的上限。因此，在数据异质性的情况下，局部数据的量与局部模型对全局模型的贡献密切相关，而$\gamma_1$通常是三个参数中最大的。 
- 步骤3:全局参数加权聚合，如公式13.

### E. Privacy Analysis
> Assuming a total privacy budget of $\epsilon$, with $T$ total training rounds, and a privacy budget per round of $\epsilon_t$. For a given privacy budget, the Renyi Differential Privacy (RDP) can select an appropriate parameter $\alpha$ such that the conversion of RDP to Differential Privacy (DP) minimizes the privacy budget. Therefore, in each round, different clients k satisfy ($\alpha^k_t, \epsilon_t$)-RDP. According to Lemma 3, it can be converted to DP as ($\epsilon'_t,\eth$)-DP. Given the parallel composition property of DP, each round satisfies ($max(\epsilon'_t),\eth$)-DP. Following the sequential composition property of DP, after $T$ rounds, it satisfies (${\textstyle \sum_{t=1}^{T}}max(\epsilon'_t),T\eth$)-DP, where $\epsilon'_t=\epsilon_t+\frac{ln(1/\eth)}{\alpha^k_t-1}$. 
- 假设总隐私预算为$\epsilon$，总训练轮数为$T$，每轮隐私预算为$\epsilon_t$。对于给定的隐私预算，瑞丽差分隐私(RDP)可以选择合适的参数$\alpha$，使RDP到差分隐私(DP)的转换最小化隐私预算。因此，在每一轮中，不同的客户端$k$满足($\alpha^k_t, \epsilon_t$)-RDP。根据引理3，它可以转换成DP为($\epsilon'_t,\eth$)-DP。鉴于DP的平行组合特性，每一轮都满足($max(\epsilon'_t),\eth$)-DP。根据DP的顺序组成性质，经过$T$轮后，满足(${\textstyle \sum_{t=1}^{T}}max(\epsilon'_t),T\eth$)-DP，其中，$\epsilon'_t=\epsilon_t+\frac{ln(1/\eth)}{\alpha^k_t-1}$。

> Since $\alpha^k_t\in[2,100]$, when $\alpha^k_t=2,\epsilon'_t$ attains its maximum value, which is $\epsilon_t+ln(1/\eth)$, leading to
![eq14](image\eq14.png)

- 由于$\alpha^k_t\in[2,100]$，当$\alpha^k_t=2$时，$,\epsilon'_t$达到最大值$\epsilon_t+ln(1/\eth)$，产生公式14.

> When $\alpha^k_t=100,\epsilon'_t$ attains its minimum value, which is $\epsilon_t+\frac{ln(1/\eth)}{99}$ , leading to
> ![eq15](image\eq15.png)

- 当$\alpha^k_t=100$时，$\epsilon'_t$达到最小值$\epsilon_t+\frac{ln(1/\eth)}{99}$，得到公式15.

> Let $Tln(1/\eth)=\mu$, then there is
![eq16](image\eq16.png)

## 论文实验 

本文主要介绍了针对联邦学习中的通信成本和隐私保护问题的三组实验，并对实验结果进行了分析和比较。

第一组实验是关于通信成本的实验，主要通过调整上传率来降低通信成本并提高模型精度。在MNIST和CIFAR-10数据集上，使用了自适应上传率调整方法（Atop-K）与经典FedAvg算法进行比较。实验结果显示，在不同上传率下，Atop-K方法可以有效地减少通信成本，同时保持较高的模型精度。而在CIFAR-10数据集上，当上传率为0.1时，虽然通信成本较高，但模型精度也有所提升。

第二组实验是关于模型鲁棒性的实验，主要验证了ImpWA方法在防御模型中毒攻击方面的效果。实验中，模拟了模型中毒攻击的情况，并将ImpWA方法与其他经典算法进行比较。实验结果显示，ImpWA方法相对于其他算法具有更高的模型精度和更好的鲁棒性。

第三组实验是关于隐私保护的实验，主要验证了RDP-ImpWA方法在保证模型精度的同时实现有效隐私保护的效果。实验中，使用了不同的隐私预算值和噪声大小，并将RDP-ImpWA方法与其他经典算法进行比较。实验结果显示，RDP-ImpWA方法相对于其他算法可以在保证模型精度的情况下更好地保护用户隐私。

综上所述，本文提出的自适应上传率调整方法、ImpWA方法以及RDP-ImpWA方法都可以有效地解决联邦学习中的通信成本和隐私保护问题，值得进一步研究和应用。
## 总结
这篇文献读起来感觉有点难，但又有点怪。
- 本文主要考虑了四种因素，分别是**隐私安全、聚合鲁棒性、通信开销以及模型准确性**
- 对于隐私性，本文主要采用DP和RDP以及它们的综合应用，还将高斯机制引入RDP。
- 对于通信开销，本文主要是用了Top-K算法， 各客户端选择自己的Top-K参数进行上传。
- 对于模型准确性，本文主要是用了动态隐私预算调整算法，不断调整每一轮的全局隐私预算，不仅改善了通信开销，而且增加了全局模型的准确性。
- 在聚合全局模型时，本文考虑了三重因素，分别是各客户端数据量、各客户端的参数上传率和参数可信度。
## 文章优点
本文提出了一种联邦学习中的隐私增强算法，该算法结合了本地差分隐私、参数稀疏化和加权聚合等技术，旨在提高跨云（跨孤岛）环境下的联邦学习模型训练过程中的隐私保护能力，并在隐私、准确度、通信开销和鲁棒性之间取得平衡。此外，该算法还考虑了调整过程中可能引入的不稳定性问题，并提出了进一步改进的方法以实现更稳定和鲁棒的训练方案。
