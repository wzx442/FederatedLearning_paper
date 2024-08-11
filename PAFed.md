Privacy-Preserving Asynchronous Federated Learning under Non-IID settings——非独立同分布下的异步联邦学习隐私保护
# 来源
- TIFS 2024
# Abstract
 >Existing privacy-preserving asynchronous FL schemes still suffer from the problem of low model accuracy caused by inconsistency between delayed model updates and current model updates, and even cannot adapt well to Non-Independent and Identically  Distributed (Non-IID) settings. 
- 为了解决分布式机器学习中数据孤岛和异构性带来的挑战，保护隐私的异步联邦学习(FL)在学术和工业领域得到了广泛的探索。然而，现有的保护隐私的异步FL方案仍然存在延迟模型更新与当前模型更新不一致导致的模型精度低的问题，甚至不能很好地适应非独立和同分布(Non-IID)的设置。**本文中数据为独立但非同分布的**
- 为了解决这些问题，本文提出了一种基于**交替方向乘子方法**的隐私保护异步联邦学习（**PAFed**），该方法能够在Non-IID设置中实现高精度模型。
- 具体而言，本文利用**向量投影技术**来纠正延迟模型更新与当前模型更新之间的不一致性，从而减少延迟模型更新对当前模型更新聚合的影响。此外，采用了一种基于**交替方向乘子法**的优化方法来适应Non-IID设置，以进一步提高模型的全局精度。
# INTRODUCTION
> However, even if FL does not leak client-side local data, the model may still be vulnerable to inference attacks. Moreover, compared with traditional central learning, FL has many client groups and limited communication resources, which often causes asynchronous update, resulting in a decrease in model accuracy. 
- 即使联邦学习不泄漏客户端本地数据，该模型仍然可能容易受到**推理攻击**。此外，与传统的集中学习相比，联邦学习的客户群体多，通信资源有限，往往导致异步更新，导致模型精度下降。
> However, existing privacypreserving asynchronous FL solutions still face many problems in practical applications.

> The first problem is that existing privacy-preserving asynchronous FL still has the problem of conflicts between delayed model updates and current model updates, resulting in a decrease in global model accuracy. 
- 第一个问题是现有的保护隐私的异步联邦学习仍然存在延迟模型更新与当前模型更新冲突的问题，导致全局模型精度下降。*因为网络延迟等问题存在，一些本地网络会延迟发送本地模型更新*
- 具体来说，在异步环境中，服务器在执行参数聚合时，可能会在不同的时间段接收来自不同客户端的模型更新。随着延迟轮数的增加，延迟客户机的模型更新方向可能会出现显著差异，从而影响全局模型精度。
- 然而，模型精度降低的根本问题是延迟模型更新与当前模型更新的方向不一致。具体来说，当延迟模型更新与当前模型更新之间存在冲突，且两次模型更新的大小有显著差异时，梯度值越大，对聚合梯度的影响就越大，与全局最优值的偏差就越大。
- 因此，解决延迟模型更新方向不一致的问题对于缓解延迟和当前更新之间的冲突，最终提高异步联邦学习系统的整体效率。
> The second problem is that the existing privacy-preserving
asynchronous FL still cannot adapt well to Non-IID settings,
thereby affecting the model accuracy.
- 第二个问题是现有的隐私保护异步联邦学习仍然不能很好地适应Non-IID设置，从而影响模型的准确性。
- 在实际场景中，大量的客户端组导致本地数据分布经常出现Non-IID设置，**客户端之间的本地模型参数可能存在较大差异**。在异步环境中，客户机模型更新之间的差异可能更加突出，从而导致聚合全局模型的准确性降低。许多同步联邦学习下的解决方案并不能很好地应用到异步环境下。
- 同态加密(HE)、安全多方计算(SMC)和差分隐私(DP)在异步通信中得到了广泛的应用。但这些方法都有一些问题，SME会导致通信效率降低，DP因注入噪声会降低模型精度，HE因为单密钥存在安全问题。
- 本文提出了一种Non-IID环境中的高精度隐私保护异步联邦学习（PAFed）。
- 为了解决延迟模型更新与当前模型更新之间的冲突，我们利用**梯度方向来识别延迟更新，并利用向量投影技术来调整延迟更新的方向，从而使其更接近当前平均更新**。 其次，本文基于**交替方向乘子方法优化损失函数**，以适应数据的异质性。
- 采用**多密钥隐私保护框架**，有效防止本地数据泄露，解决单密钥HE机制导致的低隐私安全性问题。
# PRELIMINARIES
> In this section, we introduce the goals of FL, and then
introduce the principles of the Alternating Direction Method of Multipliers (ADMM) and Cheon-Kim-Kim-Song (CKKS)
homomorphic encryption.
- 在本节中，本文将介绍联邦学习的目标，然后介绍交替方向乘子法(ADMM)和Cheon-Kim-Kim-Song (CKKS)同态加密的原理。
## A. Federated Learning (FL)
> FL aims to find the global model $\omega$ that minimize the
weighted average loss of all clients.
## B. Alternating Direction Method of Multipliers
> Alternating Direction Method of Multipliers (ADMM)  is a distributed optimization algorithm derived from the combination of the Douglas-Rachford and Bregman algorithms, primarily utilized to solve constrained optimization problems by decomposing complex challenges into manageable sub-problems for distributed computing and parallel processing.
- ADMM (Alternating Direction Method of Multipliers, ADMM)是Douglas-Rachford和Bregman算法结合而来的分布式优化算法，主要用于将复杂挑战分解为可管理的子问题进行分布式计算和并行处理，从而解决约束优化问题。
## C. Full Homomorphic Encryption (CKKS)

- CKKS是一种基于近似最小编码的全HE方案，允许对加密数据进行加法和乘法运算。CKKS在保证计算精度的同时，实现了较高的加密效率和安全性。
- CKKS包括**Initialization(初始化)**, **Encoding(编码)**,**Decoding(解码)**, **KeyGeneration(产生密钥)**, **Encryption(加密)**, **Decryption(解密)**, **Addition(同态加)**,and **Multiplication(同态乘)**.
# PROBLEM FORMULATION
> In this section, we show the system model, problem definition and design goals, respectively.
## A. System Model
> In this section, we consider asynchronous update environments. The system model of PAFed consists of servers,
online clients, delayed clients and key distribution center
(KDC). 
- 在本节中，我们将考虑异步更新环境。PAFed系统模型由服务器、在线客户端、延迟客户端和密钥分发中心(KDC)组成。
![系统模型](.\img\\PAFed\\系统模型.png)
> KDC: KDC assigns public/secret key pairs to server $S_1$ and all clients.
- KDC: KDC将公钥/密钥对分配给服务器$S_1$和所有客户机。
> Servers: In the asynchronous environment, the server
$S_0$ tracks the status of each client, dynamically adjusts
the aggregation strategy, and then transmits the result to server$S_1$ for key conversion, finally returns aggregation
result to clients.
- 服务器：在异步环境中，服务器$S_0$跟踪每个客户端的状态，动态调整聚合策略，然后将结果传送给服务器 $S_1$  进行密钥转换，最后将聚合结果返回给客户端。
> Online Clients: In each round, online clients receive the
global model and train it locally, then upload updates to
$S_0$ .
- 在线客户端:在每一轮中，在线客户端接收全局模型，然后进行本地训练，最后讲更新上传到$S_0$。
> Delayed clients: Updates uploaded by delayed clients
are not received by S0 immediately due to network delays
and other reasons, but will be received by S0 in several
delayed rounds.
- 延迟客户端：由于网络延迟等原因，延迟客户端上传的更新不会立即被 S0 接收，而是会被 S0 分几轮延迟接收。
- 具体来说，KDC将公钥/密钥对分配给每个客户端和服务器S1(步骤①)。在第T轮中，在线客户端$C_i$更新本地模型$\omega^T_i$并对其进行本地加密(步骤②)，然后将加密后的模型更新[[$\omega^T_i$]]上传到服务器$S_0$(步骤③)。除了接收在线客户端发送的更新外，$S_0$还可以接收T-ts轮的客户端$C^{T-ts}_i$上传的延迟模型更新(步骤④)。然后，服务器$S_0$进行安全聚合，然后将聚合结果发送给$S_1$进行密钥转换(步骤⑤)。最后，$S_0$将全局模型[[$\omega^{T+1}$]]发送给**在线客户端**(步骤⑥)。
## B. Threat Model
> In our system, we assume that **KDC is honest** and **servers
and clients are honest-but-curious**. Specifically, **all entities
honestly follow the initially set learning protocol**, but the
**servers will infer the client’s private information through the
client’s local model, and the client may also curiously infer
other clients’ private information through the global model**.
- 在我们的系统中，我们假设KDC是诚实的，服务器和客户端是诚实但好奇的。具体来说，所有实体都诚实地遵循最初设置的训练协议，但服务器会通过客户端的本地模型推断客户端的私有信息，客户端也可能通过全局模型好奇地推断其他客户端的私有信息。
> Furthermore, the asynchronous environment of the system
involves delayed clients. 
- 此外，系统的异步环境涉及**延迟的客户端**。
- 本文的威胁模型如下：
	+ >Reduction of model accuracy: Since the dataset of client
presents a Non-IID distribution, and the existence of the
delayed client will introduce delay in the communication
process, therefore the accuracy of global model will
eventually be greatly degraded.
	- 降低模型精度：由于客户端数据集呈现Non-IID分布，而延迟客户端的存在会在通信过程中引入延迟，因此全局模型的精度最终会大大降低。
	- > Leakage of data privacy: The central server is honest but
curious, there may be a risk that the server infers private
information from client local data.
	- 数据隐私泄露：中央服务器诚实但好奇，可能存在服务器从客户端本地数据中推断出隐私信息的风险。
## C. Problem Definition
- 给定n个客户端$C$，$C=\{C_1,C_2,...,C_n\}$，它们对应的本地数据为$D,D=\{D_1,D_2,...,D_n\}$，所有客户端有一个共同的全局模型$\omega^T$。考虑异步环境的影响，会存在延迟的客户端，我们进一步划分，假定在第T轮，d个延迟客户端组成的延迟客户端群为$C^T_d=\{C_1,C_2,...,C_d\}$，正常更新的n-d个在线客户端组成的在线客户端群为$C^T_o=\{C_{d+1,}C_{d+2},...,C_n\}$
- 如果在同步环境下，服务器$S_0$需要等待延迟客户端的更新到达才可以更新全局模型，但是在异步环境下，$S_0$可以直接抛弃延迟更新，而是直接使用在线客户端$C^T_o$发来的模型更新$\nabla\omega^T_o=\{\nabla\omega^T_{d+1},\nabla\omega^T_{d+2},...,\nabla\omega^T_n\}$进行全局模型更新，但是这种方法会导致模型精度下降。
- 为了解决这个问题，本文中$S_0$采用基于时间戳的异步聚合。具体来说，在第T轮，$S_0$可以**同时接受**来自$T_j$轮的延迟客户端群$C^{T_j}_d$的模型更新$\nabla\omega^{T_j}_d$以及当前轮的在线客户端群$C^T_o$发来的模型更新$\nabla\omega^T_o$。然后，$S_0$ 根据加权聚合的延迟轮数$T-T_j$给 $\nabla\omega^{T_j}_d$ 分配一个较低的权重值，以弥补梯度的缺失。然而，基于时间戳的异步聚合策略也有缺陷：当$T-T_j$过大时，$\nabla\omega^{T_j}_d$和 $\nabla\omega^T_o$的版本会有很大差异，导致$\nabla\omega^{T_j}_d$和$\nabla\omega^T_o$的梯度方向可能会有很大偏差和冲突。我们可以定义一个冲突度量函数$R(\nabla\omega^T_i,\nabla\omega^{T_j}_j)$来量化两个向量之间的冲突，即$R(\nabla\omega^T_i,\nabla\omega^{T_j}_j)=\nabla\omega^T_i\cdot\nabla\omega^{T_j}_j$，其中$\nabla\omega^{T}_i\in\nabla\omega^T_o,\nabla\omega^{T_j}_j\in\nabla\omega^{T_j}_d$。根据上述定义，我们可以进一步给出问题的定义：
> Definition 1: In the τ -th communication round, the update
$\nabla\omega^T_i\in\mathbb{R}^d$  of online client $C^T_i$ conflicts with the update $\nabla\omega^{T_j}_j\in\mathbb{R}^d$ of delayed client $C^{T_j}_j$, if $R(\nabla\omega^T_i,\nabla\omega^{T_j}_j)$ < 0.
- 定义 1：在第τ 轮通信中，如果$R(\nabla\omega^T_i,\nabla\omega^{T_j}_j)$< 0，则在线客户端$C^T_i$的更新$\nabla\omega^T_i\in\mathbb{R}^d$与延迟客户端$C^{T_j}_j$的更新$\nabla\omega^{T_j}_j\in\mathbb{R}^d$冲突。
- 此外，在Non-IID 环境下，不同客户端$C_i$的数据可能呈现不同的分布特征。例如，$D_i$只包含某个类别的样本，而$D_{j\neq\\i}$可能包含其他类别的样本。这意味着本地模型更新可能会针对特定设备上的数据分布而不是全局数据分布进行优化。(独立，非同分布)
- 此外，每个客户端$C_i$训练出的本地模型$\omega^T_i$可能会有很大差异，导致中央服务器聚合后$\omega^T$的精度下降。如图2所示，$\omega^T_c$指的是使用 SGD 在所有数据上训练的模型，而$\omega^T_f$指的是 fedavg 中的中心服务器聚合$\omega^T_i$和$\omega^T_j$后的模型。
![IID与Non-IID](img\PAFed\IID与Non-IID.png)
- 此外，在单密钥系统中，所有客户端共享一对密钥$K=(pk,sk)$。如果客户端$C_i$的私钥$sk$被泄露，任何客户端$C_{j\neq\\i}$都可能面临信息泄露的风险，如式 7 所示，其中$\omega^{'}_j$是包含客户端 j 本地信息的解密模型。
![公式7](img\PAFed\公式7.png)
## D. Design Goals
>Based on the above discussion, our scheme needs to design a FL that ensures both security and high accuracy in Non-IID settings and asynchronous update scenarios. The design goals are as follows:
	- Model accuracy: Our scheme should achieve high model accuracy under specific Non-IID settings and asynchronous update scenarios. Specifically, our scheme should meet or outperform FedAvg’s model performance, whether in a client environment with Non-IID settings or in asynchronous environment.
	- Privacy: During the entire FL period, neither server $S_0$, $S_1$ nor the client should infer any valid information about other clients except within the scope of their own permissions.
- 基于上述讨论，我们的方案需要设计在非独立同分布环境和异步更新情况下确保安全和高精度的 FL。设计目标如下：
	- 模型精度：我们的方案应在特定的非独立同分布环境和异步更新情况下实现较高的模型准确性。具体来说，无论是在非独立同分布环境的客户端环境下，还是在异步环境下，我们的方案都应达到或超过 FedAvg 的模型性能。
	- 隐私保护：在整个联邦学习期间，服务器$S_0$、$S_1$ 和各客户端均不得推断出其他客户端的任何有效信息，除非是在其自身权限范围内。
# PROPOSED SCHEME
## A. Technical Overview
- 传统的联邦学习不能很好地适应异步环境下的非独立同分布环境，导致模型精度降低。为了解决这个问题，我们在异步环境中引入了 ADMM 算法来优化损失函数，具体如下：
![公式8](img\PAFed\公式8.png)
	- 其中，$y^{T_r}_i\in\mathbb{R}^d$是客户端$C_i$持有的本地对偶变量，$T$指当前回合，$T_r$指客户最近的在线回合，$\rho>0$ 是二次项的系数。具体来说，我们首先更新客户端$C_i$第$T$轮的本地模型$\omega^{T+1}_i$，如公式 9 所示
![公式9](img\PAFed\公式9.png)
	然后按以下步骤更新$y^{T+1}_i$ ：
![公式10](img\PAFed\公式10.png)
最后，服务器$S_0$按如下方式更新全局模型$\omega^{T+1}$:
![公式11](img\PAFed\公式11.png)
	- 其中，$C^T$ 是被选中参与第$T$ 轮训练的客户集，$\eta$是全局学习率。
	>In addition, to better adapt to the asynchronous environment and solve the potential conflict issue between the current round of model update $\nabla\omega^T_i\in\nabla\omega^T_o$ and the delayed model update $\nabla\omega^{T_j}_{j\neq\\i}\in\nabla\omega^T_d(i.e.,\nabla\omega^T_i\cdot\nabla\omega^{T_j}_{j\neq\\i<0})$. We use the ****projection method**** to correct the direction of the delayed average update so that it is close to the current average update.
- 此外，为了更好地适应异步环境，解决本轮模型更新$\nabla\omega^T_i\in\nabla\omega^T_o$与延迟模型更新$\nabla\omega^{T_j}_{j\neq\\i}\in\nabla\omega^T_d$ 之间的潜在冲突问题（即$\nabla\omega^T_i\cdot\nabla\omega^{T_j}_{j\neq\\i<0}$）。我们使用**投影法**修正延迟平均更新的方向，使其接近当前平均更新。
- 》》具体来说，服务器$S_0$接收本轮在线客户端发送的更新$\nabla\omega^T_i$，并计算平均值$\nabla\omega^T_m$。然后，$S_0$记录本轮收到的所有延迟更新$\nabla\omega^T_d=(\nabla\omega^{T_0}_0,...,\nabla\omega^{T_d}_d)$。对于延迟更新，我们采用了多层分类处理的理念。首先，$S_0$判断$\nabla\omega^{T_j}_j$与$\nabla\omega^T_m$是否冲突（即$\nabla\omega^T_m-\nabla\omega^{T_j}_j>0$） 。然后，$S_0$通过**对满足$\nabla\omega^T_m-\nabla\omega^{T_j}_j<0$ 的延迟模型更新进行平均，计算出$\nabla\omega^T_{m_1}$**。同样的，$S_0$首先**计算满足$\nabla\omega^T_m-\nabla\omega^{T_j}_j>0$的所有延迟模型更新$\nabla\omega^{T_j}_j$的平均值，记为$\nabla\omega^T_{m_2}$**，**然后将$\nabla\omega^T_{m_2}$投影到$\nabla\omega^T_m$的法线平面上，以修正$\nabla\omega^T_{m_2}$的方向**，如公式 12 所示。
![公式12](img\PAFed\公式12.png)
- 如图 3 所示，我们将延迟模型更新$\nabla\omega^T_i$ 和$\nabla\omega^T_j$分别投影到平均更新$\nabla\omega^T$的法线平面上，使修正后的模型更新$\nabla\omega^{T^{'}}_i$，$\nabla\omega^{T^{'}}_j$与 $\nabla\omega^T$的方向更加一致。此外，图 3 显示，由$\nabla\omega^{T^{'}}_i$和$\nabla\omega^{T^{'}}_j$ 聚合而成的全局模型$\nabla\omega^{{T+1}^{'}}$与由$\nabla\omega^{T^{'}}_i$和$\nabla\omega^{T^{'}}_j$聚合而成的全局模型$\nabla\omega^{{T+1}^{'}}$相比，更有利于全局模型的性能。
![图3](img\PAFed\图3.png)
> To protect the privacy of $C_i$, we use CKKS to encrypt the model update uploaded by $C_i$ and ensure that servers perform aggregations under the ciphertext, thus preventing local data leakage. Before encrypting the model updates, we first need to quantify the model updates and classify the models, and then calculate the average of each category. Since CKKS supports additive homomorphism and multiplicative homomorphism, we can simply calculate the average value $[[\nabla\omega^T_m]]$. However, it should be noted that CKKS cannot directly support the operation of classification processing, that is, it cannot directly determine whether there is a conflict between $\nabla\omega^{T_j}_j$ and $\nabla\omega^T_m$ under the ciphertext, because $[[\nabla\omega^T_m]]\cdot[[\nabla\omega^{T_j}_j]]$ is a ciphertext value and cannot be directly decrypted by servers.
- 为了保护$C_i$的隐私，我们使用 CKKS 对$C_i$上传的模型更新进行加密，并确保服务器在密文下进行聚合，从而防止本地数据泄露。在对模型更新进行加密之前，我们首先需要对模型更新进行量化和分类，然后计算每个类别的平均值。由于 CKKS 支持加法同态和乘法同态，我们只需计算平均值$[[\nabla\omega^T_m]]$即可。但需要注意的是，CKKS 无法直接支持分类处理操作，即无法直接判断密文下的$\nabla\omega^{T_j}_j$与$\nabla\omega^T_m$之间是否存在冲突，因为$[[\nabla\omega^T_m]]\cdot[[\nabla\omega^{T_j}_j]]$是密文值，服务器无法直接解密。
> To solve this problem, we introduce the algorithm $Min(\cdot,\cdot)$ for computing the minimum value of two ciphertexts,as shown in Algorithm 1. Given two ciphertexts [[a]] and [[b]] encrypted by CKKS, $Min(\cdot,\cdot)$ returns a ciphertext whose decrypted value approximates to the value of a when a ≤ b,otherwise $Min(\cdot,\cdot)$ returns a ciphertext whose decrypted value approximates to the value of b, where a, b ∈ [0, 1].
- 为了解决这个问题，我们引入了算法$Min(\cdot,\cdot)$，用于计算两个密码文本的最小值，如算法 1 所示。给定由 CKKS 加密的两个密码文本 [[a]] 和 [[b]] ，当 a≤b 时，$Min(\cdot,\cdot)$ 返回解密值近似于 a 值的密码文本，否则$Min(\cdot,\cdot)$返回解密值近似于 b 值的密码文本，其中 a, b∈ [0, 1]。
![算法1](img\PAFed\算法1.png)
- 为了实现延迟模型更新的分类，然后对其进行平均，我们应根据算法 4 确定密文下$\nabla\omega^T_m-\nabla\omega^{T_j}_j$与 0 之间的关系。我们首先将$\nabla\omega^T_m$和$\nabla\omega^{T_j}_j$归一化，然后将$(\nabla\omega^T_m\cdot\nabla\omega^{T_j}_j+1)/2$ 和$1/2$ 输入$Min(\cdot,\cdot)$，间接得到$\nabla\omega^T_m-\nabla\omega^{T_j}_j$和 0 的大小关系。此外，我们设置两个变量$[[\nabla\omega^T_{m_1}]]$  , $[[\nabla\omega^T_{m_2}]]$并初始化为 0。对于每个延迟模型更新$\nabla\omega^{T_j}_j$，我们对$[[\nabla\omega^T_{m_1}]]$ ,$[[\nabla\omega^T_{m_2}]]$进行一次累积，如公式13。
![公式13](img\PAFed\公式13.png)
- 最后，$S_0$ 计算$[[\nabla\omega^T_{m_2}]]$的平均值，然后根据公式 14 修正$[[\nabla\omega^T_{m_2}]]$。
![公式14](img\PAFed\公式14.png)
## B. Concrete Construction of PAFed
> We divide the privacy framework into five parts, system initialization, local update and processing, delayed model update classification and correction, model aggregation, and
key transformation, as shown in Fig. 4. Next, we describe its specific process in Algorithm 3.
![图4](img\PAFed\图4.png)
> System initialization. First, KDC selects a security parameter $\lambda$, then initializes like CKKS.Initialization$(1^{\lambda})$, and then assigns a public/secret key pair $(pk_i,sk_i)$ to each client $C_i$ by calling CKKS.KeyGen$(\chi _s;\chi _e)$. Similarly, KDC also assigns a public/secret key pair $(pk_s,sk_s)$ to $S_1$.
- 系统初始化。首先，KDC 选择一个安全参数 λ，然后像 CKKS.Initialization$(1^{\lambda})$ 一样进行初始化，然后通过调用 CKKS.KeyGen$(\chi _s;\chi _e)$ 为每个客户端$C_i$分配一对公开/保密密钥$(pk_i,sk_i)$。同样，KDC 也会为$S_1$ 分配一对公开/保密密钥$(pk_s,sk_s)$。
> Local update and processing. The server $S_0$ initializes the global model $\omega^0$, and each client initializes a dual variable $y^0_i$. For the t-th iteration, $S_0$ selects the client subset $C^T$ of the τ -th round, and sends the global model $\omega^T$ to the client $C_i\in C^T_o$,  where $C^T_o$ refers to a subset of online clients in $C^T$.
- 本地更新和处理服务器。 $S_0$初始化全局模型$\omega^0$，每个客户端初始化一个对偶变量 y^0_i$。在第 t 次迭代中，$S_0$选择第 τ 轮的客户端子集$C^T$，并将全局模型 $\omega^T$发送给客户端$C_i\in C^T_o$, 其中$C^T_o$ 指的是 $C^T$中的在线客户端子集。然后，在线客户端 $C_i\in C^T_o$执行$E_i$轮局部训练，选择最小的一批数据样本 $b\in B$，目标函数如下：
![公式15](img\PAFed\公式15.png)
- 为了求解目标函数，我们使用**交替方向乘子法**，根据公式 9 计算$\omega^{T+1}_i$。如算法 2 所示，在进行 $E_i$ 轮局部训练后，客户端 $C_i$ 根据公式 10 更新$y^{T+1}_i$，然后计算模型更新$\nabla\omega^T_i$，如公式 16 所示。
![公式16](img\PAFed\公式16.png)
![算法2](img\PAFed\算法2.png)
- 此外，为了实现对密文情况下模型更新延迟的修正，$C_i$ 对$\nabla\omega^T_i$进行了标准化处理，如算法 3 第 8 行所示。虽然量化过程会在一定程度上影响模型的准确性，但其影响仍在可接受的范围内，而且量化过程也在一定程度上减少了计算和通信开销。
![算法3](img\PAFed\算法3.png)
- 然后，$C_i$使用$S_1$的公开密钥 $pk_s$进行本地加密，即 $CKKS.Enc_{pk_s}(\nabla\omega^T_i)$。最后，$C_i$将加密结果$[[\nabla\omega^T_i]]$发送给$S_0$。

> Delayed Update Classification and Correction (DUCC). As shown in Algorithm 4, the server $S_0$ receives encrypted model updates $[[\nabla\omega^T_i]]$ sent by online clients $C_i$ , and calculates the average value $[[\nabla\omega^T_m]]$. Then, $S_0$ records all delayed updates received in this round, as denoted $\nabla\omega_d=([[\nabla\omega^{T_0}_0]],...,[[\nabla\omega^{T_d}_d]])$, and initializes $[[\nabla\omega^T_{m_1}]],[[\nabla\omega^T_{m_2}]]$, $n_1,n_2$ to 0. For each delayed model update $[[\nabla\omega^{T_j}_j]]\in H$, $S_0$ executes Eq. 13 once. In addition,
$S_0$ also accumulates $n_1$ and $n_2$, where $n_1$ and $n_2$ respectively record the total accumulated weights of $[[\nabla\omega^T_{m_1}]],[[\nabla\omega^T_{m_2}]]$.
- 延迟更新分类和更正 (DUCC)。如算法 4 所示，服务器 $S_0$ 接收在线客户端$C_i$ 发送的加密模型更新 $[[\nabla\omega^T_i]]$，并计算平均值$[[\nabla\omega^T_m]]$。然后，$S_0$记录这一轮收到的所有延迟更新，表示为 $\nabla\omega_d=([[\nabla\omega^{T_0}_0]],...,[[\nabla\omega^{T_d}_d]])$，并初始化$[[\nabla\omega^T_{m_1}]]、[[\nabla\omega^T_{m_2}]]$、 $n_1、n_2$ 为 0。对于每个延迟模型更新$[[\nabla\omega^{T_j}_j]]\in H$，$S_0$执行公式13 一次。此外，$S_0$ 还会累积 $n_1$和 $n_2$，其中$n_1$和 $n_2$ 分别记录 $[[\nabla\omega^T_{m_1}]]、[[\nabla\omega^T_{m_2}]]$ 的总累积权重。
![算法4](img\PAFed\算法4.png)
- 最后，$S_0$ 用 $[[\nabla\omega^T_{m_1}]]$除以 n1，计算出$[[\nabla\omega^T_{m_1}]]$的平均值。同样，$[[\nabla\omega^T_{m_2}]]$除以 n2，得出 $[[\nabla\omega^T_{m_2}]]$的平均值。然后，$S_0$ 根据公式14 修正$[[\nabla\omega^T_{m_2}]]$。

> Model aggregation. As in Eq. 17, $S_0$ updates the global model $[[\omega^T]]$.
- 模型汇总。如公式 17 所示，$S_0$更新全局模型$[[\omega^T]]$。
![公式17](img\PAFed\公式17.png)
	- 其中，$\alpha_0、\alpha_1、\alpha_2$分别为非延迟模型更新、非冲突延迟模型更新和冲突延迟模型更新的权重值。

> Key transformation. Since $[[\omega^{T+1}]]$ is encrypted by the public key $pk_s$ and cannot be directly decrypted by the clients, we need to perform key conversion. As shown in Algorithm 5, $S_0$ first randomly generates a non-zero vector $r$ as a mask vector, where the size of $r$ is consistent with the size of $[[\omega^{T+1}]]$. Then, $S_0$ locally encrypts r by calling $CKKS.Enc_{pk_s}(r)$. Then,$S_0$ confuses $[[\omega^{T+1}]]$ as $[[\omega^{T+1}]]+[[r]]$. Finally, $S_0$ sends $[[\omega^{T+1}+r]]$ to $S_1$. After receiving $[[\omega^{T+1}+r]]$, $S_1$ first decrypts it with its own private key $sk_s$ to get $\omega^{T+1}+r$, then encrypts $\omega^{T+1}+r$ with each client’s public key by calling
$CKKS.Enc_{pk_i}(\omega^{T+1}+r)$, finally sends the encrypted result to $S_0$.$S_0$ first encrypts $r$ with the public key of each client to obtain $[[r]]$, then removes the mask by $[[\omega^{T+1}+r]]-[[r]]$,finally sends the corresponding $[[\omega^{T+1}]]$ to the next round of clients.
- 密钥转换。由于 $[[\omega^{T+1}]]$是由公钥 $pk_s$ 加密的，客户端无法直接解密，因此我们需要进行密钥转换。如算法 5 所示，$S_0$ 首先随机生成一个非零向量 $r$ 作为掩码向量，其中 $r$ 的大小与 $[[\omega^{T+1}]]$ 的大小一致。然后，$S_0$ 通过调用 $CKKS.Enc_{pk_s}(r)$对 $r$ 进行本地加密。然后，$S_0$ 将 $[[\omega^{T+1}]]$混淆为 $[[\omega^{T+1}]]+[[r]]$。最后，$S_0$ 将 $[[\omega^{T+1}+r]]$发送给 $S_1$。$S_1$ 收到$[[\omega^{T+1}+r]]$ 后，首先用自己的私钥 $sk_s$ 解密，得到 $\omega^{T+1}+r$，然后调用 $CKKS.Enc_{pk_i}(\omega^{T+1}+r)$，用每个客户端的公钥对$\omega^{T+1}+r$进行加密，最后将加密结果发送给 $S_0$。$S_0$首先用每个客户端的公开密钥对$r$  进行加密，得到 $[[r]]$ ，然后通过 $[[\omega^{T+1}+r]]-[[r]]$删除掩码，最后向下一轮客户端发送相应的 $[[\omega^{T+1}]]$。
![算法5](img\PAFed\算法5.png)
# 个人总结
## 本文求解问题
1. 因为网络延迟问题，一些客户端的本地训练结果无法按时发送给聚合服务器，在异步联邦学习环境下会严重影响训练效率、模型精度。
2. 以为联邦学习存在多个客户端，每个客户端的本地数据不可能完全一样，因此数据呈现Non-IID（非独立同分布）现象，在本文中，Non-IID被解释为独立、非同分布，现存的异步联邦学习不能很好地适应Non-IID环境，会影响模型精度。
## 本文解决方法
1. 对于延迟模型更新，本文采用交替方向乘子法，首先将延迟模型更新分为两个部分，冲突延迟模型更新$\nabla\omega^T_{m_2}$和非冲突延迟模型更新$\nabla\omega^T_{m_1}$，对于冲突延迟模型更新$\nabla\omega^T_{m_2}$，将其投影到$\nabla\omega^T_m$（在线客户端模型更新的平均值）的法线平面上，以修正$\nabla\omega^T_{m_2}$的方向，图3表明这样聚合后的全局模型更加准确。
2. 对于Non-IID问题，我们采用ADMM 算法来优化损失函数。
## 隐私保护
- 本文假设KDC是诚实的，服务器和客户端是诚实但好奇的。具体来说，所有实体都诚实地遵循最初设置的训练协议，但服务器会通过客户端的本地模型推断客户端的私有信息，客户端也可能通过全局模型好奇地推断其他客户端的私有信息。
- 本文采用全同态加密来保护本地数据，单密钥系统下所有客户端共享密钥对，并不安全，本文采用CKKS全同态加密算法，每个客户端和$S_1$有自己的密钥对。在发送本地训练结果前，$C_i$会使用$S_1$的公钥进行加密，而聚合过程发生在$S_0$而不是$S_1$，这样，因为$S_0$收到的是加密数据，且$S_0$不具有密钥，而$S_1$虽然拥有密钥，但是$S_1$只会收到来自$S_0$的聚合结果，即使$S_1$解密数据，也无法从中推断$C_i$的数据。

## 思考
1. 加密方法：可以采用其他全同态，同样地，为$C_i$分配密钥对，由服务器进行加解密。
2. 服务器数量：能否只用一台服务器并保证数据安全？似乎不可以
3. 对于延迟模型更新有无其他解决方案？
4. 对于Non-IID的影响是否有其他方法解决？