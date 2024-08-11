Privacy-Preserving Asynchronous Federated Learning under Non-IID settings�����Ƕ���ͬ�ֲ��µ��첽����ѧϰ��˽����
# ��Դ
- TIFS 2024
# Abstract
 >Existing privacy-preserving asynchronous FL schemes still suffer from the problem of low model accuracy caused by inconsistency between delayed model updates and current model updates, and even cannot adapt well to Non-Independent and Identically  Distributed (Non-IID) settings. 
- Ϊ�˽���ֲ�ʽ����ѧϰ�����ݹµ����칹�Դ�������ս��������˽���첽����ѧϰ(FL)��ѧ���͹�ҵ����õ��˹㷺��̽����Ȼ�������еı�����˽���첽FL������Ȼ�����ӳ�ģ�͸����뵱ǰģ�͸��²�һ�µ��µ�ģ�;��ȵ͵����⣬�������ܺܺõ���Ӧ�Ƕ�����ͬ�ֲ�(Non-IID)�����á�**����������Ϊ��������ͬ�ֲ���**
- Ϊ�˽����Щ���⣬���������һ�ֻ���**���淽����ӷ���**����˽�����첽����ѧϰ��**PAFed**�����÷����ܹ���Non-IID������ʵ�ָ߾���ģ�͡�
- ������ԣ���������**����ͶӰ����**�������ӳ�ģ�͸����뵱ǰģ�͸���֮��Ĳ�һ���ԣ��Ӷ������ӳ�ģ�͸��¶Ե�ǰģ�͸��¾ۺϵ�Ӱ�졣���⣬������һ�ֻ���**���淽����ӷ�**���Ż���������ӦNon-IID���ã��Խ�һ�����ģ�͵�ȫ�־��ȡ�
# INTRODUCTION
> However, even if FL does not leak client-side local data, the model may still be vulnerable to inference attacks. Moreover, compared with traditional central learning, FL has many client groups and limited communication resources, which often causes asynchronous update, resulting in a decrease in model accuracy. 
- ��ʹ����ѧϰ��й©�ͻ��˱������ݣ���ģ����Ȼ���������ܵ�**������**�����⣬�봫ͳ�ļ���ѧϰ��ȣ�����ѧϰ�Ŀͻ�Ⱥ��࣬ͨ����Դ���ޣ����������첽���£�����ģ�;����½���
> However, existing privacypreserving asynchronous FL solutions still face many problems in practical applications.

> The first problem is that existing privacy-preserving asynchronous FL still has the problem of conflicts between delayed model updates and current model updates, resulting in a decrease in global model accuracy. 
- ��һ�����������еı�����˽���첽����ѧϰ��Ȼ�����ӳ�ģ�͸����뵱ǰģ�͸��³�ͻ�����⣬����ȫ��ģ�;����½���*��Ϊ�����ӳٵ�������ڣ�һЩ����������ӳٷ��ͱ���ģ�͸���*
- ������˵�����첽�����У���������ִ�в����ۺ�ʱ�����ܻ��ڲ�ͬ��ʱ��ν������Բ�ͬ�ͻ��˵�ģ�͸��¡������ӳ����������ӣ��ӳٿͻ�����ģ�͸��·�����ܻ�����������죬�Ӷ�Ӱ��ȫ��ģ�;��ȡ�
- Ȼ����ģ�;��Ƚ��͵ĸ����������ӳ�ģ�͸����뵱ǰģ�͸��µķ���һ�¡�������˵�����ӳ�ģ�͸����뵱ǰģ�͸���֮����ڳ�ͻ��������ģ�͸��µĴ�С����������ʱ���ݶ�ֵԽ�󣬶Ծۺ��ݶȵ�Ӱ���Խ����ȫ������ֵ��ƫ���Խ��
- ��ˣ�����ӳ�ģ�͸��·���һ�µ�������ڻ����ӳٺ͵�ǰ����֮��ĳ�ͻ����������첽����ѧϰϵͳ������Ч�ʡ�
> The second problem is that the existing privacy-preserving
asynchronous FL still cannot adapt well to Non-IID settings,
thereby affecting the model accuracy.
- �ڶ������������е���˽�����첽����ѧϰ��Ȼ���ܺܺõ���ӦNon-IID���ã��Ӷ�Ӱ��ģ�͵�׼ȷ�ԡ�
- ��ʵ�ʳ����У������Ŀͻ����鵼�±������ݷֲ���������Non-IID���ã�**�ͻ���֮��ı���ģ�Ͳ������ܴ��ڽϴ����**�����첽�����У��ͻ���ģ�͸���֮��Ĳ�����ܸ���ͻ�����Ӷ����¾ۺ�ȫ��ģ�͵�׼ȷ�Խ��͡����ͬ������ѧϰ�µĽ�����������ܺܺõ�Ӧ�õ��첽�����¡�
- ̬ͬ����(HE)����ȫ�෽����(SMC)�Ͳ����˽(DP)���첽ͨ���еõ��˹㷺��Ӧ�á�����Щ��������һЩ���⣬SME�ᵼ��ͨ��Ч�ʽ��ͣ�DP��ע�������ή��ģ�;��ȣ�HE��Ϊ����Կ���ڰ�ȫ���⡣
- ���������һ��Non-IID�����еĸ߾�����˽�����첽����ѧϰ��PAFed����
- Ϊ�˽���ӳ�ģ�͸����뵱ǰģ�͸���֮��ĳ�ͻ����������**�ݶȷ�����ʶ���ӳٸ��£�����������ͶӰ�����������ӳٸ��µķ��򣬴Ӷ�ʹ����ӽ���ǰƽ������**�� ��Σ����Ļ���**���淽����ӷ����Ż���ʧ����**������Ӧ���ݵ������ԡ�
- ����**����Կ��˽�������**����Ч��ֹ��������й¶���������ԿHE���Ƶ��µĵ���˽��ȫ�����⡣
# PRELIMINARIES
> In this section, we introduce the goals of FL, and then
introduce the principles of the Alternating Direction Method of Multipliers (ADMM) and Cheon-Kim-Kim-Song (CKKS)
homomorphic encryption.
- �ڱ����У����Ľ���������ѧϰ��Ŀ�꣬Ȼ����ܽ��淽����ӷ�(ADMM)��Cheon-Kim-Kim-Song (CKKS)̬ͬ���ܵ�ԭ��
## A. Federated Learning (FL)
> FL aims to find the global model $\omega$ that minimize the
weighted average loss of all clients.
## B. Alternating Direction Method of Multipliers
> Alternating Direction Method of Multipliers (ADMM)  is a distributed optimization algorithm derived from the combination of the Douglas-Rachford and Bregman algorithms, primarily utilized to solve constrained optimization problems by decomposing complex challenges into manageable sub-problems for distributed computing and parallel processing.
- ADMM (Alternating Direction Method of Multipliers, ADMM)��Douglas-Rachford��Bregman�㷨��϶����ķֲ�ʽ�Ż��㷨����Ҫ���ڽ�������ս�ֽ�Ϊ�ɹ������������зֲ�ʽ����Ͳ��д����Ӷ����Լ���Ż����⡣
## C. Full Homomorphic Encryption (CKKS)

- CKKS��һ�ֻ��ڽ�����С�����ȫHE����������Լ������ݽ��мӷ��ͳ˷����㡣CKKS�ڱ�֤���㾫�ȵ�ͬʱ��ʵ���˽ϸߵļ���Ч�ʺͰ�ȫ�ԡ�
- CKKS����**Initialization(��ʼ��)**, **Encoding(����)**,**Decoding(����)**, **KeyGeneration(������Կ)**, **Encryption(����)**, **Decryption(����)**, **Addition(̬ͬ��)**,and **Multiplication(̬ͬ��)**.
# PROBLEM FORMULATION
> In this section, we show the system model, problem definition and design goals, respectively.
## A. System Model
> In this section, we consider asynchronous update environments. The system model of PAFed consists of servers,
online clients, delayed clients and key distribution center
(KDC). 
- �ڱ����У����ǽ������첽���»�����PAFedϵͳģ���ɷ����������߿ͻ��ˡ��ӳٿͻ��˺���Կ�ַ�����(KDC)��ɡ�
![ϵͳģ��](.\img\\PAFed\\ϵͳģ��.png)
> KDC: KDC assigns public/secret key pairs to server $S_1$ and all clients.
- KDC: KDC����Կ/��Կ�Է����������$S_1$�����пͻ�����
> Servers: In the asynchronous environment, the server
$S_0$ tracks the status of each client, dynamically adjusts
the aggregation strategy, and then transmits the result to server$S_1$ for key conversion, finally returns aggregation
result to clients.
- �����������첽�����У�������$S_0$����ÿ���ͻ��˵�״̬����̬�����ۺϲ��ԣ�Ȼ�󽫽�����͸������� $S_1$  ������Կת������󽫾ۺϽ�����ظ��ͻ��ˡ�
> Online Clients: In each round, online clients receive the
global model and train it locally, then upload updates to
$S_0$ .
- ���߿ͻ���:��ÿһ���У����߿ͻ��˽���ȫ��ģ�ͣ�Ȼ����б���ѵ������󽲸����ϴ���$S_0$��
> Delayed clients: Updates uploaded by delayed clients
are not received by S0 immediately due to network delays
and other reasons, but will be received by S0 in several
delayed rounds.
- �ӳٿͻ��ˣ����������ӳٵ�ԭ���ӳٿͻ����ϴ��ĸ��²��������� S0 ���գ����ǻᱻ S0 �ּ����ӳٽ��ա�
- ������˵��KDC����Կ/��Կ�Է����ÿ���ͻ��˺ͷ�����S1(�����)���ڵ�T���У����߿ͻ���$C_i$���±���ģ��$\omega^T_i$��������б��ؼ���(�����)��Ȼ�󽫼��ܺ��ģ�͸���[[$\omega^T_i$]]�ϴ���������$S_0$(�����)�����˽������߿ͻ��˷��͵ĸ����⣬$S_0$�����Խ���T-ts�ֵĿͻ���$C^{T-ts}_i$�ϴ����ӳ�ģ�͸���(�����)��Ȼ�󣬷�����$S_0$���а�ȫ�ۺϣ�Ȼ�󽫾ۺϽ�����͸�$S_1$������Կת��(�����)�����$S_0$��ȫ��ģ��[[$\omega^{T+1}$]]���͸�**���߿ͻ���**(�����)��
## B. Threat Model
> In our system, we assume that **KDC is honest** and **servers
and clients are honest-but-curious**. Specifically, **all entities
honestly follow the initially set learning protocol**, but the
**servers will infer the client��s private information through the
client��s local model, and the client may also curiously infer
other clients�� private information through the global model**.
- �����ǵ�ϵͳ�У����Ǽ���KDC�ǳ�ʵ�ģ��������Ϳͻ����ǳ�ʵ������ġ�������˵������ʵ�嶼��ʵ����ѭ������õ�ѵ��Э�飬����������ͨ���ͻ��˵ı���ģ���ƶϿͻ��˵�˽����Ϣ���ͻ���Ҳ����ͨ��ȫ��ģ�ͺ�����ƶ������ͻ��˵�˽����Ϣ��
> Furthermore, the asynchronous environment of the system
involves delayed clients. 
- ���⣬ϵͳ���첽�����漰**�ӳٵĿͻ���**��
- ���ĵ���вģ�����£�
	+ >Reduction of model accuracy: Since the dataset of client
presents a Non-IID distribution, and the existence of the
delayed client will introduce delay in the communication
process, therefore the accuracy of global model will
eventually be greatly degraded.
	- ����ģ�;��ȣ����ڿͻ������ݼ�����Non-IID�ֲ������ӳٿͻ��˵Ĵ��ڻ���ͨ�Ź����������ӳ٣����ȫ��ģ�͵ľ������ջ��󽵵͡�
	- > Leakage of data privacy: The central server is honest but
curious, there may be a risk that the server infers private
information from client local data.
	- ������˽й¶�������������ʵ�����棬���ܴ��ڷ������ӿͻ��˱����������ƶϳ���˽��Ϣ�ķ��ա�
## C. Problem Definition
- ����n���ͻ���$C$��$C=\{C_1,C_2,...,C_n\}$�����Ƕ�Ӧ�ı�������Ϊ$D,D=\{D_1,D_2,...,D_n\}$�����пͻ�����һ����ͬ��ȫ��ģ��$\omega^T$�������첽������Ӱ�죬������ӳٵĿͻ��ˣ����ǽ�һ�����֣��ٶ��ڵ�T�֣�d���ӳٿͻ�����ɵ��ӳٿͻ���ȺΪ$C^T_d=\{C_1,C_2,...,C_d\}$���������µ�n-d�����߿ͻ�����ɵ����߿ͻ���ȺΪ$C^T_o=\{C_{d+1,}C_{d+2},...,C_n\}$
- �����ͬ�������£�������$S_0$��Ҫ�ȴ��ӳٿͻ��˵ĸ��µ���ſ��Ը���ȫ��ģ�ͣ��������첽�����£�$S_0$����ֱ�������ӳٸ��£�����ֱ��ʹ�����߿ͻ���$C^T_o$������ģ�͸���$\nabla\omega^T_o=\{\nabla\omega^T_{d+1},\nabla\omega^T_{d+2},...,\nabla\omega^T_n\}$����ȫ��ģ�͸��£��������ַ����ᵼ��ģ�;����½���
- Ϊ�˽��������⣬������$S_0$���û���ʱ������첽�ۺϡ�������˵���ڵ�T�֣�$S_0$����**ͬʱ����**����$T_j$�ֵ��ӳٿͻ���Ⱥ$C^{T_j}_d$��ģ�͸���$\nabla\omega^{T_j}_d$�Լ���ǰ�ֵ����߿ͻ���Ⱥ$C^T_o$������ģ�͸���$\nabla\omega^T_o$��Ȼ��$S_0$ ���ݼ�Ȩ�ۺϵ��ӳ�����$T-T_j$�� $\nabla\omega^{T_j}_d$ ����һ���ϵ͵�Ȩ��ֵ�����ֲ��ݶȵ�ȱʧ��Ȼ��������ʱ������첽�ۺϲ���Ҳ��ȱ�ݣ���$T-T_j$����ʱ��$\nabla\omega^{T_j}_d$�� $\nabla\omega^T_o$�İ汾���кܴ���죬����$\nabla\omega^{T_j}_d$��$\nabla\omega^T_o$���ݶȷ�����ܻ��кܴ�ƫ��ͳ�ͻ�����ǿ��Զ���һ����ͻ��������$R(\nabla\omega^T_i,\nabla\omega^{T_j}_j)$��������������֮��ĳ�ͻ����$R(\nabla\omega^T_i,\nabla\omega^{T_j}_j)=\nabla\omega^T_i\cdot\nabla\omega^{T_j}_j$������$\nabla\omega^{T}_i\in\nabla\omega^T_o,\nabla\omega^{T_j}_j\in\nabla\omega^{T_j}_d$�������������壬���ǿ��Խ�һ����������Ķ��壺
> Definition 1: In the �� -th communication round, the update
$\nabla\omega^T_i\in\mathbb{R}^d$  of online client $C^T_i$ conflicts with the update $\nabla\omega^{T_j}_j\in\mathbb{R}^d$ of delayed client $C^{T_j}_j$, if $R(\nabla\omega^T_i,\nabla\omega^{T_j}_j)$ < 0.
- ���� 1���ڵڦ� ��ͨ���У����$R(\nabla\omega^T_i,\nabla\omega^{T_j}_j)$< 0�������߿ͻ���$C^T_i$�ĸ���$\nabla\omega^T_i\in\mathbb{R}^d$���ӳٿͻ���$C^{T_j}_j$�ĸ���$\nabla\omega^{T_j}_j\in\mathbb{R}^d$��ͻ��
- ���⣬��Non-IID �����£���ͬ�ͻ���$C_i$�����ݿ��ܳ��ֲ�ͬ�ķֲ����������磬$D_i$ֻ����ĳ��������������$D_{j\neq\\i}$���ܰ���������������������ζ�ű���ģ�͸��¿��ܻ�����ض��豸�ϵ����ݷֲ�������ȫ�����ݷֲ������Ż���(��������ͬ�ֲ�)
- ���⣬ÿ���ͻ���$C_i$ѵ�����ı���ģ��$\omega^T_i$���ܻ��кܴ���죬��������������ۺϺ�$\omega^T$�ľ����½�����ͼ2��ʾ��$\omega^T_c$ָ����ʹ�� SGD ������������ѵ����ģ�ͣ���$\omega^T_f$ָ���� fedavg �е����ķ������ۺ�$\omega^T_i$��$\omega^T_j$���ģ�͡�
![IID��Non-IID](img\PAFed\IID��Non-IID.png)
- ���⣬�ڵ���Կϵͳ�У����пͻ��˹���һ����Կ$K=(pk,sk)$������ͻ���$C_i$��˽Կ$sk$��й¶���κοͻ���$C_{j\neq\\i}$������������Ϣй¶�ķ��գ���ʽ 7 ��ʾ������$\omega^{'}_j$�ǰ����ͻ��� j ������Ϣ�Ľ���ģ�͡�
![��ʽ7](img\PAFed\��ʽ7.png)
## D. Design Goals
>Based on the above discussion, our scheme needs to design a FL that ensures both security and high accuracy in Non-IID settings and asynchronous update scenarios. The design goals are as follows:
	- Model accuracy: Our scheme should achieve high model accuracy under specific Non-IID settings and asynchronous update scenarios. Specifically, our scheme should meet or outperform FedAvg��s model performance, whether in a client environment with Non-IID settings or in asynchronous environment.
	- Privacy: During the entire FL period, neither server $S_0$, $S_1$ nor the client should infer any valid information about other clients except within the scope of their own permissions.
- �����������ۣ����ǵķ�����Ҫ����ڷǶ���ͬ�ֲ��������첽���������ȷ����ȫ�͸߾��ȵ� FL�����Ŀ�����£�
	- ģ�;��ȣ����ǵķ���Ӧ���ض��ķǶ���ͬ�ֲ��������첽���������ʵ�ֽϸߵ�ģ��׼ȷ�ԡ�������˵���������ڷǶ���ͬ�ֲ������Ŀͻ��˻����£��������첽�����£����ǵķ�����Ӧ�ﵽ�򳬹� FedAvg ��ģ�����ܡ�
	- ��˽����������������ѧϰ�ڼ䣬������$S_0$��$S_1$ �͸��ͻ��˾������ƶϳ������ͻ��˵��κ���Ч��Ϣ����������������Ȩ�޷�Χ�ڡ�
# PROPOSED SCHEME
## A. Technical Overview
- ��ͳ������ѧϰ���ܺܺõ���Ӧ�첽�����µķǶ���ͬ�ֲ�����������ģ�;��Ƚ��͡�Ϊ�˽��������⣬�������첽������������ ADMM �㷨���Ż���ʧ�������������£�
![��ʽ8](img\PAFed\��ʽ8.png)
	- ���У�$y^{T_r}_i\in\mathbb{R}^d$�ǿͻ���$C_i$���еı��ض�ż������$T$ָ��ǰ�غϣ�$T_r$ָ�ͻ���������߻غϣ�$\rho>0$ �Ƕ������ϵ����������˵���������ȸ��¿ͻ���$C_i$��$T$�ֵı���ģ��$\omega^{T+1}_i$���繫ʽ 9 ��ʾ
![��ʽ9](img\PAFed\��ʽ9.png)
	Ȼ�����²������$y^{T+1}_i$ ��
![��ʽ10](img\PAFed\��ʽ10.png)
��󣬷�����$S_0$�����·�ʽ����ȫ��ģ��$\omega^{T+1}$:
![��ʽ11](img\PAFed\��ʽ11.png)
	- ���У�$C^T$ �Ǳ�ѡ�в����$T$ ��ѵ���Ŀͻ�����$\eta$��ȫ��ѧϰ�ʡ�
	>In addition, to better adapt to the asynchronous environment and solve the potential conflict issue between the current round of model update $\nabla\omega^T_i\in\nabla\omega^T_o$ and the delayed model update $\nabla\omega^{T_j}_{j\neq\\i}\in\nabla\omega^T_d(i.e.,\nabla\omega^T_i\cdot\nabla\omega^{T_j}_{j\neq\\i<0})$. We use the ****projection method**** to correct the direction of the delayed average update so that it is close to the current average update.
- ���⣬Ϊ�˸��õ���Ӧ�첽�������������ģ�͸���$\nabla\omega^T_i\in\nabla\omega^T_o$���ӳ�ģ�͸���$\nabla\omega^{T_j}_{j\neq\\i}\in\nabla\omega^T_d$ ֮���Ǳ�ڳ�ͻ���⣨��$\nabla\omega^T_i\cdot\nabla\omega^{T_j}_{j\neq\\i<0}$��������ʹ��**ͶӰ��**�����ӳ�ƽ�����µķ���ʹ��ӽ���ǰƽ�����¡�
- ����������˵��������$S_0$���ձ������߿ͻ��˷��͵ĸ���$\nabla\omega^T_i$��������ƽ��ֵ$\nabla\omega^T_m$��Ȼ��$S_0$��¼�����յ��������ӳٸ���$\nabla\omega^T_d=(\nabla\omega^{T_0}_0,...,\nabla\omega^{T_d}_d)$�������ӳٸ��£����ǲ����˶����ദ���������ȣ�$S_0$�ж�$\nabla\omega^{T_j}_j$��$\nabla\omega^T_m$�Ƿ��ͻ����$\nabla\omega^T_m-\nabla\omega^{T_j}_j>0$�� ��Ȼ��$S_0$ͨ��**������$\nabla\omega^T_m-\nabla\omega^{T_j}_j<0$ ���ӳ�ģ�͸��½���ƽ���������$\nabla\omega^T_{m_1}$**��ͬ���ģ�$S_0$����**��������$\nabla\omega^T_m-\nabla\omega^{T_j}_j>0$�������ӳ�ģ�͸���$\nabla\omega^{T_j}_j$��ƽ��ֵ����Ϊ$\nabla\omega^T_{m_2}$**��**Ȼ��$\nabla\omega^T_{m_2}$ͶӰ��$\nabla\omega^T_m$�ķ���ƽ���ϣ�������$\nabla\omega^T_{m_2}$�ķ���**���繫ʽ 12 ��ʾ��
![��ʽ12](img\PAFed\��ʽ12.png)
- ��ͼ 3 ��ʾ�����ǽ��ӳ�ģ�͸���$\nabla\omega^T_i$ ��$\nabla\omega^T_j$�ֱ�ͶӰ��ƽ������$\nabla\omega^T$�ķ���ƽ���ϣ�ʹ�������ģ�͸���$\nabla\omega^{T^{'}}_i$��$\nabla\omega^{T^{'}}_j$�� $\nabla\omega^T$�ķ������һ�¡����⣬ͼ 3 ��ʾ����$\nabla\omega^{T^{'}}_i$��$\nabla\omega^{T^{'}}_j$ �ۺ϶��ɵ�ȫ��ģ��$\nabla\omega^{{T+1}^{'}}$����$\nabla\omega^{T^{'}}_i$��$\nabla\omega^{T^{'}}_j$�ۺ϶��ɵ�ȫ��ģ��$\nabla\omega^{{T+1}^{'}}$��ȣ���������ȫ��ģ�͵����ܡ�
![ͼ3](img\PAFed\ͼ3.png)
> To protect the privacy of $C_i$, we use CKKS to encrypt the model update uploaded by $C_i$ and ensure that servers perform aggregations under the ciphertext, thus preventing local data leakage. Before encrypting the model updates, we first need to quantify the model updates and classify the models, and then calculate the average of each category. Since CKKS supports additive homomorphism and multiplicative homomorphism, we can simply calculate the average value $[[\nabla\omega^T_m]]$. However, it should be noted that CKKS cannot directly support the operation of classification processing, that is, it cannot directly determine whether there is a conflict between $\nabla\omega^{T_j}_j$ and $\nabla\omega^T_m$ under the ciphertext, because $[[\nabla\omega^T_m]]\cdot[[\nabla\omega^{T_j}_j]]$ is a ciphertext value and cannot be directly decrypted by servers.
- Ϊ�˱���$C_i$����˽������ʹ�� CKKS ��$C_i$�ϴ���ģ�͸��½��м��ܣ���ȷ���������������½��оۺϣ��Ӷ���ֹ��������й¶���ڶ�ģ�͸��½��м���֮ǰ������������Ҫ��ģ�͸��½��������ͷ��࣬Ȼ�����ÿ������ƽ��ֵ������ CKKS ֧�ּӷ�̬ͬ�ͳ˷�̬ͬ������ֻ�����ƽ��ֵ$[[\nabla\omega^T_m]]$���ɡ�����Ҫע����ǣ�CKKS �޷�ֱ��֧�ַ��ദ����������޷�ֱ���ж������µ�$\nabla\omega^{T_j}_j$��$\nabla\omega^T_m$֮���Ƿ���ڳ�ͻ����Ϊ$[[\nabla\omega^T_m]]\cdot[[\nabla\omega^{T_j}_j]]$������ֵ���������޷�ֱ�ӽ��ܡ�
> To solve this problem, we introduce the algorithm $Min(\cdot,\cdot)$ for computing the minimum value of two ciphertexts,as shown in Algorithm 1. Given two ciphertexts [[a]] and [[b]] encrypted by CKKS, $Min(\cdot,\cdot)$ returns a ciphertext whose decrypted value approximates to the value of a when a �� b,otherwise $Min(\cdot,\cdot)$ returns a ciphertext whose decrypted value approximates to the value of b, where a, b �� [0, 1].
- Ϊ�˽��������⣬�����������㷨$Min(\cdot,\cdot)$�����ڼ������������ı�����Сֵ�����㷨 1 ��ʾ�������� CKKS ���ܵ����������ı� [[a]] �� [[b]] ���� a��b ʱ��$Min(\cdot,\cdot)$ ���ؽ���ֵ������ a ֵ�������ı�������$Min(\cdot,\cdot)$���ؽ���ֵ������ b ֵ�������ı������� a, b�� [0, 1]��
![�㷨1](img\PAFed\�㷨1.png)
- Ϊ��ʵ���ӳ�ģ�͸��µķ��࣬Ȼ��������ƽ��������Ӧ�����㷨 4 ȷ��������$\nabla\omega^T_m-\nabla\omega^{T_j}_j$�� 0 ֮��Ĺ�ϵ���������Ƚ�$\nabla\omega^T_m$��$\nabla\omega^{T_j}_j$��һ����Ȼ��$(\nabla\omega^T_m\cdot\nabla\omega^{T_j}_j+1)/2$ ��$1/2$ ����$Min(\cdot,\cdot)$����ӵõ�$\nabla\omega^T_m-\nabla\omega^{T_j}_j$�� 0 �Ĵ�С��ϵ�����⣬����������������$[[\nabla\omega^T_{m_1}]]$  , $[[\nabla\omega^T_{m_2}]]$����ʼ��Ϊ 0������ÿ���ӳ�ģ�͸���$\nabla\omega^{T_j}_j$�����Ƕ�$[[\nabla\omega^T_{m_1}]]$ ,$[[\nabla\omega^T_{m_2}]]$����һ���ۻ����繫ʽ13��
![��ʽ13](img\PAFed\��ʽ13.png)
- ���$S_0$ ����$[[\nabla\omega^T_{m_2}]]$��ƽ��ֵ��Ȼ����ݹ�ʽ 14 ����$[[\nabla\omega^T_{m_2}]]$��
![��ʽ14](img\PAFed\��ʽ14.png)
## B. Concrete Construction of PAFed
> We divide the privacy framework into five parts, system initialization, local update and processing, delayed model update classification and correction, model aggregation, and
key transformation, as shown in Fig. 4. Next, we describe its specific process in Algorithm 3.
![ͼ4](img\PAFed\ͼ4.png)
> System initialization. First, KDC selects a security parameter $\lambda$, then initializes like CKKS.Initialization$(1^{\lambda})$, and then assigns a public/secret key pair $(pk_i,sk_i)$ to each client $C_i$ by calling CKKS.KeyGen$(\chi _s;\chi _e)$. Similarly, KDC also assigns a public/secret key pair $(pk_s,sk_s)$ to $S_1$.
- ϵͳ��ʼ�������ȣ�KDC ѡ��һ����ȫ���� �ˣ�Ȼ���� CKKS.Initialization$(1^{\lambda})$ һ�����г�ʼ����Ȼ��ͨ������ CKKS.KeyGen$(\chi _s;\chi _e)$ Ϊÿ���ͻ���$C_i$����һ�Թ���/������Կ$(pk_i,sk_i)$��ͬ����KDC Ҳ��Ϊ$S_1$ ����һ�Թ���/������Կ$(pk_s,sk_s)$��
> Local update and processing. The server $S_0$ initializes the global model $\omega^0$, and each client initializes a dual variable $y^0_i$. For the t-th iteration, $S_0$ selects the client subset $C^T$ of the �� -th round, and sends the global model $\omega^T$ to the client $C_i\in C^T_o$,  where $C^T_o$ refers to a subset of online clients in $C^T$.
- ���ظ��ºʹ���������� $S_0$��ʼ��ȫ��ģ��$\omega^0$��ÿ���ͻ��˳�ʼ��һ����ż���� y^0_i$���ڵ� t �ε����У�$S_0$ѡ��� �� �ֵĿͻ����Ӽ�$C^T$������ȫ��ģ�� $\omega^T$���͸��ͻ���$C_i\in C^T_o$, ����$C^T_o$ ָ���� $C^T$�е����߿ͻ����Ӽ���Ȼ�����߿ͻ��� $C_i\in C^T_o$ִ��$E_i$�־ֲ�ѵ����ѡ����С��һ���������� $b\in B$��Ŀ�꺯�����£�
![��ʽ15](img\PAFed\��ʽ15.png)
- Ϊ�����Ŀ�꺯��������ʹ��**���淽����ӷ�**�����ݹ�ʽ 9 ����$\omega^{T+1}_i$�����㷨 2 ��ʾ���ڽ��� $E_i$ �־ֲ�ѵ���󣬿ͻ��� $C_i$ ���ݹ�ʽ 10 ����$y^{T+1}_i$��Ȼ�����ģ�͸���$\nabla\omega^T_i$���繫ʽ 16 ��ʾ��
![��ʽ16](img\PAFed\��ʽ16.png)
![�㷨2](img\PAFed\�㷨2.png)
- ���⣬Ϊ��ʵ�ֶ����������ģ�͸����ӳٵ�������$C_i$ ��$\nabla\omega^T_i$�����˱�׼���������㷨 3 �� 8 ����ʾ����Ȼ�������̻���һ���̶���Ӱ��ģ�͵�׼ȷ�ԣ�����Ӱ�����ڿɽ��ܵķ�Χ�ڣ�������������Ҳ��һ���̶��ϼ����˼����ͨ�ſ�����
![�㷨3](img\PAFed\�㷨3.png)
- Ȼ��$C_i$ʹ��$S_1$�Ĺ�����Կ $pk_s$���б��ؼ��ܣ��� $CKKS.Enc_{pk_s}(\nabla\omega^T_i)$�����$C_i$�����ܽ��$[[\nabla\omega^T_i]]$���͸�$S_0$��

> Delayed Update Classification and Correction (DUCC). As shown in Algorithm 4, the server $S_0$ receives encrypted model updates $[[\nabla\omega^T_i]]$ sent by online clients $C_i$ , and calculates the average value $[[\nabla\omega^T_m]]$. Then, $S_0$ records all delayed updates received in this round, as denoted $\nabla\omega_d=([[\nabla\omega^{T_0}_0]],...,[[\nabla\omega^{T_d}_d]])$, and initializes $[[\nabla\omega^T_{m_1}]],[[\nabla\omega^T_{m_2}]]$, $n_1,n_2$ to 0. For each delayed model update $[[\nabla\omega^{T_j}_j]]\in H$, $S_0$ executes Eq. 13 once. In addition,
$S_0$ also accumulates $n_1$ and $n_2$, where $n_1$ and $n_2$ respectively record the total accumulated weights of $[[\nabla\omega^T_{m_1}]],[[\nabla\omega^T_{m_2}]]$.
- �ӳٸ��·���͸��� (DUCC)�����㷨 4 ��ʾ�������� $S_0$ �������߿ͻ���$C_i$ ���͵ļ���ģ�͸��� $[[\nabla\omega^T_i]]$��������ƽ��ֵ$[[\nabla\omega^T_m]]$��Ȼ��$S_0$��¼��һ���յ��������ӳٸ��£���ʾΪ $\nabla\omega_d=([[\nabla\omega^{T_0}_0]],...,[[\nabla\omega^{T_d}_d]])$������ʼ��$[[\nabla\omega^T_{m_1}]]��[[\nabla\omega^T_{m_2}]]$�� $n_1��n_2$ Ϊ 0������ÿ���ӳ�ģ�͸���$[[\nabla\omega^{T_j}_j]]\in H$��$S_0$ִ�й�ʽ13 һ�Ρ����⣬$S_0$ �����ۻ� $n_1$�� $n_2$������$n_1$�� $n_2$ �ֱ��¼ $[[\nabla\omega^T_{m_1}]]��[[\nabla\omega^T_{m_2}]]$ �����ۻ�Ȩ�ء�
![�㷨4](img\PAFed\�㷨4.png)
- ���$S_0$ �� $[[\nabla\omega^T_{m_1}]]$���� n1�������$[[\nabla\omega^T_{m_1}]]$��ƽ��ֵ��ͬ����$[[\nabla\omega^T_{m_2}]]$���� n2���ó� $[[\nabla\omega^T_{m_2}]]$��ƽ��ֵ��Ȼ��$S_0$ ���ݹ�ʽ14 ����$[[\nabla\omega^T_{m_2}]]$��

> Model aggregation. As in Eq. 17, $S_0$ updates the global model $[[\omega^T]]$.
- ģ�ͻ��ܡ��繫ʽ 17 ��ʾ��$S_0$����ȫ��ģ��$[[\omega^T]]$��
![��ʽ17](img\PAFed\��ʽ17.png)
	- ���У�$\alpha_0��\alpha_1��\alpha_2$�ֱ�Ϊ���ӳ�ģ�͸��¡��ǳ�ͻ�ӳ�ģ�͸��ºͳ�ͻ�ӳ�ģ�͸��µ�Ȩ��ֵ��

> Key transformation. Since $[[\omega^{T+1}]]$ is encrypted by the public key $pk_s$ and cannot be directly decrypted by the clients, we need to perform key conversion. As shown in Algorithm 5, $S_0$ first randomly generates a non-zero vector $r$ as a mask vector, where the size of $r$ is consistent with the size of $[[\omega^{T+1}]]$. Then, $S_0$ locally encrypts r by calling $CKKS.Enc_{pk_s}(r)$. Then,$S_0$ confuses $[[\omega^{T+1}]]$ as $[[\omega^{T+1}]]+[[r]]$. Finally, $S_0$ sends $[[\omega^{T+1}+r]]$ to $S_1$. After receiving $[[\omega^{T+1}+r]]$, $S_1$ first decrypts it with its own private key $sk_s$ to get $\omega^{T+1}+r$, then encrypts $\omega^{T+1}+r$ with each client��s public key by calling
$CKKS.Enc_{pk_i}(\omega^{T+1}+r)$, finally sends the encrypted result to $S_0$.$S_0$ first encrypts $r$ with the public key of each client to obtain $[[r]]$, then removes the mask by $[[\omega^{T+1}+r]]-[[r]]$,finally sends the corresponding $[[\omega^{T+1}]]$ to the next round of clients.
- ��Կת�������� $[[\omega^{T+1}]]$���ɹ�Կ $pk_s$ ���ܵģ��ͻ����޷�ֱ�ӽ��ܣ����������Ҫ������Կת�������㷨 5 ��ʾ��$S_0$ �����������һ���������� $r$ ��Ϊ�������������� $r$ �Ĵ�С�� $[[\omega^{T+1}]]$ �Ĵ�Сһ�¡�Ȼ��$S_0$ ͨ������ $CKKS.Enc_{pk_s}(r)$�� $r$ ���б��ؼ��ܡ�Ȼ��$S_0$ �� $[[\omega^{T+1}]]$����Ϊ $[[\omega^{T+1}]]+[[r]]$�����$S_0$ �� $[[\omega^{T+1}+r]]$���͸� $S_1$��$S_1$ �յ�$[[\omega^{T+1}+r]]$ ���������Լ���˽Կ $sk_s$ ���ܣ��õ� $\omega^{T+1}+r$��Ȼ����� $CKKS.Enc_{pk_i}(\omega^{T+1}+r)$����ÿ���ͻ��˵Ĺ�Կ��$\omega^{T+1}+r$���м��ܣ���󽫼��ܽ�����͸� $S_0$��$S_0$������ÿ���ͻ��˵Ĺ�����Կ��$r$  ���м��ܣ��õ� $[[r]]$ ��Ȼ��ͨ�� $[[\omega^{T+1}+r]]-[[r]]$ɾ�����룬�������һ�ֿͻ��˷�����Ӧ�� $[[\omega^{T+1}]]$��
![�㷨5](img\PAFed\�㷨5.png)
# �����ܽ�
## �����������
1. ��Ϊ�����ӳ����⣬һЩ�ͻ��˵ı���ѵ������޷���ʱ���͸��ۺϷ����������첽����ѧϰ�����»�����Ӱ��ѵ��Ч�ʡ�ģ�;��ȡ�
2. ��Ϊ����ѧϰ���ڶ���ͻ��ˣ�ÿ���ͻ��˵ı������ݲ�������ȫһ����������ݳ���Non-IID���Ƕ���ͬ�ֲ��������ڱ����У�Non-IID������Ϊ��������ͬ�ֲ����ִ���첽����ѧϰ���ܺܺõ���ӦNon-IID��������Ӱ��ģ�;��ȡ�
## ���Ľ������
1. �����ӳ�ģ�͸��£����Ĳ��ý��淽����ӷ������Ƚ��ӳ�ģ�͸��·�Ϊ�������֣���ͻ�ӳ�ģ�͸���$\nabla\omega^T_{m_2}$�ͷǳ�ͻ�ӳ�ģ�͸���$\nabla\omega^T_{m_1}$�����ڳ�ͻ�ӳ�ģ�͸���$\nabla\omega^T_{m_2}$������ͶӰ��$\nabla\omega^T_m$�����߿ͻ���ģ�͸��µ�ƽ��ֵ���ķ���ƽ���ϣ�������$\nabla\omega^T_{m_2}$�ķ���ͼ3���������ۺϺ��ȫ��ģ�͸���׼ȷ��
2. ����Non-IID���⣬���ǲ���ADMM �㷨���Ż���ʧ������
## ��˽����
- ���ļ���KDC�ǳ�ʵ�ģ��������Ϳͻ����ǳ�ʵ������ġ�������˵������ʵ�嶼��ʵ����ѭ������õ�ѵ��Э�飬����������ͨ���ͻ��˵ı���ģ���ƶϿͻ��˵�˽����Ϣ���ͻ���Ҳ����ͨ��ȫ��ģ�ͺ�����ƶ������ͻ��˵�˽����Ϣ��
- ���Ĳ���ȫ̬ͬ�����������������ݣ�����Կϵͳ�����пͻ��˹�����Կ�ԣ�������ȫ�����Ĳ���CKKSȫ̬ͬ�����㷨��ÿ���ͻ��˺�$S_1$���Լ�����Կ�ԡ��ڷ��ͱ���ѵ�����ǰ��$C_i$��ʹ��$S_1$�Ĺ�Կ���м��ܣ����ۺϹ��̷�����$S_0$������$S_1$����������Ϊ$S_0$�յ����Ǽ������ݣ���$S_0$��������Կ����$S_1$��Ȼӵ����Կ������$S_1$ֻ���յ�����$S_0$�ľۺϽ������ʹ$S_1$�������ݣ�Ҳ�޷������ƶ�$C_i$�����ݡ�

## ˼��
1. ���ܷ��������Բ�������ȫ̬ͬ��ͬ���أ�Ϊ$C_i$������Կ�ԣ��ɷ��������мӽ��ܡ�
2. �������������ܷ�ֻ��һ̨����������֤���ݰ�ȫ���ƺ�������
3. �����ӳ�ģ�͸��������������������
4. ����Non-IID��Ӱ���Ƿ����������������