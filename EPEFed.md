# Efficient and Privacy-Enhanced Federated Learning
Based on Parameter Degradation--���ڲ����˻��ĸ�Ч��˽��ǿ����ѧϰ
## ��Դ
TSC	2024 
����: [https://ieeexplore.ieee.org/abstract/document/10528912](https://ieeexplore.ieee.org/abstract/document/10528912)
## ����
ͨ��ǧ�ʵ�����[Github](https://github.com/wzx442/paper_reading/blob/main/%E5%AF%BC%E8%AF%BB/%E3%80%90%E5%AF%BC%E8%AF%BB%E3%80%91%EF%BC%8824%20TSC%EF%BC%89Efficient_and_Privacy-Enhanced_Federated_Learning_Based_on_Parameter_Degradation.md)
## Abstract
> Federated Learning ensures that clients can collaboratively train a global model by uploading local gradients,keeping data locally, and preserving the security of sensitive data.
> However, studies have shown that attackers can infer local data from gradients, raising the urgent need for gradient protection.
> The differential privacy technique protects local gradients by adding noise. This paper proposes a federated privacy-enhancing algorithm that combines local differential privacy, parameter sparsification, and weighted aggregation for cross-silo setting.
- ����ѧϰͨ���ϴ������ݶȣ����������ڱ��أ��������������ݵİ�ȫ��ȷ���ͻ�����Эͬѵ��ȫ��ģ�͡�
- Ȼ�����о�������***�����߿��Դ��ݶ����ƶϳ���������***�����������Ҫ�����ݶȱ�����
- �����˽����ͨ����������������ֲ��ݶȡ����������һ��������˽��ǿ�㷨�����㷨����˾ֲ������˽������ϡ�軯�͹¾����������µļ�Ȩ�ۺϡ�
```python
cross-cilo��
silo �ı����ǵؽѡ�������cross-silo ���silo��Ȼ���������˼,����ָ����Ϣ�¾���,��˼��ͬ�Ĵ��� island,��Ϣ�µ���
��ν��Ϣ�¾�����ָ: һ����ҵ�и������ŵ���Ϣϵͳ�໥����������Ϊս��û����ϵ������һ�ڿڹ¾���
�����Ľ����Ȼ��Ч�ʽ��ͣ���Ϣ�ظ��ȸߣ��׳�ͻ����
��˳��� cross-silo �ĸ�����Ǵ�ͨ��Щ�¾�������������һ�壬�����������Ϣһ���ԡ����Ч�ʣ�����������׼ȷ������
���� cross-silo ���Է��� �¾����������߹¾���ͨ�ȵȡ�
��Introduction����ϸ�Ա�
```
> Firstly, our method introduces $R\acute{e} nyi$ differential privacy by adding noise before uploading local parameters, achieving local differential privacy. Moreover, we dynamically adjust the privacy budget to control the amount of noise added, balancing privacy and accuracy. 
> Secondly, considering the diversity of clients�� communication abilities, we propose a novel Top-K method with dynamically adjusted parameter upload rates to effectively reduce and properly allocate communication costs.
> Finally, based on the data volume, trustworthiness, and upload rates of participants, we employ a weighted aggregation method, which enhance the robustness of the privacy framework.
- ���ȣ�����������  **"$R\acute{e} nyi$ �����˽"** �������ϴ����ز���ǰ����������Ӷ�ʵ�ֱ��ز����˽�����⣬���ǻ�ͨ����̬������˽Ԥ����������������������Ӷ�����˽��׼ȷ��֮��ȡ��ƽ�⡣
- ��Σ����ǵ��ͻ���ͨ�������Ķ����ԣ����������һ�ֶ�̬���������ϴ��ʵ����� Top-K ����������Ч���Ͳ��������ͨ�ųɱ���
- ��󣬸��ݿͻ��˵�**������**��**���Ŷ�**��**�ϴ���**�����Ĳ�����**��Ȩ�ۺϷ�**���Ӷ���ǿ����˽��ܵ��Ƚ��ԡ�
> Index Terms��Federated learning, differential privacy, communication costs, credibility, aggregation.
- �ؼ��ʣ�����ѧϰ�������˽��ͨ�ųɱ������Ŷȡ��ۺϡ�
## INTRODUCTION
### A. Background
> Different organizations are hesitant to contribute their own data due to privacy concerns. This has resulted in data silos, which hinder effective data integration.
- ���Ŵ����ݵķ�չ���˹����ܴﵽ���µĸ߶ȡ�Ȼ����ʵ�ָ߾���ѧϰģ����Ҫ���ģ�͸�����������֧�֡��ź����ǣ���ͬ����֯���ڶ���˽�Ŀ��ǣ���Ը�����Լ������ݡ������������ݹµ����谭����Ч���������ϡ�
- ����ѧϰ��һ�����˵�**�ֲ�ʽ����ѧϰ����**��ͨ������������˽������Ȩ�������һ���⡣������ѧϰ�У��ͻ����ڱ���ѵ���Լ���ģ�ͣ�����ģ�Ͳ����ϴ�����������������������������Щ����������ȫ��ģ�ͣ��������º��ģ�ͷ��ظ��ͻ��ˡ���һ���̷������У�ֱ��ȫ��ģ�ʹﵽԤ���ľ��Ȼ�����������
> In the realm of FL, there exist two predominant configurations: cross-device and cross-silo. In cross-device FL,participants typically comprise edge devices such as smart gadgets and laptops, which may number in the thousands or even millions. These participants are generally considered unreliable and possess limited computational capabilities. In contrast, in the cross-silo FL paradigm, the stakeholders are organizations; the number of participants is relatively limited,usually ranging between 2 and 100. Given the nature of the participants, the process is generally deemed reliable,and each entity possesses significant computational resources.Cross-silo scenarios are exceedingly common in real-world applications, such as credit card fraud detection, clinical disease prediction, 6G network and so on.

- ������ѧϰ���򣬴���������Ҫ���ã����豸�Ϳ�µ����ڿ��豸����ѧϰ�У��ͻ���ͨ���������ܹ��ߺͱʼǱ����Եȱ�Ե�豸�����������ܴﵽ��ǧ������������Щ�ͻ���ͨ������Ϊ�ǲ��ɿ��ģ����Ҽ����������ޡ�
- ���֮�£��ڿ�µ�����ѧϰ�У��������������֯���ͻ��˵�����������ޣ�ͨ���� 2 �� 100 ֮�䡣���ڿͻ��˵����ʣ��ù���ͨ������Ϊ�ǿɿ��ģ�����**ÿ��ʵ�嶼ӵ�д����ļ�����Դ**������ʵ�����Ӧ���У������ÿ���թ��⡢�ٴ�����Ԥ�⡢6G ����ȣ���µ�������Ϊ������
```c
Cross-device��Cross-Silo������ѧϰ������:
1��ģʽ��ͬ
Cross-device����ѧϰ�����豸�����ģʽ��

Cross-Silo����ѧϰ������豸����ѧϰ�������෴��Cross-Silo ����ѧϰ��������Ƶ�ĳЩ����ǳ���
�����֯���ֻ���빲��ѵ��ģ�ͣ��������������ʱ��cross-silo�����Ƿǳ��õ�ѡ��
Cross-Silo ����ѧϰ��������Ҫ�����¼���Ҫ�㣺���ݷָ�������ơ�������˽���������ӷֽ⡣

2����ԵĿͻ��˲�ͬ
Cross-device����ѧϰ��Cross-device FL��Ե����Ǳ�Яʽ�����豸������ʽ�����豸�ȣ�ͳ��Ϊ�����豸��Internet of Things, IoT devices����

Cross-Silo����ѧϰ��Cross-silo FL��ԵĿͻ�������ҵ���𡢻�����λ����ġ�

3���ͻ���״̬��ͬ **
Cross-device����ѧϰ����״̬��ÿ���ͻ����Խ������һ���������ͨ���ٶ���ÿ�ּ����ж���һ����δ�����Ŀͻ�����������

#Cross-Silo����ѧϰ����״̬��ÿ���ͻ��˶����Բ�������ÿһ�֣�������Я��״̬��

4���ɶ�λ�Բ�ͬ
Cross-device����ѧϰ��û�ж�����ţ��޷�ֱ��Ϊ�ͻ�����������

#Cross-Silo����ѧϰ���ж�����ţ�ÿ���ͻ��˶���һ����ʶ�����ƣ��ñ�ʶ����������ϵͳר�ŷ��ʡ�

5����չƿ����ͬ
Cross-device����ѧϰ�����㴫�俪����ͨ�Ų��ȶ���

Cross-Silo����ѧϰ�������칹��

6��ͨ�������ͬ
Cross-device����ѧϰ�����ȶ������ɿ���

Cross-Silo����ѧϰ���ȶ��ҿɿ���

7�����ݻ������ݲ�ͬ
Cross-device����ѧϰ�����򻮷֡�

Cross-Silo����ѧϰ���ɺ������򻮷֡�
```

- ������µ������У�����ѧϰ�ܹ���������˽��������������������ͻ���֮��Ľ�����Ȼ��ԭʼģ�Ͳ�������Ȼ����ѧϰ����Ҫ���������ݣ������Ӳ������¾����ƶϳ����ؽڵ�������˽��Ϣ����ع����������ݶȵı�¶���Խ�ʾ������ʵ�ڵ�����ʾ������Ҫ�����޹ص������е��������ԣ���������ѵ����������Ϣ����ˣ��б�Ҫ��ǿ����ѧϰģ�͵���˽��������Ҳ�Ǳ����������о�������--��µ�����ѧϰ�µ���˽��ǿ������

> Differential privacy (DP) achieves the privacy goal through data perturbation and introduces minimal additional computational burden, making it widely applicable in various scenarios. This paper also embraces differential privacy as a means to enhance privacy in federated learning. Differential privacy achieves privacy preservation by injecting noise into the parameters. The greater the amount of noise, the stronger the privacy, albeit at the expense of accuracy. Privacy refers to the extent to which details of the client��s local data are protected from disclosure. Accuracy refers to the degree to which the trained final model predicts accurately in the face of new samples. Consequently, balancing privacy with accuracy is a focal point in the design of differential privacy approaches.
- �����˽��DP��ͨ�������Ŷ�ʵ����˽Ŀ�꣬��������ļ��㸺��������ͣ���˹㷺�����ڸ��ֳ��������Ļ��������˽��Ϊ��ǿ����ѧϰ����˽������һ���ֶΡ������˽ͨ���������ע��������ʵ����˽����������Խ����˽�Ծ�Խǿ�����ܻ�����׼ȷ�ԡ���˽��ָ�ͻ��������ݵ�ϸ���ڶ��̶����ܵ�����������й¶��׼ȷ����ָѵ�����ص�����ģ�������������ʱԤ��׼ȷ�ĳ̶ȡ���ˣ�ƽ����˽��׼ȷ������Ʋ�����˽�������ص㡣

> Irrespective of the privacy protection method employed, communication cost remains a pivotal challenge to address.
- ���۲���������˽����������**ͨ�ſ�����Ȼ����Ҫ����Ĺؼ�����**��ͨ����ָ������ѵ�������У���������ͻ���֮�佻���Ĳ�����������ÿ�������У��ͻ�����Ҫ���䱾��ģ�Ͳ������͸�����������������ͻ��ˡ�
- Ȼ����ģ�Ͳ���ͨ�����������޵���������ϴ�����˴��������ݿ��ܻᵼ��ͨ���ӳٻ���ϣ����ս�������ѧϰ������Ч�ʡ�
- ���⣬���ڲ����˽����ӵ���������������������ȣ�����Խ�࣬��ģ��׼ȷ�Ե�Ӱ��Խ��
- ���⣬ÿ���ͻ��˵��豸������������ͨ�����������������ԣ�Ӧ����ʵ��������к��������Ӧ������
- ��ˣ��������Ӧ��ʵ��ģ�;��ȡ���˽��ͨ�ųɱ�֮���ƽ������һ����Ҫ���⡣

> Finally, the robustness of global model aggregation are key issues in privacy-enhanced federated learning.
>  In this paper, the predictable disturbance is a carefully designed noise added to achieve differential privacy.Unforeseeable disturbances are mainly attacks, such as model poisoning attacks launched by malicious attackers.
>  Against the backdrop of data distribution heterogeneity, there are variations in the quality and quantity of local data among participants.The process of aggregating noised parameters from multiple local nodes to obtain a comprehensive model representationis critical to the success of this approach. 
- ���ȫ��ģ�;ۺϵ��Ƚ�������˽��ǿ����ѧϰ�Ĺؼ����⡣³������ָ��Կ�Ԥ���Ͳ���Ԥ���ĸ���ʱ��ģ��ѵ�������������е�������
- ��**�����У���Ԥ���ĸ�����Ϊʵ�ֲ����˽�����������ӵ�����������Ԥ���ĸ�����Ҫ�ǹ���**������⹥���߷����ģ���ж������������ݷֲ��칹�ı����£��ͻ���֮�䱾�����ݵ�����������Ҳ���ڲ��졣�Ӷ���ֲ��ڵ�������������Ի���ۺ�ģ�ͱ�ʾ�Ĺ��̣������ַ����ĳɹ�������Ҫ���ֲ��ڵ�ѵ�����̵���Ч�ԻἫ���Ӱ��ȫ��ģ�;ۺϵ��������м������ػ�Ӱ�챾�ؽڵ�ѵ����Ч������ѵ�����ݵ����������������ϴ������������Լ����⹥���ߵȡ���ˣ�**����ȫ�������ֲ������Ŀ��Ŷȣ���ȷ���ۺ�����**��

> The heterogeneity of client data distribution makes the local data quality of clients uneven, which will directly affect the accuracy and robustness of aggregation.
- �ͻ������ݷֲ���������ʹ�ÿͻ��˵ı������������β�룬�⽫ֱ��Ӱ��ۺϵ�׼ȷ�Ժ�³���ԡ�
- ��˽��ͨ�����������ʵ�ֵģ�������Ҳ��Ӱ��׼ȷ�Ժ�³���ԡ������Ĵ�С����ͨ���������������������ƣ������ͨ�Ų���������̫�٣�����ģ�͵�׼ȷ�Ժ��Ƚ��Բ�������Ӱ�졣��֮������һ����Ч���Ƚ���������˽��ǿ�ܹ����뿼����˽��׼ȷ�ԡ�ͨ�ź��Ƚ��ԡ�

> The relationship between these properties is shown in Fig. 1, where the ��+�� indicates positive correlation and the ��?�� indicates negative correlation.
- ��Щ����֮��Ĺ�ϵ��ͼ 1 ��ʾ������ "+"��ʾ����أ�"-"��ʾ����ء�
![ͼ1](img\EPEFed\ͼ1.png)
### B. Contributions
> Our algorithm adaptively adjusts the privacy budget and parameter upload rate and employs importance-weighted aggregation to achieve robust learning in scenarios involving malicious
participants.
- Ϊ��Ӧ����Щ��ս�������ƽ����˽��׼ȷ�ԡ�ͨ�ųɱ��;ۺ�³���ԣ����������һ��������˽��ǿ�ܹ������ĵ��㷨��������Ӧ�ص�����˽Ԥ��Ͳ����ϴ��ʣ���������Ҫ�Լ�Ȩ�ۺϣ��Ӷ����漰����ͻ��˵������ʵ���Ƚ�ѧϰ��

> We summarize the main contributions of this paper as follows:
> 	- We introduce a simple yet effective dynamic privacy budget adjustment mechanism for $R\acute{e} nyi$ differential privacy. This adjustment, based on changes in global model accuracy within a given time window, directly mitigates the accuracy decline caused by added noise.
> - Addressing the issue of communication cost, we propose an adaptive parameter upload rate adjustment method based on communication latency. This method first assesses the capabilities of participating nodes and then dynamically adjusts the parameter upload rate based on the heterogeneity of node capabilities.
> - We propose an importance-weighted aggregation method.By evaluating the contribution of local node parameters to the global model through multiple factors and considering the credibility of parameters by integrating both localglobal and intra-local node relationships, we effectively enhance the robustness and efficiency of global model aggregation.
- ����������һ�ּ򵥶���Ч��**��̬��˽Ԥ���������**�������� $R\acute{e} nyi$�����˽�����ֵ������ƻ��ڸ���ʱ�䴰����**ȫ��ģ�;���**�ı仯����ֱ�ӻ������������Ӷ����µľ����½���
- ���ͨ�ųɱ����⣬���������һ��**����ͨ���ӳ�**��**����Ӧ�����ϴ��ʵ�������**�����ַ���������������ڵ��������Ȼ����ݽڵ������������Զ�̬���������ϴ��ʡ�
- ���������һ����Ҫ�Լ�Ȩ���ܷ���ͨ���������������ؽڵ������ȫ��ģ�͵Ĺ��ף����ۺ� **����-ȫ��**��**�����ڲ��ڵ�**��ϵ���ǲ����Ŀ��Ŷȣ�������Ч�������ȫ��ģ�;ۺϵ�³���Ժ�Ч�ʡ�

```c
1. Privacy:
�ڱ��ĵ��о��У�Ϊ "Renyi�����˽"��RDP��������һ�ּ򵥶���Ч�Ķ�̬��˽Ԥ��������ơ�
���ݸ�����ʷʱ�䴰����ȫ��ģ��׼ȷ�ȵı仯����̬������һ�ֵ���˽Ԥ�㣬ֱ�ӻ������������Ӷ����µ�׼ȷ���½���   
2. Communication:
������˽��׼ȷ�ԣ�ͨ����Ҳ��Ӱ������ѧϰ��ֱ�����ء�
������ѧϰ�У�ͨ�ſ���ָ���ǽ����еĲ���������ͨ����Խ����Ҫ�����Ĳ���������Խ�ࡣ
�ͻ��˲���Ĳ����˽���ڽ���ǰΪÿ�����������������������Խ�࣬Ҫ�ﵽ��ͬ�̶ȵ���˽����Ҫ�����������
�����ͬʱ��ͨ�������٣����ϴ��Ĳ������٣���ģ�Ͳ����Ĳ������Ʊػ����ģ�͵Ĳ�׼ȷ�ԡ�
��ˣ���Ҫͬʱ������˽��׼ȷ�Ժ�ͨ�ųɱ�֮���Ȩ�⡣      
����ϡ�軯������ѧϰ�й㷺ʹ�õ�һ�ֽ���ͨ�ųɱ��ļ�����
�����������ѵ�������У��ͻ��˿����ϴ�����ֵ����ĳ����ֵ���ݶȡ�Ȼ�������幤���жԲ���ѡ���ʵ�ȷ����û�������о���
ͨ��������ѡ���ʱ���Ϊһ������������֤��ģ�͵��ȶ��ԡ�        
�ڱ��ĵ��о��У������һ�ֻ���ͨ���ӳٵ�����Ӧ�����ϴ����ʵ���������
�÷���������������ڵ��������Ȼ����ݽڵ������������Զ�̬���������ϴ����ʣ�
�Ӷ�ʹ������Ϣѹ�����ʵ����޸��ӽ��գ���Ч����ͨ�ųɱ���      
3. Global Model Aggregation
��ȫ��ģ�;ۺ϶��ԣ����칹���ݷֲ��ı����£��ͻ���֮��ı��������������������ڲ��죬�����һ����������Ҫ��
�ۺ����Զ�����ؽڵ�������������γ�һ���ۺ�ģ�ͣ����ڲ����˽�����ĳɹ�������Ҫ��
�����Ĵ��ڣ��ټ���Ǳ�ڵĶ����ⲿ�����ߣ����谭�ۺ��㷨����ȷ������       
�ڱ��о��У������һ����Ҫ�Լ�Ȩ�ۺϷ�����
�÷������ö��������������ؽڵ������ȫ��ģ�͵Ĺ��ף���ͨ�����ϱ���-ȫ�ֺͱ��ؽڵ��ڲ��Ĺ�ϵ�����ǲ����Ŀ��Ŷȡ�
ͨ���Ծۺϵ�������������Ч�ü�Ȩ����Ч�������ȫ��ģ�;ۺϵ�³���Ժ�Ч�ʡ�       

```

## PRELIMINARY
> In a federated learning system that leverages local differential privacy, users upload perturbed parameter values instead of the original ones, ensuring that the perturbed parameters are safeguarded against privacy inference attacks. The privacy level of differential privacy is determined by the privacy budget $\epsilon$.
> A lower privacy budget provides a higher degree of privacy protection,but it can also lead to lower model accuracy.
- �����ݼ��д洢�ڵ�һ�����������ݴ��������ݷ�����������£�����ʽ�����˽������Ч�����������ݼ�����˽��Ȼ����������ѧϰ�У��������Է�ɢ�ķ�ʽ�洢�ڶ���ͻ����еģ�ͨ�������ñ��ز����˽�������Թ�����������������ȸ衢ƻ����΢���֪����˾�ѽ����ز����˽���ɵ����Ʒ�С�
- �����þֲ������˽������ѧϰϵͳ�У��û��ϴ��Ŷ�����ֵ������ԭʼ����ֵ���Ӷ�ȷ���Ŷ�����������˽�ƶϹ����������˽����˽��������˽Ԥ��$\epsilon$ ������
- ��˽Ԥ��Խ�ͣ���˽�����̶�Խ�ߡ���Ҳ���ܵ���ģ�;��Ƚ��͡�

```c

�����˽��Ҫͨ�����������������ʵ����˽������
��˽Ԥ�㣨�ţ���������������������Ӷ�ƽ������˽�����ݵ������ԡ�
��˽Ԥ�㣬�򵥵�˵���������������õ���ģ�;��ȣ�
��˽Ԥ��Խ�ͣ����õ���ģ�;��Ⱦ�Խ�ͣ�Ҳ�����������Խϴ��������    

```

> **Definition 1.** (($\epsilon, \eth$)-differential privacy, ($\epsilon, \eth$)-DP): A randomization mechanism $\mathrm{M}$ satisfies ($\epsilon, \eth$)-differential privacy ( $\epsilon > 0, \eth > 0$) when and only when for any adjacent input datasets *D* and *D��* and any possible set of output values $R_M$, there is:
![eq1](img\EPEFed\eq1.png)

- ���� 1. (($\epsilon, \eth$)-�����˽��($\epsilon, \eth$)-DP)����������� $\mathrm{M}$ ����($\epsilon, \eth$)-�����˽ ( $\epsilon > 0, \eth > 0$)�����ҽ��������������ڵ��������ݼ� *D* �� *D��*���Լ�������ܵ����ֵ�� $R_M$����Eq(1)��
- �ſ�Ĳ����˽���� ($\epsilon, \eth$)-DP �������Ϊ���û�������С��  1 - $\eth$  �������� $\epsilon$-DP ��

- [ ] >>>**ʲô��($\epsilon, \eth$)-DP��**<<<
- [X] >>�ο� [֪��](https://zhuanlan.zhihu.com/p/264779199)<< 
- [X] �ο���һ���� [csdn](https://blog.csdn.net/m0_43424329/article/details/121650574?ops_request_misc=&request_id=&biz_id=102&utm_term=(%CF%B5,%20%CE%B4)-DP&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-121650574.nonecase&spm=1018.2226.3001.4187)
- [X] M��һ�����������$R_M$��һ�����ֵ���ϣ�$P_r$��һ�����ʺ�����
- [X] �ڲ����˽��Differential Privacy�������У���Pr��ͨ��ָ���Ǹ��ʣ�Probability��������������ʾĳ���¼������ĸ��ʡ��ڲ����˽�Ķ����У�Pr ��������������ͬ���ݼ���ִ�в�ѯ���㷨����ĸ��ʡ������˽�Ļ���˼����ͨ�����ѯ������������������������������˽��ʹ���κι۲��߶��޷��Ӳ�ѯ������ƶϳ���������ľ�����Ϣ�������˽����ʽ���������£�
	- �����������ڵ����ݼ� D �� D�� ������ֻ���һ����¼��һ����������� M ����**?-DP**����������п��ܵ���� S �У�
$P_r[M(D)\in S]\le e^{\epsilon}\cdot P_r[M(D')\in S]$
����� Pr[M(D)��S] ��ʾ���� M �����ݼ� D ������ʱ��������ڼ��� S �ڵĸ��ʡ�ͬ���أ�Pr?[M(D��)��S] ��ʾ���� M �����ݼ� D�� ������ʱ�����������ͬ���� S �ڵĸ��ʡ�
? ��һ����ʵ��������������˽�����ĳ̶ȡ��� ? ������ 0 ʱ����ζ�Ų����˽�ı����̶ȸ��ߣ�������ĸ��ʼ�����ȫ���������ݼ��еĸ�����Ϣ������ ? �ϴ�ʱ����ζ�ű����̶Ƚϵͣ�������ܸ������ܵ����ݼ��и�����Ϣ��Ӱ�졣
	- **���� ($\epsilon, \eth$)-DP** �������Ϊ**Eq(1)**������
$P_r[M(D)\in S]\le e^{\epsilon}\cdot P_r[M(D')\in S]+\eth$��
�Ƹ�������� $\mathrm{M}$ ����($\epsilon, \eth$)-differential privacy�������˽������дΪ **($\epsilon, \eth$)-DP**

> **Sequential Composition:** If $F_1(x)$ satisfies ($\epsilon_1, \eth_1$)-DP, while $F_2(x)$ satisfies ($\epsilon_2, \eth_2$)-DP, then the mechanism $G(x)=(F_1(x),F_2(x))$ satisfies ($\epsilon_1+\epsilon_2, \eth_1+\eth_2$)-DP.
- ˳����ϣ����$F_1(x)$ ���� ($\epsilon_1, \eth_1$)-DP �������� $F_2(x)$ ���� ($\epsilon_2, \eth_2$)-DP ��������ô���� $G(x)=(F_1(x),F_2(x))$ ���� ($\epsilon_1+\epsilon_2, \eth_1+\eth_2$)-DP������

```c

#��֪һ�������������˽��ʧ����ô K ����������ܵ���˽��ʧ�Ƕ����أ�
#�ܾ��ȵط��֣���ÿһ������������ӵ�������noise���໥����ʱ����˽��ʧ�ǿ����ۼӵģ��������������˽��ʧ��ָCm����֪�������пɼ�������

```

> **Parallel Composition:** If the dataset $D$ is divided into $k$ disjointed subdata blocks $x_1\cup x_2 \cup ... \cup x_k=D,F(x_1),...,F(x_k)$  satisfy ($\epsilon_1,\eth$)-DP$,...,$($\epsilon_k,\eth$)-DP respectively, then the mechanism for publishing all results $F(x_1),...,F(x_k)$ is satisfied ($max(\epsilon_1,...,\epsilon_k),\eth$)-DP.
- ������ϣ�������ݼ� D ������Ϊ k ���������������ݿ� $x_1\cup x_2 \cup ... \cup x_k=D,F(x_1),...,F(x_k)$�ֱ�����   ($\epsilon_1,\eth$)-DP$,...,$($\epsilon_k,\eth$)-DP����ô������˽��ʧ $F(x_1),...,F(x_k)$�Ա�����($max(\epsilon_1,...,\epsilon_k),\eth$)-DP��
```c

�����ݼ�����ΪK�ݣ�ÿһ��������Ե������������ô���е����������Աȣ�����max       

```
> **Definition 2.** ($\ell_2$-Sensitivity): For the real-valued function $f$ acting on the dataset *D* and *D��* , the$\ell_2$ sensitivity of $s$ is expressed as
![eq2](img\EPEFed\eq2.png)
- **���� 2**��$\ell_2$ �����ȣ����������������ݼ� *D* �� *D��* ��ʵֵ���� $f$��$s$ �� $\ell_2$ �����ȱ�ʾΪEq(2)��
- ��������ָ�������ݵı仯���������ݿ��ѯ���Ӱ�����ĳ̶ȡ�
- ����$f$����һ���򵥵Ĳ�ѯ������

> **Lemma 1.** DP for Gaussian mechanism: One way to make the mechanism satisfy differential privacy is to add noise to the results. Gaussian mechanisms help mechanisms achieve differential privacy by adding noise that satisfies a Gaussian distribution. But the Gaussian mechanism cannot satisfy ?differential privacy, it can satisfy ($\epsilon,\eth$)-differential privacy.For a random function $F(x)$, the Gaussian mechanism can be used to obtain a random function satisfying (($\epsilon,\eth$)-differential privacy $F^�� (x)$:
![eq3](img\EPEFed\eq3.png)
- **���� 1.** ��˹���Ƶ� DP��ʹ������������˽��һ�ַ������ڽ���������������˹����ͨ����������˹�ֲ�����������������ʵ�ֲ����˽������˹���Ʋ�������$\epsilon$-DP����ֻ������ ($\epsilon,\eth$)-DP�������������$F(x)$�������ø�˹���ƻ������ ($\epsilon,\eth$)-DP ��������� $F^ �� (x)$����Eq(3)��
- ���У�$\sigma^2=\frac{2s^2ln(1.25/\eth)}{\epsilon^2}$��$s$ �� $F$ ������������˽��¶�̶ȵ������ȣ�$\mathcal{N}(\sigma^2)$ ��ʾ��˹����̬���ֲ��ĳ�����������ֵΪ 0������Ϊ $\epsilon^2$��
- ��˹���Ƶ��ŵ�֮һ�ǣ�Ϊʵ����˽��������ӵĸ�˹��������������Դ��������ͬ�����⣬������˹�ֲ�֮����Ȼ�Ǹ�˹�ֲ��������˽���ƶ�ͳ�Ʒ�����Ӱ����ܸ��������;�����

> **Definition 3.**  ($R\acute{e}nyi$ differential privacy, RDP): If for all the Neighboring dataset *D* and *D��* , the random mechanism $F$ satisfies:
![eq4](img\EPEFed\eq4.png)
- **���� 3**��$R\acute{e}nyi$i differential privacy��RDP����������������ڽ����ݼ� *D* �� *D��*���������(�������) $F$ ����Eq(4)����ô������� $F(x)$ ���� ($\alpha,\epsilon$)-RDP��Renyi�����˽(RDP)��˼����Ҫ������Renyi ɢ���������������ݼ��ֲ�֮��Ĺ�ϵ��
- ��ͳ�����˽ʹ�ò��� ? ��������˽��ʧ�������������˽RDP��������һ������ �� �����岻ͬ����˽������
- RDP ��ʵ��Ӧ����ͨ���봫ͳ�� ?-�����˽���ϡ����磬����ͨ���� RDP ת��Ϊ��ͳ�� 
?-�����˽��������˽��ʧ������ת������ͨ�����¹�ʽ��ɣ�
$\epsilon=(\alpha-1)\cdot\Delta_{\alpha}(M,D,D')$
- ����$\Delta_{\alpha}(M,D,D')$������Ϊ��Eq(4)�е�ln���֣�ʵ������Щ��𣩡�

> **Sequential Composition:** If $F_1(x)$ satisfies ($\alpha,\epsilon_1$)-RDP, while $F_2(x)$ satisfies ($\alpha,\epsilon_2$)-RDP, then the Composition mechanism of $F_1(x), F_2(x)$ satisfies ($\alpha,\epsilon_1+\epsilon_2$)-RDP.
- ˳����ϣ���� $F_1(x)$ ���� ($\alpha,\epsilon_1$)-RDP���� $F_2(x)$  ����  ($\alpha,\epsilon_2$)-RDP, ��ô $F_1(x), F_2(x)$ �ĺϳɻ������� ($\alpha,\epsilon_1+\epsilon_2$)-RDP��
- ��$\frac{1}{\alpha -1}\ln_{}{(\frac{F_1(x)+F_2(x)}{F_1(x')+F_2(x')})^\alpha }\le \epsilon_1+\epsilon_2$

> **Lemma 2.** RDP for Gaussian mechanism: Gaussian mechanism is the basic mechanism to achieve Renyi differential  privacy. For a function $f$ : $\mathcal{D} \to \mathbb{R}^k$ with sensitivity $s$, a mechanism $F$ follows ($\alpha,\epsilon$)-RDP can be constructed by
![eq5](img\EPEFed\eq5.png)
- **���� 2.** ��˹���Ƶ� RDP����˹������ʵ��RDP�Ļ������ơ����ں��� $f$ ��$\mathcal{D} \to \mathbb{R}^k$����������Ϊ $s$�����ͨ��Eq(5)��������һ����ѭ ($\alpha,\epsilon$)-RDP�Ļ��� $F$

> **Lemma 3.** From ($\alpha,\epsilon$)-RDP to ($\epsilon,\eth$)-DP: If $F(x)$ satisfies ($\alpha,\epsilon$)-RDP, then for any given $\eth>0$, $F$ satisfies ($\epsilon',\eth$)differential privacy, where $\epsilon'=\epsilon+\ln{}{\frac{1/\eth}{\alpha-1}}$. The value of $\eth$ is generally taken as $\eth\le\frac{1}{n^2}$.
- **���� 3.** �ӣ�($\alpha,\epsilon$)-RDP ���� ($\epsilon,\eth$)-DP�����$F(x)$ ���� ($\alpha,\epsilon$)-RDP����ô������������� $\eth>0$, $F$ ���� ($\epsilon',\eth$)-DP������ $\epsilon'=\epsilon+\ln{}{\frac{1/\eth}{\alpha-1}}$ ��$\eth$��ֵһ��ȡΪ$\eth\le\frac{1}{n^2}$��
- RDP ��Ϊ�������� $\alpha$ �������� $\Delta f$ �����һ��
- RDP �ɸ��ݲ�ͬ��Ӧ�ó���ѡ����ʵĲ��� $\alpha$���� $\alpha=1$ ʱ��RDP ���� DP���� $\alpha>1$ ʱ�������ṩ��ǿ����˽�������� $\sigma^2$ ��ֵ����ʱ��ʹ��RDP��˳������������ظ�Ӧ�ø�˹���Ƶ���˽���ģ�Ȼ�� RDP ת��Ϊ  ($\epsilon,\eth$)-DP��ͨ�����ַ����õ�������˽����ͨ����ֱ��Ӧ��  ($\epsilon,\eth$)-DP  ��˳����ϵõ���Ҫ�͵öࡣ������һ���ԣ����Ļ�ʹ��RDP��ʵ�־ֲ����ݼ��еĲ����Ŷ���
```c

�е���

```

## METHOD DESIGN
> This section focuses on a new federated privacy enhancement architecture to achieve client-level privacy protection,low communication overhead, and high robust aggregation. 

### A. System model
> Our aim is to devise a privacy-centric and robust federated learning framework tailored for cross-silo settings.
> Our specific objectives are outlined as follows:
> 	- Communication: We have designed an adaptive parameter upload rate adjustment method based on Top-K, which tightens the lower bound of local information compression rate, effectively reducing communication costs.
> - Robustness: We propose an importance-weighted aggregation method, significantly enhancing the robustness of global model aggregation under differential privacy with noised parameters.
> - Privacy: We strive to achieve client-level differential privacy by employing a dynamic privacy budget adjustment mechanism. This approach facilitates a judicious balance between privacy and accuracy.

- ͨ�ţ����������һ�ֻ��� Top-K ������Ӧ�����ϴ��ʵ����������ս��˱�����Ϣѹ�����ʵ����ޣ���Ч������ͨ�ſ�����
- ³���ԣ����������һ����Ҫ�ȼ�Ȩ�ۺϷ�����������ǿ���ں������������Ĳ����˽������ȫ��ģ�;ۺϵ�³���ԡ�
- ��˽�����Ĳ��ö�̬��˽Ԥ��������ƣ�Ŭ��ʵ�ֿͻ������컯��˽���������ַ�������������˽��׼ȷ��֮��ȡ�����ǵ�ƽ�⡣

> The proposed architecture executes an iterative process, comprising the following steps��
> 1. The server broadcasts the initialized model, the current round��s upload rate, and the privacy budget to all local clients. Each client then performs local stochastic gradient descent using their local data to obtain updated local weight differences.
> 2. To reduce communication costs, Top-K parameter sparsification is performed based on the given upload rate.
> 3. To protect client privacy, Gaussian noise is introduced to perturb the sparsified model parameters, based on the given privacy budget.
> 4. The noised weight difference parameters are uploaded to the server.
> 5. The server performs weighted aggregation of the uploaded model parameters, considering factors such as each client��s data volume, upload rate, and parameter reliability, to obtain a new global model. Additionally, the server dynamically adjusts the privacy budget for the next round based on the performance of the global model and assesses the communication capabilities of local devices by analyzing the time delay in node parameter uploads, thereby dynamically adjusting the parameter upload rate for each client in the next round.

- ����ļܹ�ִ��һ���������̣��������²��裺
 	- �����������б��ؿͻ��˹㲥��ʼ��ģ�͡������ϴ��ʺ���˽Ԥ�㡣Ȼ��ÿ���ͻ��������䱾������ִ�б�������ݶ��½����Ի�ø��º�ı���Ȩ�ء�
 	- Ϊ�˽���ͨ�ųɱ���Top-K ����ϡ�軯�Ǹ��ݸ������ϴ����ʽ��еġ�
 	- Ϊ�����ͻ���˽�����ݸ�������˽Ԥ�㣬�����˹�������Ŷ�ϡ�軯ģ�Ͳ�����
 	- ������Ȩ�ز�����ϴ�����������
 	- ���������ϴ���ģ�Ͳ������м�Ȩ�ۺϣ�ͬʱ����ÿ���ͻ��˵����������ϴ��ʺͲ����ɿ��Ե����أ��Ӷ��õ�һ���µ�ȫ��ģ�͡����⣬�������������ȫ��ģ�͵����ܶ�̬������һ�ֵ���˽Ԥ�㣬��ͨ�������ڵ�����ϴ���ʱ���ӳ������������豸��ͨ���������Ӷ�����һ�ֶ�̬����ÿ���ͻ��˵Ĳ����ϴ��ʡ�

- ����������ͼ 2 ��ʾ���㷨 1 �����˱���ѵ��������α���롣
![ͼ2](img\EPEFed\ͼ2.png)
![�㷨1](img\EPEFed\�㷨1.png)
### B. Dynamic privacy budget adjustment
- ��̬��˽Ԥ��������ҪĿ������ģ��׼ȷ�Ժ�������˽֮��ʵ��΢���ƽ�⡣����ʹ�ò����˽��������˽���ڲ����ϴ���������֮ǰΪ����Ӿ�����Ƶ�������
- ��˽Ԥ��Խ�ͣ�������Խ����˽��Խ�ߣ�������ʧ���ҲԽ�ߣ��Ӷ�������׼ȷ�ȡ�
- ��ͳ�����˽�㷨����˽Ԥ����ÿһ�ֶ��ǹ̶��ġ�Ϊ����׼ȷ�Ժ���˽֮��ȡ��ƽ�⣬��Ҫ����ģ�͵�ʵ�����ܶ�̬����ÿһ�ֵ���˽Ԥ�㡣��ˣ�����������ݵ�ǰһ��ȫ��ģ�͵�׼ȷ�Ա�����������һ�ֵ���˽Ԥ�㡣
- ����ԭ���ǣ������ǰһ�ֵ�ģ��׼ȷ�Ե���Ԥ��Ч������������һ�ֵ���˽Ԥ�㡣��֮�������ǰһ�ֵ�ģ��׼ȷ�ȳ���Ԥ��Ч������һ�ֵ���˽Ԥ��ͻ���١���Ҫ���ǣ�**Ԥ��ģ��׼ȷ�ȱ�����Ϊһ����ֵ��Χ�������ǵ�һ��ֵ**�����Ϊ���������ṩ������ԣ�����Ӧ���������
- ����ԭ�����£�

> Step 1: The change in accuracy value of the global model in time window of round $t$ and round $t$ ? 1 is $\Delta acc_{t-1}=acc_t-acc_{t-1}$. If $\Delta acc_{t-1}<0$, it means that the accuracy of the model at the end of the $t$-th training round has decreased instead of increased. In this case, the next round should add less noise, and $\epsilon_{t+1}$ should be larger. Assuming that the amount of noise is desired to be reduced by at least $c$, we can use the following formula (6):
![eq6](img\EPEFed\eq6.png)
 ���� 1��ȫ��ģ���ڵ� $t$ �ֺ͵� $t$ - 1 ��ʱ�䴰���еľ���ֵ�仯Ϊ$\Delta acc_{t-1}=acc_t-acc_{t-1}$����� $\Delta acc_{t-1}<0$�����ʾ�� $t$ ��ѵ������ʱģ�͵�׼ȷ�Ȳ�������������������£���һ�����ӵ�����Ӧ�ø��٣�$\epsilon_{t+1}$ ҲӦ�ø��󡣼���ϣ�����������ټ��� $c$.
```c

�ӵ�t-1�ֵ���t�֣�ģ��׼ȷ�ȱ仯ֵ�Ǹ��ģ�˵����t�ֵ�׼ȷ���½��ˣ�   
˵����t-1�ֵ���˽Ԥ��̫С�ˣ������˹������������Ӧ���ڵ�t+1��������˽Ԥ�㣨����������   

```

- ������**���﹫ʽ6Ӧ�ø����˻��߸�����**��Ӧ����$\epsilon_{t+1}\ge \epsilon_t +c$������

> Step 2: If $\Delta acc_{t-2}-\Delta acc_{t-1}\ge d,(\Delta acc_{t-2}>0,\Delta acc_{t-1}>0)$, then it means that the model accuracy at the end of the $t$-th training round has increased but the effect is very little. Then the next round should add less noise, and $\epsilon_{t+1}$ should be larger. In this way, the same result as (6).
- ���� 2����� $\Delta acc_{t-2}-\Delta acc_{t-1}\ge d,(\Delta acc_{t-2}>0,\Delta acc_{t-1}>0)$����˵���� t ��ѵ������ʱ��ģ�;���������ߣ���Ӱ���С����ô��һ�����ӵ�����Ӧ�ø��٣�$\epsilon_{t+1}$ҲӦ�ø�������������루6����ͬ��
- ������**��ô����ԭ�ĵĹ�ʽ����д����**��$\epsilon_{t+1}\ge \epsilon_t +c$������ȷ�Ĺ�ʽ6��**�������д�ӭָ��**������
```c
�����ı���������0��˵��ģ�;���������ߵģ����ǵ�t-2��ʱ�䴰�ڵĸı������ڵ�t-1��ʱ�䴰�ڵĸı�����
˵��������ʱ�䴰���£���˽Ԥ���Ӱ���С������˵��˽Ԥ����һ���С�ˣ�
��ô�����ڵ�t��ʱ�䴰�ڣ���t+1��ѵ����Ҫ������˽Ԥ�㡣
```
> Step 3: If $\Delta acc_{t-1}-\Delta acc_{t-2}\ge 1,(\Delta acc_{t-1}>0)$, then it means that the accuracy of the model is improved more at the $t$-th training round, and stronger protection of privacy can also be implemented while ensuring the training is carried out properly. Then the noise should be increased in the next round, and $\epsilon_{t+1}$  should be smaller.Similarly, assuming that the amount of noise is desired to increase by at least c, we have:
![eq7](img\EPEFed\eq7.png)
- ���� 3����� $\Delta acc_{t-1}-\Delta acc_{t-2}\ge l,(\Delta acc_{t-1}>0)$����˵���ڵ� t ��ѵ���У�ģ�͵�׼ȷ�Եõ��˸������ߣ���ȷ��ѵ���������е�ͬʱ��Ҳ���Զ���˽���и������ı�������ô����һ��ѵ���У�����Ӧ������$\epsilon_{t+1}$ Ӧ�ü�С��ͬ��������ϣ���������������� c
- ������**���ټ���c������Ӧ��$\epsilon_{t+1}\le\epsilon_{t}-c$��**������
> Step 4: In the remaining cases, $\epsilon_{t+1}$ is the average of the remaining privacy budget, i.e. $\epsilon_{t+1}=\frac{\epsilon- {\textstyle \sum_{i}^{t}}\epsilon_i}{T-t}$.
- ���� 4������������£�$\epsilon_{t+1}$��ʣ����˽Ԥ���ƽ��ֵ����$\epsilon_{t+1}=\frac{\epsilon- {\textstyle \sum_{i}^{t}}\epsilon_i}{T-t}$��
- �����У���ֵ $c$ ���ڿ�������Ӧ��˽Ԥ������ķ��ȡ�$c$ ֵԽ����ζ����˽Ԥ��ı仯Խ����ᵼ������ѵ�����̼����ȶ�����ˣ�����ѡ��ƽ����˽Ԥ��� $1/\upsilon$������$\upsilon$Ϊ����������ֵ $d$ �� $l$ ������ʱ������˽Ԥ�㡣�� $d$ �� $l$ ��ֵԽ��˵��ѵ�������жԲ��ȶ��Ե����̶�Խ�ߣ�ѵ�������е�����Ӧ��������Խ�١���������� $c��d��l$ �ľ���ȡֵ��Ҫ����ʵ��������е�����������ʵ������ҵ�һ����Ժ��ʵ�ֵ��$T$ ��ѵ���������������� ($\alpha,\epsilon$)-RDP������($\epsilon+\frac{\ln{1/\eth}{}}{\alpha-1}{},\eth$)-DP��

### C. Adaptive upload rate adjustment Top-K
> TOPK is a widely used parameter selection mechanism to reduce communication cost. K is a predefined threshold which balances the training accuracy and communication cost.
> In cross-silo federated learning, there exist variations in device computing power and communication bandwidth among participants. When a participant��s device has limited computing power and communication bandwidth, it may experience prolonged local computation and communication times, or even face challenges in uploading parameters. In such cases, uploading a reduced number of parameters can effectively reduce communication costs.

- TOP-K ��һ�ֹ㷺ʹ�õĲ���ѡ����ƣ����ڽ���ͨ�ųɱ���K ��һ��Ԥ�������ֵ������ƽ��ѵ�����Ⱥ�ͨ�ųɱ���
- �ڿ�µ�����ѧϰ�У��ͻ���֮����豸����������ͨ�Ŵ�����ڲ��졣��ĳ���ͻ��˵��豸����������ͨ�Ŵ�������ʱ�������ܻᾭ���ϳ��ı��ؼ����ͨ��ʱ�䣬�������ϴ�����ʱ������ս������������£��ϴ����������Ĳ�������Ч����ͨ�ųɱ����෴���ϴ�����������԰���ȫ��ģ�͸��õ���Ӧ�������ݣ��������ѧϰ��׼ȷ�ԡ�
- Ȼ�����������Ĺ����У���ֵ K ͨ���ǹ̶��ģ���һ����û�г�ַ���ͨ�������õĿͻ��˵����ã���һ����Ҳ�����˱�����Ϣѹ���ʵ����ޣ��Ӷ���һ���̶���������ͨ�ųɱ���
- ������˵���� K ֵ��Сʱ��ͨ�������õĿͻ��˿����ڲ�Ӱ��ѵ�����ȵ�������ϴ����������ʹ����ģ�͸��Ӿ�ȷ�������� K ֵ�̶���ֻ���˷Ѳ��ִ����� K ֵ����ʱ��ͨ��������Ŀͻ��˾���Ҫ������ͨ��ʱ�������ɽ������Ӷ�����ѵ�����ȡ���ˣ�������ݲ����豸��ʵ�������̬���������ϴ����������Ż�ͨ�ųɱ���ͬʱȷ�� RDP ����ѧϰ��Ч�ʺ�׼ȷ�ԡ�
- ���Ը��ݱ��ؽڵ��ʱ���ӳ��������豸���������ݴ�ȷ����һ�ֵĲ����ϴ����ʡ�

> Local nodes with low time latency are generally considered to have better communication and computation capabilities, while those with high time latency are considered to have lower capabilities.The specific process is shown below:
> - Step 1: Set the initial parameter upload rate $p_0$.
> - Step 2: The server records the parameter upload time for each participant $k$ in the past $r$ rounds separately,${d^{t-r}_k,...,d^t_k}$. Then the average upload duration is $\overline{d_k}=\frac{ {\textstyle \sum_{j}^{r}}d^j_k}{r}$ .
> - Step 3:
![eq8](img\EPEFed\eq8.png)

> - Step 4: The $p^{t+1}_k \times n$ largest parameters need to be selected for upload, where $n$ is the total number of model parameters.
- һ����Ϊ��ʱ���ӳٵ͵ı��ؽڵ���нϺõ�ͨ�źͼ�����������ʱ���ӳٸߵı��ؽڵ��������ϵ͡������������£�
	- ���� 1�����ó�ʼ�����ϴ����� $p_0$��
	- ���� 2���������ֱ��¼ÿ���ͻ��� $k$ �ڹ�ȥ $r$ ���еĲ����ϴ�ʱ��${d^{t-r}_k,...,d^t_k}$����ôƽ���ϴ�ʱ��Ϊ $\overline{d_k}=\frac{ {\textstyle \sum_{j}^{r}}d^j_k}{r}$ ��
	- ����3���繫ʽ8�����У�$\varrho$ ����Ҫ�����Ŀͻ������İٷֱȡ�����ȷ�� $0<p^{t+1}_k <1$�������� $p^{t+1}_k=p^t_k$��
		 ```c 
		
		 ���ƽ���ϴ�ʱ�����ֵ����һ���ϴ�����䣬˵���ϴ�ʱ��̫�����������ϴ���̫�ߣ�Ӧ�ý��Ͳ����ϴ��ʡ���֮Ӧ�����Ӳ����ϴ��ʡ�
		
		  ```
	- ���� 4����Ҫѡ�� $p^{t+1}_k \times n$ ���������ϴ������� $n$ ��ģ�Ͳ�����������

### D. Weighted Aggregation based on Importance
��ͳ�ļ�Ȩ�ۺϷ������ݱ��ؽڵ��������ȷ��Ȩ�ء��ڱ��Ŀ�������������������Ӧ���ǲ������ϴ������Ŀɿ��ԡ�������˽�������ⲿ���⹥���ߵļ���ή�;ۺϽ����׼ȷ�ԡ����������ھۺ�ǰ�������յ��Ĳ����Ŀɿ��ԣ���Ϊ�ɿ��Խϵ͵Ĳ�������ϵ͵�Ȩ�أ��Լ��������Ͷ��������ģ��׼ȷ�Ե�Ӱ�졣
> Parameter credibility is assessed based on the similarity between two consecutive rounds of parameters from a node and the similarity with the global parameters. Due to the large number of parameters and fine-tuning based on previous training, parameters from adjacent rounds usually exhibit similar orientations and magnitudes. Therefore, the local parameter similarity between two consecutive parameter uploads by a node can serve as a measure of upload confidence. Moreover, nodes significantly contributing to global parameters generally have similar directions and magnitudes compared to the global parameters. Consequently, the similarity between local parameters uploaded by a node in the current round and the global parameters from the previous round can be used to assess the trustworthiness of parameter uploads.
> Global model weighted aggregation is determined by considering three key factors: the amount of node data, the parameter upload rate, and parameter credibility.
- �������Ŷȵ���������**һ���ڵ��������ֲ���֮���������**�Լ�**��ȫ�ֲ�����������**�����ڲ��������϶࣬�Ҹ�����ǰ��ѵ��������΢���������ִεĲ���ͨ�����ֳ����Ƶķ���ͷ��ȡ���ˣ�**�ڵ����������ϴ�����֮��ľֲ��������ƶȿ�����Ϊ�ϴ����Ŷȵĺ�����׼**�����⣬��ȫ�ֲ�����ȣ�**��ȫ�ֲ�������Ҫ���׵Ľڵ�ͨ���������Ƶķ���ͷ���**����ˣ��ڵ��ڱ����ϴ��ľֲ���������һ���ϴ���ȫ�ֲ���֮������ƶȿ��������������ϴ��Ŀ��Ŷȡ�
- ���⣬���ǵ������ϴ��ʵĶ�̬������ͨ��**�����ϴ��ʸߵĿͻ�����ζ��Ӳ����ʩ��**�����Ӧ�����䱾��ģ�Ͷ�ȫ��ģ�͵Ĺ��ס�
- ȫ��ģ�ͼ�Ȩ�ۺ��������ؼ����ؾ������ڵ��������������ϴ��ʺͲ������Ŷȡ�

> Step 1: Calculate the parameter credibility $Cied_k$ of node $k$. According to (10) we can calculate $\cos (\Delta w^{t-1}_k,\Delta w^t_k)$  and $\cos (\Delta w^t_k,\Delta w^{t-1})$  respectively, then we have
        ![eq9](img\EPEFed\eq9.png)

> The similarity of vectors $A, B$ is calculated by the cosine similarity. That is
![eq10](img\EPEFed\eq10.png)

> As we aim to measure the similarity in direction between the two parameter vectors, we
consider the case of low similarity as the opposite direction.
![eq11](img\EPEFed\eq11.png)

Step 2: Calculate the importance score $Imp^t_k$ based on the amount of data, the parameter upload rate and parameter credibility of node $k$:
![eq12](img\EPEFed\eq12.png)

Step 3: Global parameter weighted aggregation:
![eq13](img\EPEFed\eq13.png)

- ���� 1������ڵ� $k$ �Ĳ������Ŷ� $Cied_k$������ (10)�����ǿ��Էֱ���� $\cos (\Delta w^{t-1}_k,\Delta w^t_k)$�� $\cos (\Delta w^t_k,\Delta w^{t-1})$, ���ڹ�ʽ9������ 0 < $\beta$ < 1��
	- 	�������ϴ�Ľڵ�ͨ����ȫ��ģ�͵�Ӱ�������ˣ�������ÿһ���ϴ��ľֲ�Ȩ�ؿ�������һ�ֵ�ȫ��Ȩ�ظ����ơ���ˣ�����Ϊ��Щ�ڵ����ýϸߵ�Ȩ��ֵ$(1-\beta)$��
	- ���� $A��B$ �����ƶ�ͨ���������ƶȼ���ó������У�$A$ �� $B$ ��ʾ����������$\cdot$ ��ʾ�����ĵ�����㣬$||A||$ �� $||B||$ ��ʾ $A$ �� $B$ �� L2 ������
	- �������ƶȵķ�Χ��-1��1������**�ӽ�1��ֵ��ʾ���������ڷ����ϵ������Խϸ�**���෴��**ֵԽ�ӽ�-1��ʾ�����ϵĲ���Խ��**��**ֵԽ�ӽ�0��ʾ��������**����**�ϵĲ���Խ��**���������ǵ�Ŀ���ǲ���������������֮�䷽���ϵ������ԣ�������ǽ��������Ե��������Ϊ�෴�����繫ʽ11.
- ����2: ���ݽڵ�i���������������ϴ��ʺͲ������Ŷȼ�����Ҫ�Է���$Imp^t_k$��
	- ���У�$n_k$��ÿ���ͻ�������������0 < $\gamma_1,\gamma_2,\gamma_3$ < 1�� $\gamma_1+\gamma_2+\gamma_3=1$�������������ֱ�����˾ۺ�ʱ������������ռ�ȡ������ϴ��ʺͲ������Ŷȡ�
	- ��������������ģ�͵�ѵ��������������Ҫ�����ã�����һ���̶��Ͼ����˾ֲ�ģ��ѵ�����ȵ����ޡ���ˣ������������Ե�����£��ֲ����ݵ�����ֲ�ģ�Ͷ�ȫ��ģ�͵Ĺ���������أ���$\gamma_1$ͨ�����������������ġ� 
- ����3:ȫ�ֲ�����Ȩ�ۺϣ��繫ʽ13.

### E. Privacy Analysis
> Assuming a total privacy budget of $\epsilon$, with $T$ total training rounds, and a privacy budget per round of $\epsilon_t$. For a given privacy budget, the Renyi Differential Privacy (RDP) can select an appropriate parameter $\alpha$ such that the conversion of RDP to Differential Privacy (DP) minimizes the privacy budget. Therefore, in each round, different clients k satisfy ($\alpha^k_t, \epsilon_t$)-RDP. According to Lemma 3, it can be converted to DP as ($\epsilon'_t,\eth$)-DP. Given the parallel composition property of DP, each round satisfies ($max(\epsilon'_t),\eth$)-DP. Following the sequential composition property of DP, after $T$ rounds, it satisfies (${\textstyle \sum_{t=1}^{T}}max(\epsilon'_t),T\eth$)-DP, where $\epsilon'_t=\epsilon_t+\frac{ln(1/\eth)}{\alpha^k_t-1}$. 
- ��������˽Ԥ��Ϊ$\epsilon$����ѵ������Ϊ$T$��ÿ����˽Ԥ��Ϊ$\epsilon_t$�����ڸ�������˽Ԥ�㣬���������˽(RDP)����ѡ����ʵĲ���$\alpha$��ʹRDP�������˽(DP)��ת����С����˽Ԥ�㡣��ˣ���ÿһ���У���ͬ�Ŀͻ���$k$����($\alpha^k_t, \epsilon_t$)-RDP����������3��������ת����DPΪ($\epsilon'_t,\eth$)-DP������DP��ƽ��������ԣ�ÿһ�ֶ�����($max(\epsilon'_t),\eth$)-DP������DP��˳��������ʣ�����$T$�ֺ�����(${\textstyle \sum_{t=1}^{T}}max(\epsilon'_t),T\eth$)-DP�����У�$\epsilon'_t=\epsilon_t+\frac{ln(1/\eth)}{\alpha^k_t-1}$��

> Since $\alpha^k_t\in[2,100]$, when $\alpha^k_t=2,\epsilon'_t$ attains its maximum value, which is $\epsilon_t+ln(1/\eth)$, leading to
![eq14](img\EPEFed\eq14.png)

- ����$\alpha^k_t\in[2,100]$����$\alpha^k_t=2$ʱ��$,\epsilon'_t$�ﵽ���ֵ$\epsilon_t+ln(1/\eth)$��������ʽ14.

> When $\alpha^k_t=100,\epsilon'_t$ attains its minimum value, which is $\epsilon_t+\frac{ln(1/\eth)}{99}$ , leading to
> ![eq15](img\EPEFed\eq15.png)

- ��$\alpha^k_t=100$ʱ��$\epsilon'_t$�ﵽ��Сֵ$\epsilon_t+\frac{ln(1/\eth)}{99}$���õ���ʽ15.

> Let $Tln(1/\eth)=\mu$, then there is
![eq16](img\EPEFed\eq16.png)

## ����ʵ�� 

������Ҫ�������������ѧϰ�е�ͨ�ųɱ�����˽�������������ʵ�飬����ʵ���������˷����ͱȽϡ�

��һ��ʵ���ǹ���ͨ�ųɱ���ʵ�飬��Ҫͨ�������ϴ���������ͨ�ųɱ������ģ�;��ȡ���MNIST��CIFAR-10���ݼ��ϣ�ʹ��������Ӧ�ϴ��ʵ���������Atop-K���뾭��FedAvg�㷨���бȽϡ�ʵ������ʾ���ڲ�ͬ�ϴ����£�Atop-K����������Ч�ؼ���ͨ�ųɱ���ͬʱ���ֽϸߵ�ģ�;��ȡ�����CIFAR-10���ݼ��ϣ����ϴ���Ϊ0.1ʱ����Ȼͨ�ųɱ��ϸߣ���ģ�;���Ҳ����������

�ڶ���ʵ���ǹ���ģ��³���Ե�ʵ�飬��Ҫ��֤��ImpWA�����ڷ���ģ���ж����������Ч����ʵ���У�ģ����ģ���ж����������������ImpWA���������������㷨���бȽϡ�ʵ������ʾ��ImpWA��������������㷨���и��ߵ�ģ�;��Ⱥ͸��õ�³���ԡ�

������ʵ���ǹ�����˽������ʵ�飬��Ҫ��֤��RDP-ImpWA�����ڱ�֤ģ�;��ȵ�ͬʱʵ����Ч��˽������Ч����ʵ���У�ʹ���˲�ͬ����˽Ԥ��ֵ��������С������RDP-ImpWA���������������㷨���бȽϡ�ʵ������ʾ��RDP-ImpWA��������������㷨�����ڱ�֤ģ�;��ȵ�����¸��õر����û���˽��

�����������������������Ӧ�ϴ��ʵ���������ImpWA�����Լ�RDP-ImpWA������������Ч�ؽ������ѧϰ�е�ͨ�ųɱ�����˽�������⣬ֵ�ý�һ���о���Ӧ�á�
## �ܽ�
��ƪ���׶������о��е��ѣ������е�֡�
- ������Ҫ�������������أ��ֱ���**��˽��ȫ���ۺ�³���ԡ�ͨ�ſ����Լ�ģ��׼ȷ��**
- ������˽�ԣ�������Ҫ����DP��RDP�Լ����ǵ��ۺ�Ӧ�ã�������˹��������RDP��
- ����ͨ�ſ�����������Ҫ������Top-K�㷨�� ���ͻ���ѡ���Լ���Top-K���������ϴ���
- ����ģ��׼ȷ�ԣ�������Ҫ�����˶�̬��˽Ԥ������㷨�����ϵ���ÿһ�ֵ�ȫ����˽Ԥ�㣬����������ͨ�ſ���������������ȫ��ģ�͵�׼ȷ�ԡ�
- �ھۺ�ȫ��ģ��ʱ�����Ŀ������������أ��ֱ��Ǹ��ͻ��������������ͻ��˵Ĳ����ϴ��ʺͲ������Ŷȡ�
## �����ŵ�
���������һ������ѧϰ�е���˽��ǿ�㷨�����㷨����˱��ز����˽������ϡ�軯�ͼ�Ȩ�ۺϵȼ�����ּ����߿��ƣ���µ��������µ�����ѧϰģ��ѵ�������е���˽����������������˽��׼ȷ�ȡ�ͨ�ſ�����³����֮��ȡ��ƽ�⡣���⣬���㷨�������˵��������п�������Ĳ��ȶ������⣬������˽�һ���Ľ��ķ�����ʵ�ָ��ȶ���³����ѵ��������