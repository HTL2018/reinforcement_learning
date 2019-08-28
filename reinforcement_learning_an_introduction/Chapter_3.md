# Chapter_3: Finite Markov Decision Process有限马尔可夫决策过程 
与前一章multi-armed bandit问题相比:  
> 相同点是都要评估系统反馈.  
> 不同点:MDP还需要在不同场景下选择不同的行动。  
  (MDP也是强化学习问题一种数学上理想化的形式) 
  
## 3.1 The Agent-Environment Interface　　
### 3.1.1 agent 和 environment相互作用的过程:  
> ![0](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/0.jpg)   
> ![1](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/1.png)   
   
任何对受目标引导的行为的学习问题，都可以简化为三个信号在 agent 和 environment 间前后传递的模型：  
> actions: agent 的决策行动  
>  states: 进行行动选择的基准  
> rewards: agent 的目标  
  
简单实例:  
![3](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/3.png)   
![4](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/4.png)   
> 注:图中左侧表格中第三行第二列的`high`应改为`rescued`  
### 3.1.2 Markov property  和 Markov Decision Processes  
**Markov property**:  
![2](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/2.png)   
这一假设要求state包含可能对将来造成影响的所有之前的agent与environment的互动信息，这一性质也被称为Markov property.  
**Markov Decision Processes**:  
满足Markov property的强化学习任务被称为马尔科夫决策过程(Markov decision process)，简称MDP。  
如果state和action的空间是有限的，那么该任务被称为有限马尔科夫决策过程(finite Markov decision process)，简称finite MDP。  
**finite MDP的动态过程**:(这本书提到的大多数的理论不言明都是假定是finite MDP)  
![12](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/12.png)   
给定明确的动态过程，任意其他想知道的关于environment的信息可以被计算得到，比如:  
![13](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/13.png)   
## 3.2 Goals and Rewards  
**agent的目标**可以理解为最大化reward的累积和的期望值。  
每个时间步的reward是一个数值。  
## 3.3 Returns  
假设t时刻后的reward的序列表示为:  
![5](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/5.svg)   
则**return**收益即累计的reward可如下定义:   
![6](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/6.svg)   
其中T是最后terminal state的时间点，这适用于某些有结局的场景，例如游戏结束或者逃出迷宫，这一类问题被称作**episodic task**, 每一个episode到达最终态后，我们可以将系统重置到初始状态再重头开始下一轮试验.  
当然，有些问题可能没有终止状态，比如自动驾驶我们希望车可以一直安全的行驶而不出现终止状态，我们称这一类问题为**continuing task**。通常对于这类问题我们会定义一个折扣率，而return也变为**discounted return**：   
![7](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/7.svg)   
其中:![8](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/8.svg) ,叫做discount rate,越接近1，将来的reward所占比重越来越大。  
discounted return的递归形式：  
![9](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/9.svg)   
## 3.4 Unified Notation for Episodic and Continuing Tasks
如果episode最终到达一个特殊的吸收态(absorbing state)，这个状态只会转移到自己并且产生的reward是0，那么两种任务就达到了统一。比如说，考虑如下的状态转移示意图：  
![11](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/11.jpg)   
可以将episodic task和continuing task的return合并为同一形式：  
![10](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/10.svg)   
## 3.5 Policies and Value Functions  
**value functions**是定义在states(或者state-action pairs)上的函数，用来评估agent若处在某个state的好坏(或是评估agent若处在某个state采取某个action的好坏)。  
> 这里的好坏被定义为期望接收到的future rewards，或者准确说是expected return。当然agent期望得到的future rewards还取决于他所采取的action。因此，value functions由特定的policies决定。  
  
policy是将每个state以及每个action映射到某个概率值的函数.  
**value functions定义**:  
state-value function:  
![14](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/14.png)   
或者action-value function:  
![15](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/15.png)   
**蒙特卡洛方法估计value function**:  
![16](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/16.png)   
如果状态空间太大,可以考虑参数化函数近似:  
![17](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/17.png)   
**价值函数value functions的Bellman equation**:  
![18](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/18.png)   
注:  
![19](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/19.png)   
用backup diagram来表示这一关系:  
![20](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/20.jpg)   
注:  
![21](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/21.png)   
> 用图表示了一种关系，这种关系作为强化学习方法的核心，构成了强化学习中更新和传递操作的基准。这种操作将后继的状态(或者state-action对)的value回溯给当前状态(或者state-action对)的value。这本书使用backup diagrams来作为算法的图示总结(需要注意与状态转移图不同，backup diagrams中不同的状态节点可能表示相同的状态，比如说一个状态也可能是它的后继状态。而准确表示方向的箭头也被省略了，因为一般假定时间总是向下流转)。  
## 3.6 Optimal Policies and Optimal Value Functions  
解决一个强化学习任务需要意味着寻找一个从长远角度能够获得足够多的reward的policy。对finite MDP，可以按照如下的方式准确定义一个最优的policy。  
**Optimal Policies 和 Optimal Value Functions 的定义**:  
![23](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/23.png)   
![24](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/24.png)   
**导出Bellman optimality equation**:  
![25](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/25.png)   
optimal state-value和action-value的传递图:  
![26](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/26.jpg)   
> 小的改动，agent做选择action的节点由原来直接取policy下的期望值变为添加一段弧来表示选取最大值对应的action。  

说明:  
![27](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/27.png)   
![28](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/28.png)   
optimal action-value function能够在不知道任何关于可能的后继状态以及它们的value取值的情况选择最优的actions，也就是说一旦知道了optimal action-value function，那么选择最优actions无需知道任何关于environment的动态变化信息(dynamics)。  
**贝尔曼最优方程 应用实例**:  
![29](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/29.png)   
![30](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_3/30.png)   
## 3.7 Optimality and Approximation  
实际问题中，Bellman optimality equation可能不易求解，很多强化学习方法就是围绕如何近似求解Bellman optimality equation而进行的。  
对于我们感兴趣的各种任务，只能以极高的计算成本才能生成最优策略。即使我们有一个完整和准确的环境动态模型， 通常不可能通过求解贝尔曼最优方程来简单地计算最优策略。可用的内存是一个重要的限制。 强化学习问题的框架迫使我们解决近似问题。强化学习的在线性质使得其有可能以更多的方式来近似最优策略，以便为经常遇到的状态作出良好的决策，而不用花费很少的努力来处理不经常遇到的状态。 这是将强化学习与其他方法区分开来，近似解决MDP问题的一个关键属性。  
## 3.8 Summary  
强化学习需要从交互学习如何做出选择来达到某个目标。强化学习中agent与environment之间的交互过程体现在一个离散的时间序列上，明确它们之间的相互作用就定义了一个特定的任务.强化学习问题的要素:  
>actions是用户做出的选择  
> states是做出选择的基础  
> rewards是评价选择的基准  

agent可以知晓并控制agent内部的一切；任意定义在agent外部的事物，agent不能完全控制，但是是否完全知晓是具体情况而定。  
policy是定义在states上的函数，它给出了一套agent选择actions的随机规则 。  
agent的目标是从长远角度最大化接收到的reward的总和。  
  
return是关于future rewards的函数，同时也是agent尝试去最大化的量。  
依据任务的特性(episodic or continuing)以及是否考虑discounted reward，return有一些不同的定义方式。不考虑discounted的公式适用于episodic tasks，这种任务中agent与environment之间的交互过程可以分成多个episodes。  
考虑discounted的公式适用于continuing tasks，这种任务中agent与environment的交互不能分成多个episodes，而是没有限制地持续下去。  
  
[参考1](https://zhuanlan.zhihu.com/p/55079492)   
[参考2](https://zhuanlan.zhihu.com/p/51283820)   
