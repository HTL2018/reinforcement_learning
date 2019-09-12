# Chapter 7 n-step Bootstrapping( n 步自助法)  
 **本章内容**: 主要是将前面的 MC 和 1-step TD 方法做了一个整合，提出了 n-step TD 方法。   
**内容顺序安排**: 先考虑 prediction 问题然后再考虑 control 问题。即,先考虑 n-step 方法如何在给定策略时预测 return，然后我们拓展到 action values 以及 control 的方法。  
**`n-step TD` 方法的优点**:  
 > 1.  不管是 `MC 方法`还是 `TD 方法`都不能总保证算法性能最好。这一章将提出 n-step 方法，这个方法可以概括前面提到的两种方法。因此对于特定任务的需求，我们**可以平稳地从一种方法过渡到另一种方法**。n-step TD方法涵盖了很大范围，其中一端是 MC 方法，而另一端是单步的TD方法。最佳的方法通常位于两个极端方法的中间。   
 > 2. n-step TD 方法的另一种优点在于**它使我们不再局限于时间步**。(在 one-step TD 方法中，同样的时间步明确了 action 变化的频率以及 bootstrapping 进行的时间间隔。)在很多应用中，我们希望 action 能够很快更新，这样任何已经发生的变化都会被算法考虑进去。但是`当经过一段重要且可辨识的状态变化发生的时间步上时，bootstrapping的效果才最好`。如果使用 one-step TD 方法，这些时间间隔是一样的，因此需要一些折中。n-step方法可以让 bootstrapping 出现在多个不同的时间步上，而不受单个时间步的限制。  
 
`N-step方法`的idea和`eligibility traces`很像，`eligibility traces`同时使用多个不同的`time intervals`进行`bootstarp`。   
## 7.1 n-step TD Prediction  
`蒙特卡洛方法`更新一个 state 的 value 是基于这个 episode 中从 state 开始直至episode 终止所观测到的 reward 序列。  
`one-step TD 方法`只是基于下一个 reward，以及bootstrapping 下一状态的 value 估计作为后续剩下 reward 的替代。  
`n-step TD方法`:一种中间的方法可以按照下面的方式更新 state 的 value：它基于下面不至一步的 reward 序列，但不使用后续剩下的全部 reward。   
> 比如说，一个 two-step 的方法会基于下面两个 reward 以及两步之后的 state 的 value 估计更新当前 state 的 value。类似地，可以得到 three-step 更新、four-step 更新等等。  
  
  如下图展示了对 ![0](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_7/0.svg) 的 **n-step 更新的回溯图**，其中最左边是 one-step TD 更新，从左向右逐渐转化为蒙特卡洛更新:  
  ![1](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_7/1.png)   
    
  **n-step TD 方法**:  
  应用 n-step 更新的方法仍然属于 TD 方法，因为它们仍然`基于后续 state 的` value 估计来改变之前 state 的 value 估计，只是现在之后的状态估计不是一个时间步后的，而是 `n 个时间步`后的。temporal difference 扩展到 n 个时间步的方法称为 `n-step TD 方法`。  
> 前一章介绍的TD方法均采用的是 one-step 更新，这也是我们称它为 one-step TD 方法的原因。  
  
  **n-step TD 方法的严格描述**:  
  > ![2](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_7/2.png)   
  > 注:  
  > ![3](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_7/3.png)   
  
  **n-step TD的算法描述**:  
 >  ![4](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_7/4.jpg)   
 
 **error reduction property**:  
>  ![5](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_7/5.png)   
> 更直观的解释:  
> ![6](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_7/6.png)   
  >  n-step TD 方法的收敛性由 error reduction property 这一性质来保证.有了这个性质，理论上可以证明 n-step TD 确实能够收敛。因此 n-step TD 方法构成了一类合理的方法，而 one-step TD 方法和蒙特卡洛方法都是这种方法的极端情形的形式。  
  
 **Example 7.1: n-step TD Methods on the Random Walk**  
 > ![7](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_7/7.png)   

## 7.2 n-step Sarsa  
n-step方法不仅可以用来 prediction，还可以用来 control。  
**本节内容**:  介绍如何**将n-step方法直接结合 Sarsa** 构造一个on-policy的TD control方法，称之为 **n-step Sarsa**。  
前面一章提到的 Sarsa 后面称它为 one-step Sarsa，用 Sarsa(0) 表示。  
**n-step 方法的回溯图**:  
> ![8](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_7/8.png)   
> 注意:  
> n-step Sarsa的回溯图(图7.3)，与n-step TD的回溯图(图7.1)相似，**由交替变化的 states 和 actions 组成**，不同的是 **Sarsa 的根节点和叶节点都是表示 actions 而不是 states**(对比之下,n-step 方法的根节点和叶节点都是表示 states而不是actions)  。   
> 倒数第二列为 Monte Carlo 方法.(**无穷步的Sarsa即为Monte Carlo方法**).  
> 最右边是 **n-step Expected Sarsa** 的回溯图。  

**算法的核心想法是**:  
>将 prediction 问题中的 states 替换成 actions (state-action pairs)，然后使用 ![9](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_7/9.svg) -greedy策略。   

**n-step Sarsa算法的更新公式**:  
> 我们根据estimated action values **重新定义 n-step returns** (update targets)为:  
> ![10](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_7/10.png)   

**n-step Sarsa 算法的的伪代码**:  
> ![11](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_7/11.jpg)   

**n-step Sarsa 方法能够加速 one-step Sarsa 方法的原因**:  
> ![12](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_7/12.png)   
> 图7.4 展示了 n-step Sarsa 方法能够加速 one-step Sarsa 方法的原因。  
> 左图表示一个 episode 中 agent 选取的路径，路径终止于对应高 reward 的状态 G。在这个例子中除到达状态 G 是正的 reward 外，别的对应的 reward 都是 0。  
> 另两张图展示了通过这个 episode 的训练 one-step 和 n-step 方法加强的 action value。  
> one-step 方法只会加强直接到达 G 的 action 的 value，而 n-step 方法会加强这个 episode 中到达状态 G 前的最后 n 个 action的 value，因此它比 one-step 学习到的更多。  

**Expected Sarsa 的 n-step 的形式**:  
> Expected Sarsa 的 n-step 的形式的回溯图见图7.3最右侧。  
> 和 n-step Sarsa 一样，它包含了一个action和state构成的序列，只是**最后出现了分支，表示最后一个状态下所有可能的action，各个叶节点包含根据策略 ![13](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_7/13.svg)  选取该 action 的概率**。  
> 算法和 n-step Sarsa 基本一致，只需要将 n-step return 重新定义为：  
> ![14](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_7/14.png)   
## 7.3 n-step Off-policy Learning by Important Sampling  
**off-policy**:   
> 之前提过 `off-policy` 通过按照策略 b 交互的数据来学习策略![13](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_7/13.svg) 的 value function。通常 ![13](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_7/13.svg) 是按照当前 action-value function 估计的贪心策略，而 b 是某种更偏向 exploratory 的策略，比如 ![15](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_7/15.svg) -greedy。为了使用来自策略 b 的数据，我们必须考虑两种策略之间的差异, 使用采用策略的相对概率(见章节5.5)。   

**importance sampling ratio**:  
> 在 n-step 方法中，returns是构建在 n 个时间步之上的，所以我们只考虑这 n  个 actions 的相对概率。比如说，在一种 n-step TD 方法 off-policy 的简单形式里，对时间步 `t` 的更新(实际在时间步`t+n`进行)可以简单地加以权重![16](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_7/16.svg):    
> ![17](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_7/17.png)   
> 如果这些 actions 中存在一个 action 始终不会被策略![13](/home/tenglong/0.png)选取(也就是 ![18](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_7/18.svg)  )，那么 n-step return 赋予的权重为 0，因此 n-step return 会被忽略。  
> 另一方面，如果凑巧一个 action 在 ![13](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_7/13.svg) 中选择的概率远大于在策略  `b` 中的，那么相应的权重会增加。这是合理的，因为这个 action 是策略 ![13](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_7/13.svg) 的特有的(因此我们要学习它)，但是都在源自策略 `b` 的数据中它很少会出现，因此我们在它出现时赋予它较大的更新权重。  
> 注意到如果两种策略完全一样(on-policy)，那么 importance ratio 始终是 1。因此新的更新公式(7.7)涵盖并可以替换之前介绍的 n-step TD 更新。  

**将n-step Sarsa更新公式替换成如下的 off-policy 的形式**:  
> ![19](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_7/19.png)   

**算法的伪代码**:  
> ![20](/home/tenglong/0.png)   

**n-step Expected Sarsa 的 off-policy 形式**:  
> ![21](/home/tenglong/0.png)   
## 7.4 *Per-reward Off-policy Methods  
前面展示的 multi-step off-policy 非常简单且概念上很清楚，但有可能不是最有效的。   
一种更高端的方法是用5.9节介绍的 `per-reward importance sampling`的思想。为了理解这种方法，先来看普通的 n-step return （7.1）可以采用递归的写法：  
![22](/home/tenglong/0.png)   
本小节，前面小节以及第五章所用的 importance sampling 使得 off-policy 学习成为可能。但是也付出了**增加更新所带来的方差**的代价。  
**高方差**使得我们要用小步长参数，使得学习很缓慢。一般情况下，off-policy 训练要比 on-policy 训练要慢——毕竟，数据与要学的东西相关性要小一些。  
但是，我们这里提的方法很可能都是可以提升的。一种可能是根据观察到的方差快速自适应调整步长。下一小节我们考虑没有采用 importance sampling 的 off-policy 方法。  
## 7.5 Off-policy Learning Without Importance Sampling: The n-step Tree Backup Algorithm  ( 无重要性采样的离策略学习：n步树备份算法)  
没有重要性采样的离策略学习:   
> 第6章中的`Q-learning`和`预期的Sarsa`针对一步式案例进行了此操作.   
  
**算法的思想**(不使用 importance sampling 的 off-policy 的多步算法):  
> 用下图的三步 tree-backup 回溯图（backup diagram）来介绍:  
> ![23](/home/tenglong/0.png)   
> **backup diagram 的解读**:  
> 顺着中线往下且标注的是三个采样的 states 和 rewards 以及 两个采样的 actions。这些是表示发生在初始 state-action pair `St`， `At` 的事件的随机变量。 挂在每个 state 旁边的是没被选到的 actions。（对于最后的 state， 所有的 actions 都被看做还没被选择）。由于我们没有未被选择 action 的采样数据，我们 `bootstrap`(我们引导并使用它们的值估计来形成更新目标)，利用他们的 value 来构成更新需要的目标（target）。  
> 这稍微拓展了一下回溯图的思想。  
> 现在，我们已经更新了图最顶端的节点的估计 value，更新目标结合了随之而来的 rewards（适当衰减了）和底端节点的估计 values。  
>**在 tree-backup ，目标包含了所有层上所有悬挂在一边的 actions 的估计 value。这是被称作 tree-backup 的原因，这是利用整个树中所有的估计 action values 进行的更新**。   
> **更准确地表述**:  
> ![24](/home/tenglong/0.png)   

我们可以将这个三步 tree-backup 的更新看做包括 6 个半步（half-step），在`从一个 action开始到随后的 state 采样` 半步以及`考虑所有在此 policy 下选择 action 的概率的求期望` 半步。  

**相关公式推导**:   
> ![25](/home/tenglong/0.png)   
> ![26](/home/tenglong/0.png)   
> ![27](/home/tenglong/0.png)   
> 没有衰减相乘的可以看作衰减因子为1。则n-step Sarsa 中用来更新 action-value 的目标为:  
> ![28](/home/tenglong/0.png)   

**算法伪代码**:  
> ![29](/home/tenglong/0.png)   
## 7.6 *A Unifying Algorithm: n-step Q(σ)  (\*统一算法：n步 Q(σ))    
本章到目前为止我们已经考虑了三种不同的 action-value 算法，对应于图7.5展示的前三个回溯图。  
> `n-step Sarsa` 是基于所有实际采样的数据.  
> `tree-backup 算法`则是利用所有的 state-to-action 的分岔的转换关系，不需要实际采样.  
> `n-step Expected Sarsa` 则是有着所有的采样数据，除了最后一层是分岔的。  

**将三种方法统一起来**？  
> 一种思想如图 7.5 的第四种所示。  
> 这种**思想是**每一步决定是偏向像 Sarsa 一样采取 action 作为样本，还是考虑像 tree-backup 中一样在所有 actions 上的期望。然后如果一直选择采样，就会得到 Sarsa 方法，如果从不采样，就会得到 tree-backup 算法。Expected Sarsa 是除了最后一次外一路都进行采样。  
> 这种方法有其他很多种可能，就像下图这样的。我们甚至可以考虑在采样和去期望之间连续变化。  

**backup diagram**:   
> ![30](/home/tenglong/0.png)   

**n-step Q(σ)相关公式推导**:  
> ![31](/home/tenglong/0.png)   
> ![32](/home/tenglong/0.png)   
> ![33](/home/tenglong/0.png)   

**算法伪代码**:  
> ![34](/home/tenglong/0.png)   
## 7.7 Summary  
在本章中我们已经拓展了一系列的介于之前章节介绍的单步 TD 方法 以及 MC 方法之间的 temporal-difference learning 方法。  
有着中等数量的 bootstrapping 很重要，因为它们一般表现地会比取极端要好（单步 TD 以及 MC）。  
 我们本章主要集中于 n-step 方法。下面两个 4-step 回溯图一起总结了本章的方法。   
> ![35](/home/tenglong/0.png)   
> 带有 importance sampling 的 n-step TD 的 state-value 更新以及 n-step `Q(σ)` 的 action-value 更新。这些是 Expected Sarsa 和 Q-learning 的推广。  
> 所有的 n-step 方法在更新前都会有一个 `n` **时间步的延迟**，因为只有这样所需要的未来事件才能知晓。  
> **一个缺点**相比较于之前的方法是每一个时间步都会需要更多的计算。与 one-step 方法相比，n-step 需要更多的存储空间来记录 states，actions，rewards，有时还有前 n 步的别的信息。  
> 最终，在第12章，我们将会看到 multi-step TD 方法是如何利用 eligibility traces 在小存储空间以及计算复杂度实现的。但是还是会比 one-step 方法会消耗更多的计算资源。这些代价是值得的，因为可以脱离单步的限制。  

**尽管 n-step 方法相比较于采用 eligibility traces 的方法更复杂，他们仍有很多好处，我们已经寻求了通过在 n-step 情况下的两种 off-policy 方法来利用这些好处**。  
> **一种是基于 importance sampling**，概念上很简单，但是方差很大。如果目标策略（target policy）和行为策略（behavior policy）非常不同，可能需要更多的算法思想才能使其变得实用。  
> **另一种是基于 tree-backup 更新**，是通过随机 target policies 将 Q-learning 扩展到 n-step 情况。这没有应用 importance sampling，但是如果目标策略（target policy）和行为策略（behavior policy）非常不同， bootstrapping 只会扩展一些步，即使 n 非常大。  

[参考1](https://mxxhcm.github.io/2019/07/30/reinforcement-learning-an-introduction-%E7%AC%AC7%E7%AB%A0%E7%AC%94%E8%AE%B0/)   
[参考2](https://zhuanlan.zhihu.com/p/51920961)   
[参考3](https://zhuanlan.zhihu.com/p/58801845)   
[参考4](https://rl.qiwihui.com/zh_CN/latest/partI/chapter7/n_step_bootstrapping.html)   

