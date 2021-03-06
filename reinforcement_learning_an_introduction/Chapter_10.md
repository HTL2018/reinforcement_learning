# 第10章 基于函数逼近的同轨策略控制  
**本章内容:**   
> 本章中，我们回到**控制问题**，现在使用动作-价值函数 q^(s,a,w)≈q∗(s,a) 的参数近似， 其中 w∈Rd 是有限维权重向量。    
> 我们将上一章中提出的函数近似思想从状态值扩展到动作价值。 然后我们扩展它们以控制遵循策略GPI的一般模式，使用 ε 贪婪进行动作选择,将它们扩展到控制问题中。    

## 10.1 分幕式半梯度控制
**分幕式半梯度单步Sarsa基本思想与原理:**   
![0](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_10/0.png)     
为了形成控制方法，我们需要将这些**动作价值预测方法**与**策略改进**和**行动选择技术**相结合。    
 适用于连续动作或来自大型离散集的动作的适当技术是正在进行的研究主题，但尚未明确解决。    
另一方面，如果动作集是离散的并且不是太大，那么我们可以使用前面章节中已经开发的技术。    
![1](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_10/1.png)    
## 10.2 半梯度n步Sarsa
**算法原理:**   
![2](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_10/2.png)   
**算法伪代码:**    
![3](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_10/3.png)    
## 10.3 平均收益：持续任务中的新的问题的设定  
除**"分幕式"设定**和**"折扣"设定**外,此处引出马尔科夫决策问题的第三个经典的目标设定:**"平均收益"设定**.    
 **与"折扣"设定一样**，平均奖励 设置适用于持续存在的问题，即智能体与环境的交互一直持续而没有对应的终止和开始状态.    
 **与"折扣"设定不同的是**:这里不考虑任何折扣,智能体对于延迟收益的重视和即时收益一样.    
 
 **平均收益的定义:**:   
 ![4](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_10/4.png)    
 **差分回报和差分价值函数:**    
 ![5](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_10/5.png)    
 ![6](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_10/6.png)    
 **完整算法的伪代码**:  
 ![7](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_10/7.png)   
##  10.4 弃用折扣  
对于策略 π，折扣回报的平均值总是 r(π)/(1−γ)， 也就是说，它本质上就是平均回报 r(π)。   
特别是，**平均折扣回报设置中所有策略的 排序 与平均收益设置中的排序完全相同**。   
 因此，折扣率 γ 对问题的表述没有影响。它实际上可能为 零，排序保持不变。   
 
**基本思想**可以通过对称的观点来解释。 每个时步与其他步骤完全相同。    
 ![8](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_10/8.png)    
 此示例和框中更一般的参数表明，如果我们优化了在策略分布上的折扣值， 那么效果将与优化 未折扣 的平均奖励相同；γ 的实际值将无效。    

 **折扣控制设置的困难的根本原因**:  
 > 通过函数近似，我们已经失去了策略提升定理（第4.2节）。 如果我们改变策略以提高一个状态的折扣值，那么我们就可以保证在任何有用的意义上改善整体策略。 这种保证是我们强化学习控制方法理论的关键。随着函数近似我们失去了它！
 
 **注意:**   
 > 缺乏策略提升定理也是分幕式设定和平均收益设定的理论空白。 一旦我们引入函数近似，我们就无法再保证在任何的设定下都一定会有策略的提升。    
##  10.5 差分半梯度n步Sarsa  
![9](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_10/9.png)   
 **完整算法的伪代码**:   
 ![10](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_10/10.png)    
##  本章小节:  
在本章中，我们将前一章介绍的参数化函数近似和半梯度下降的思想扩展到控制。     
  
 对于分幕式案例，延期是即时的，但对于持续的情况，我们必须基于最大化每个时步的 平均收益设置 来引入全新的问题公式。    
   
 令人惊讶的是，折扣设置不能在存在近似值的情况下进行控制。在大致情况下，大多数策略不能用价值函数表示。 剩下的任意策略需要排序，标量平均奖励 r(π) 提供了一种有效的方法。    
 
平均收益公式包括价值函数的新 **差分 版本**，**Bellman方程**和**TD误差**，但所有这些都与旧的并行，并且概念变化很小。 对于平均奖励情况，还有一组新的并行的差分算法。    
