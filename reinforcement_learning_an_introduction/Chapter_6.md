# Chapter_6: 时序差分学习（Temporal-Difference Learning）   
temporal-difference (TD) learning是**是蒙特卡洛思想和动态规划(DP)的结合**。  
**与蒙特卡洛方法类似**，TD方法可以直接从原经验数据中学习，而不需要知道环境的动态更新模型。  
**与动态规划方法类似**，TD方法更新的估计值一定程度上依赖于其他的估计值，而不是等待最终的结果(they bootstrap)。  
> 第七章将会介绍n-step算法，这种算法搭建了从TD到蒙特卡洛方法的桥梁.  
> 第十二章将会介绍TD(![0](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/0.svg) )算法，这种算法统一了它们。  

本章仍然从**策略评估(policy evaluation)或预测(prediction)问题**出发.  
> 建立给定策略![1](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/1.svg) 对应的value function ![2](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/2.svg)  的估计。对于控制(control)问题(寻找最优策略)，  
>  对于**控制问题**（找到最优策略），DP、TD和蒙特卡洛方法都使用广义策略迭代（GPI）的一些变体。 方法的差异主要在于它们对预测问题的方法的差异。  
##  6.1 TD Prediction  
TD和蒙特卡洛方法都是利用经验数据来解决prediction问题。   
给定一些服从策略![1](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/1.svg)的经验数据，两种方法都是更新经验数据中出现的非终止状态 ![3](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/3.svg)  的value function ![2](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/2.svg)   对应的 V 。   
**蒙特卡洛方法与TD方法的对比**:  
> ![4](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/4.png)   
> ![46](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/46.png)   
> ![47](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/47.png)   

**TD(0)方法的算法流程**:  
> ![5](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/5.jpg)   

TD(0)方法的更新一定程度上基于已有的估计值，这和DP一样，都属于bootstrapping的方法。  
第三章中有如下表达式:  
![6](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/6.svg)   
大体来说，蒙特卡洛方法使用(6.3)式作为目标，而DP方法使用(6.4)式作为目标。  
**蒙特卡洛方法的目标是一种估计是因为**(6.3)式的期望值是未知的。一次采样中的return就被用来替代实际的期望return。  
**DP方法的目标是一种估计**:  
> ![7](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/7.png)   
  
**TD方法的目标也是一种估计**:  
> ![8](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/8.png)   
  
**TD方法结合了蒙特卡洛方法的采样和DP方法的bootstrapping。TD(0)方法的回溯图:**:  
> ![9](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/9.png)   
> 图中顶端状态节点的估计值的更新依赖于一次从它转移到下一状态的采样。  
> 我们将TD和蒙特卡洛方法的更新称为样本更新(sample updates)，因为它们都包含考虑后继状态(或者state-action pair)的样本，并且采用后继状态的估计值以及到达这一后继状态过程中所接受到的reward值来回溯到顶端节点计算状态value，从而更新相应状态(或者state-action pair)value的估计值。  
> 样本更新与DP方法的期望更新不同在样本更新是基于后继状态的一个样本而不是基于所有可能的后继状态的分布信息。  

**TD error**(TD(0)方法更新表达式(6.2)中括号里的数值可以看做是是一种误差)且**如果数组 V 在一个episode内没有发生变化(类似于蒙特卡洛方法中的情形)，那么蒙特卡洛方法的误差可以写成TD error的累积和**:  
> ![10](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/10.png)   
> **这个等式在 V 随着一个episode内的更新会发生变化时并不成立**   (TD(0)更新过程就会改变 V )  ，但是如果步长规模较小，等式仍然近似成立。  
> 这一等式的泛化在时序差分学习(temporal-difference learning)的理论和算法中有非常重要的作用。  
  
**Example 6.1: Driving Home**   
> 当你每天下班回家的时候，你试着预测回家所花的时间。当你离开办公室时，你注意到时间、周几、天气以及任意别的可能相关的事。假如你在周五的6点离开办公室，并且你估计回家需要花30分钟。你在6:05到达你的车的位置，你发现这时候开始下雨了。雨天的交通会慢一些，因此你重新估计回家所需要的时间为35分钟，合计40分钟。15分钟后你行驶完了回家途中的高速公路部分，你到了二级公路并且重新估计总的花费时间为35分钟。但是很不幸你跟在一辆很慢的货车后面，道路太窄你并不能超车。直到6:40你结束了跟车，驶进了一个小巷中。三分钟后你回到了家。你回家过程中的状态、时间、预测序列可以如下表示:   
> ![11](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/11.png)   
这个例子中的**reward是**每一段旅程的经过时长。  
> 这里并不考虑衰减，因此**每个状态的return是**从该状态开始后实际的花费时长。  
> **每个状态的value是**期望花费时间。   
> 表中的第二个数字列表示的就是当前**对遇到的状态的value的估计**。  
> 
>  蒙特卡洛方法(左)与TD方法(右)在driving home例子上的比较:  
> ![12](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/12.jpg)   
> 蒙特卡洛方法的分析:  
> ![13](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/13.png) 
> TD方法的分析:  
> ![14](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/14.png)   
## 6.2 Advantages of TD Prediction Methods
**TD方法优于DP方法的地方在于TD方法并不依赖实际的environment的模型，也就是reward和可能的下一状态的概率分布**。  
**相比于蒙特卡洛方法，TD方法一个最明显的优点在于**:  
> 它是在线地以完全增量形式实现(on-line, fully incremental fashion)。蒙特卡洛方法需要等到一个episode结束才能够更新，只有episode终止了return才能知晓。而**TD方法只需要等到下一个时间步**。这通常也是最关键的考虑，因为一些应用的episode的持续时间步很长，如果等到episode终止再学习那会很慢。还有一些应用属于continuing tasks，时间步是无穷的。最后，正如前一章节提到过的，一些蒙特卡洛方法会忽略或者低估一些通过采取试验性行为得到的episode，这会减缓学习的速度。TD方法不太会受这些问题的影响，因为它是根据每次状态转移学习，而不管后续的actions是什么。   
  
  **TD方法是合理的**:   
  > ![15](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/15.png)   
    
  **TD方法和蒙特卡洛方法哪个收敛的更快**?  
> 如果TD方法和蒙特卡洛方法都渐近收敛到正确的预测结果，那么自然就有关于哪个方法收敛更快的问题，以及哪个方法可以更高效地利用有限的数据。现在这是个open question，因为**现今还没有人能够从数学的角度证明其中一个方法比另一个方法收敛快**。事实上，如何以合适的方式来表达这一问题都是不确切的。  
  
在实际应用中，TD方法在随机任务上通常constant-![16](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/16.svg) MC方法收敛得更快，举个例子 如Example 6.2所示。  

![17](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/17.png)   
> Markov reward process，简称MRP，是不包含actions的马尔科夫决策过程。  
> 当专注于预测问题的时候，通常仅仅考虑MRP，此时没有必要考虑因为agent的影响改变environment动态过程。  
> 在这个MRP中，所有的episode都从中心状态C开始，然后在每个时间步以相同的概率向左或者向右前进一个状态节点。当到达最左边或者最右边的状态时，episodes终止。除了终止在最右边状态处得到+1的reward外，到达其它状态的reward都是0。  
> 比如说，一个可能的episode包括如下的state-and-reward序列：C,0,B,0,C,0,D,0,E,1。这一任务不考虑衰减，因此每一状态的实际value就是从该状态开始终止在最右边状态的概率。因此，中心状态的实际value是![18](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/18.svg) ，而所有状态的实际value从左到右依次为![19](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/19.svg) 。   
> ![20](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/20.png)   
  
##  6.3 Optimality of TD(0)  
假设只有有限数目的经验数据，比如10个episode或者100个时间步。   
**batch updating方法**:  
> 在这种情形下，**一种适用于增量学习方法的通用的策略是重复使用这些经验数据直至方法收敛到一个结果**。给定一个近似的value function V，(6.1)式以及(6.2)式明确的增量在每一次访问一个非终止的节点的时间步时都会被计算，但是value function通过求取所有增量的和却只改变了一次，然后所有可获得的经验数据在新的value function上通过产生全部新的增量又一遍被学习。就这样进行下去，直至value function收敛。**这种方法被称为batch updating**，因为更新只在学习完一个完整地batch的训练数据后才会进行。  

对于batch updating，只要step-size参数![21](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/21.svg) 充分小，TD(0)方法便确切地收敛到一个与![21](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/21.svg) 无关的结果。在同样的条件下，constant-![21](/home/tenglong/0.png)  MC方法确切地收敛到一个与上述不同的结果。  
**理解这两个结果可以助于理解两种方法的区别**。在通常的更新中，方法并不一定一直往各自的batch结果更新，而是在某种意义上往这些方向更新。为了更好理解在任意任务中的两种结果，**先看一些实例**。   
**Example 6.3: Random walk under batch updating**:  
> ![22](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/22.png)   
> 蒙特卡洛方法只是在有限的情况下是最优的，而TD方法最优性的表现方式与预测return更为相关。  

**Example 6.4: You are the Predictor**  
> ![23](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/23.png)   
> ![24](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/24.png)   
  
Example 6.4展示了**batch TD(0)方法以及batch蒙特卡洛方法的估计的一般差异**:  
>**batch蒙特卡洛方法总是最小化训练集上关于value估计的均方误差，而batch TD(0)方法总是寻求马尔科夫过程的最大似然估计**。  
> 通常来说，**最大似然估计得到的参数能够最大化生成训练数据的概率**。  
> 在这种情况下，最大似然估计是基于观察到的episodes自然得到的马尔科夫过程的模型：从 i 到 j 的转移概率的估计基于观测到的episodes中从 i 到 j 的占比；相应的期望reward是观察到的从 i 到 j 的reward的平均。  
> 如果一个模型完全正确，那么我们可以完全准确地计算出value function的估计。因为这等价于假定underlying process的估计是确定性的，而不是近似的，称为**certainly-equivalence estimate**。一般来说，batch TD(0)方法收敛于certainly-equivalence estimate。  

**TD方法比蒙特卡洛方法收敛更快的原因**:  
> 在batch形式下，TD(0)方法比蒙特卡洛方法收敛更快是因为它计算了实际的certainly-equivalence estimate。  
> nonbatch TD(0)比constant-![25](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/25.svg) MC方法收敛更快是因为它趋于一个更好的估计，即使它并不总是趋于更好的估计。   
> ![26](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/26.png)   

**尽管certainly-equivalence estimate在某种意义上是最优解，但是直接计算几乎是不可行的。在状态数很多的任务上，TD方法可能是近似certainly-equivalence解答的唯一可行方法**:  
> ![27](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/27.png)   
## 6.4 Sarsa: On-policy TD Control  
接下来考虑将TD预测方法用在control问题上，通常我们会遵循general policy iteration(GPI)的模式，只有这一次使用TD方法作为评估或者预测的部分。  
与蒙特卡洛方法相似，我们需要平衡exploration和exploitation，进而方法被分成了两类：**on-policy和off-policy**。**这一小节将介绍一种on-policy的基于TD的control方法**。   
**一种on-policy的基于TD的control方法---Sarsa算法**:  
> ![28](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/28.png)   
  
**Sarsa算法的backup diagram**见下图:  
> ![29](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/29.png)   
  
 **一般形式的Sarsa control算法如下图**:  
 > ![30](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/30.png)   
 
 **Example 6.5: Windy Gridworld**:   
 > ![31](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/31.png)   
##  6.5 Q-learning: Off-policy TD Control  
**一种off-policy TD control算法---Q-learning方法**:   
 > ![32](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/32.png)   

**Q-learning算法流程图**:  
> ![33](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/33.jpg)   

**Q-learning算法的backup diagram**:  
> ![34](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/34.png)   
> 关于Q-learning算法的backup diagram，(6.8)式更新的是一个state-action pair，因此回溯图的根节点是小的实心点表示的action节点，而更新的数值同样也是源自action节点，选取的是下一状态可能的actions中action-value最大对应的节点，因此回溯图的叶节点应该是所有的这些action节点。需要注意的是我们选取的是最大值，通常用一段弧表示，这跟这个系列文章里图3.6右图相似。Q-learning算法的backup diagram见上图。  

**Example 6.6: Cliff Walking**:  
> ![35](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/35.png)   
## 6.6 Expected Sarsa
**Expected Sarsa算法**:  
> ![36](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/36.png)   

**Expected Sarsa的backup diagram**:  
> ![37](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/37.png)   
  
**Expected Sarsa与Sarsa与Q-learning在cliff-walking任务上的性能对比**:  
  > ![38](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/38.png)   
  > ![39](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/39.png)   
## 6.7 Maximization Bias and Double Learning  
**maximization bias**:  
  > ![40](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/40.png)   

**Example 6.7: Maximization Bias Example** :  
> ![41](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/41.png)   

**能够避免maximization bias的算法---double learning的思想**:  
> ![42](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/42.png)   

**将double learning的思想拓展到full MDPs的算法中**:  
> ![43](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/43.png)   

**完整的Double Q-learning算法如下图给出**:  
> 这个算法也是图6.16中结果对应的算法，图6.16反映了这个算法似乎消除了maximization bias造成的损失。当然，对于Sarsa和Expected Sarsa，也有相应的double learning的版本。  
> ![44](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/44.jpg)   
## 6.8 Games, Afterstates, and Other Special Cases  
> 这本书希望能够提出一个针对一大类任务的统一方法，但是通常总有例外的任务，这些任务需要特殊处理才能有更好的效果。  
  
  **afterstates**:  
> 比如说，一个一般方法是学习action-value function，但是在第1章中我们提出了一种学会tic-tac-toe的TD方法，这个TD方法训练的结果更像state-value function。如果我们更细致地分析那个例子，其实比较容易看出学习到的function通常既不是action-value function也不是state-value function。一个传统的state-value function对那些agent可以选择action的状态进行评估，但是tic-tac-toe中使用的state-value function却是估计的玩家选择action后的棋盘位置，这被称为afterstates，value function相应称为afterstate value functions。当我们知道environment的部分动态变化过程(无需全部知晓)时afterstates是很有用的。比如说，在一些games中，我们可以知道我们选择的action随即的效果。我们知道落子后各个棋子的位置，但是不知道对手的应对方式。afterstate value functions可以有效利用这种关于environment的knowledge，因此可以产生一种更为有效的方法。  
  
从tic-tac-toe的例子可以看出设计考虑afterstates的算法是更为有效的。一个传统的action-value function会将每个位置和每个移动映射到一个value的评估值。但是**许多position-move pairs会得到相同的位置结果**，比如图6.18
> ![45](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_6/45.png)   
  
**afterstates在很多任务中都会出现，而不仅仅在game中**。  
> 比如在排队任务中，有一些关于分配客户给服务人员、拒绝客户的actions。在这种情形下，actions事实上是定义在它们造成的影响上的，这是完全可以获知的。   

**描述所有可能的特殊问题以及相应的特殊学习算法是不可能的。然而，这本书沿用的方法准则可以被应用得很广**  
> 比如说，afterstate方法可以根据generalized policy iteration适当地描述，策略和afterstate value function之间的交互本质上跟之前的定义仍然一致。在很多情况下，研究人员仍然会因为持续探索的需求面对on-policy和off-policy的选择。  
## 6.9 Summary  
这一章**介绍了一种新的强化学习方法——temporal-difference(TD) learning，并且展示了该方法如何应用在强化学习问题上**。与之前一样，我们将全部问题分成prediction问题和control问题。  
TD方法跟蒙特卡洛方法一样都可以解决预测问题。两种情形都可以通过generalized policy iteration(GPI)的想法拓展到control问题。这是一种使近似策略和value functions通过交互逐渐趋于最优的想法。   
组成GPI的第一个过程是驱使value function能够准确预测当前策略的return，也就是预测问题。另一个过程是驱使策略在基于当前value function的情况得到提升。当第一个过程是基于经验数据的时候，维持充足的探索需要考虑随之出现的复杂度。  
我们可以根据TD方法有没有考虑这个复杂度(on-policy还是off-policy)来对TD control方法进行分类。**Sarsa是on-policy的方法，Q-learning是off-policy的方法**，我们**这一章提到的Expected Sarsa也是off-policy的方法(其可以是on-policy方法)**。还有第三种方式可以将TD方法拓展到control问题上，叫做actor-critic方法，这一章不会涉及，而第13章会详细说明。  
这一章提到的方法是现在被广泛应用的强化学习方法。这可能是由于它们的简洁性：**它们可以通过on-line实现，以很小的计算的代价，通过与environment的交互产生经验数据；它们可以通过一个等式完整地表示出来，而这个等式可以通过简单的计算机程序实现**。  
接下来的几章将会拓展这些算法，尽管会使这些方法稍微复杂化，但是却能得到明显提升的性能。所有算法都会保留下面提到的本质：**它们可以通过很简单的计算on-line产生经验数据，同时是TD errors驱动的。**  
这一章介绍的TD方法准确来讲可以称为**one-step，tabular，model-free**的TD方法。  
接下来的两章将会将TD方法拓展到multistep的形式(联系蒙特卡洛方法)以及建立包含environment模型的形式(联系动态规划)。  
而这本书的第二部分将会将它们拓展到各种函数近似的形式而不仅仅是表格形式(联系深度学习与人工神经网络)。  

最后需要注意的是，这一章是在强化学习问题的范畴下讨论的TD方法，但是TD方法实际上更为普遍。它们是学习预测动态系统的长期效果的通用方法。比如说，TD方法可能与预测金融数据、寿命、选举结果、天气模式、动物行为、电力需求以及客户购买力等有关。只有当TD方法独立于强化学习中的应用而被视为纯粹的预测问题来进行分析时，它们的理论性质才能被很好地理解。即便如此，TD学习方法其他的一些潜在应用仍然没有被充分开发出来。  

[参考1](https://zhuanlan.zhihu.com/p/56059574)   
[参考2](https://rl.qiwihui.com/zh_CN/latest/partI/chapter6/temporal_difference_learning.html)   
