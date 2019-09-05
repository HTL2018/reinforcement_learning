# Chapter_5: 蒙特卡洛方法（Monte Carlo Methods）  
**本章主要内容**: 强化学习中的蒙特卡洛算法(Monte Carlo Methods, 以下简称MC）   
**从学习任务来讲可分为**: 利用MC进行prediction和进行control   
**从学习方式来讲可分为**: on-policy和off-policy learning   
## 5.0 蒙特卡洛基本概念  
### 5.0.1 Dynamic Programming 与 Monte Carlo Methods的区别:  
1. Dynamic Programming方法假设了我们已知关于环境的模型，即它是一种model-based reinforcement learning。  
2. MC方法不需要知道环境模型，而是通过不断的采样来进行学习。  
3. 与动态规划方法相比，MC方法不需要进行**`bootstrap`**，即不需要根据接下来的状态的value来更新当前状态的value，所以对于很多Markov property条件不满足的实际问题，MC方法仍然适用。  
> Monte Carlo（蒙特卡洛）方法只需要 **`experience`**——对实际或者仿真中与 environment 的 interaction 中产生的 states, actions, rewards 的采样。  
> 从真实的经历中学习是非常令人瞩目的因为其不需要关于 environment 的动态变化的先验知识。从仿真中学习也非常得引人, 尽管需要一个 model， 但这个 model 仅仅需要产生一些 transitions 的采样，而不需要像动态规划一样需要知道所有 transitions 的概率分布。  
> 对于episodic task，我们可以进行多次episode的采样过程，并将每次的return记录下来再求经验平均值，得到相应的value function。  
### 5.0.2 Monte Carlo与多臂赌博机的区别:  
**Monte Carlo方法**对每个 state-action 对采样并对 return 取均值,**多臂老虎机问题**中采样并对每个 action 的 reward 取均值。   
**主要区别在于**这里有很多的 states， 可以看作是很多相关联的多臂老虎机问题。 即， 在某个state 执行某个 action 之后得到的 return 取决于在同一个 episode 中，后续的 states 下采取的 actions。 由于所有对 action 的选择都要经过学习，因此在早期的 state下，问题是不稳定的。  
> 解决这种不稳定问题:  
> ![0](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/0.png)   
### 5.0.3 Monte Carlo（MC）的基本思想:  
1. 用样本分布代替总体分布，估计一些总体分布的参数  
2. 简单来说，就是假设想知道一些真实分布的一些信息，比如期望，或函数的期望，如果我们不知道真实分布的表达式，或者知道，但是很难推导求解，就需要模拟出一批样本，再做平均，虽然有误差，可只要样本量足够大，根据大数定律还是收敛的   
3. 从 experience 中估计 value 的方法是对于那个 state 之后观测到的 return 求均值。随着观测到的 return 越来越多，其均值会收敛到期望值。这种思想被称为 Monte Carlo。  
## 5.1 蒙特卡洛预测 Monte Carlo Prediction  
MC方法进行prediction，即已知policy的情况下估计 value function的问题。   
由定义可知: state value function的定义是对于该state的return的期望：   
![1](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/1.svg)   
其中: return 的期望为 : 从这个state开始累加 discounted reward 的期望值。   
对于MC方法，我们将期望换为了`experience`平均值，在采样数量足够多的情况下，根据大数定律，`experience`平均值会收敛于期望。(`experience`含义见上)  
假设我们要估计 ![2](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/2.svg) ，并且我们有了若干episode的在policy ![3](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/3.svg) 下经过state s的采样集合，每经历一次state s，我们称为对 s 的一次访问**`visit`**，在一次episode中，s可能被访问若干次，其中第一次访问称为`first visit`  
因而我们可以有**两种相似但略有不同的MC方式**：  
**一种是**`first-visit MC`即用每个episode中第一次访问s的return来求经验平均值.  
**另一种是**`every-visit MC`即用每个episode中所有访问s的return来求经验平均值。(经验:experience 访问: visit, 具体含义见上)  
> **两种方法很像但有一些理论性质不一样**。  
> **first-visit MC**方法研究地更多，也是本章主要关注的重点。  
> **every-visit MC**方法可以很自然地扩展到function approximation 和 eligibility traces 中，这在之后的第九章和第十二章会进行讨论。   
### 5.1.1 First-visit MC 的实现步骤:  
![4](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/4.jpg)   
> 注:   
> ![5](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/5.png)   
> 改变框中 first-visit 条件，可以同理得到every-visit。  
> 对于action value function的MC prediction方法类似，只是将对state的访问转变为对state-action pair的访问。  
### 例5.1：二十一点（Blackjack） 
**规则简介**:  21点的游戏规则详细很容易就能够找到，这里进行简单的介绍。   
> 在这里智能体(Agent)扮演玩家(Player)，对方是庄家(Dealer)。   
> 点数(Score)：2-10的点数为牌面数字；J，Q，K是10点；A有两种算法，1或者11，算11总点数不超过21时则必须算成11(usable)，否则算作1。  
> 庄家需要亮(Show)一张牌，玩家根据自己手中的牌和庄家亮的牌决定是要牌(hits)还是停牌(sticks)。  
> 庄家要牌和停牌的规则是固定的，即点数小于17必须要牌，否则停牌。  
> 爆牌(goes bust)：牌总数操过21点，谁爆牌谁输，谁首先凑到21点谁赢，没有爆牌的时候谁点数大谁赢，同时凑到21点为和局。  

**转换成MDP**  
了解规则后，我们将游戏转换成MDP，MDP的几大要素：状态(S: State)，行动(A: Action)，奖励(R: Reward)，策略Policy，状态值函数V(s): State-Value Function，行动值函数Q(s, a)Action-Value Function。  
> 行动A：要牌(hits)还是停牌(sticks)   
> 状态S：状态是由双方目前牌的点数决定的，但是当玩家点数小于等于11时，当然会毫不犹豫选择要牌，所以真正涉及到做选择的状态是12-21点的状态，此时庄家亮牌有A-10种情况，再加上是否有11的A(usable A)，所以21点游戏中所有的状态一共只有200个。  
> 奖励R：玩家赢牌奖励为1，输牌奖励为-1，和局和其他状态奖励为0。  
> 策略Policy：该状态下，要牌和停牌的概率  

玩家采用的策略：一直要牌，直到点数和等于20或21时停止。  
为了使用蒙特卡洛方法找到这个策略下的状态价值函数， 我们使用一个模拟器模拟了许多次的游戏，游戏中玩家使用上述的策略。然后我们将每个状态的回报值求平均，作为对应状态的价值函数。 通过这种方法求得的价值函数如图5.1所示。 可以看到，如果A可用，相对于A不可用，估计的值会有更多不确定性，更加不规则，因为这些状态不是很常见。 经过500,000次的游戏，我们看到价值函数被近似得很好。  
![6](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/6.png)   
**在这个任务中，虽然我们对环境有完全的了解,为何选用ＭＣ而不用ＤＰ？**  
![7](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/7.png)   
### 例5.2：肥皂泡   
![10](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/10.png)   
假设一根线围成一个闭环，在肥皂水中浸泡后，表面形成了一个肥皂薄膜或者泡泡。 **如果线是不规则的但是已知，如何计算肥皂泡表面的形状**？ 已知泡泡的形状有一个特性：在表面任一点，受到临近的力之和为零（如果不为零，泡泡的形状会改变，直到稳定下来）。 这个性质意味着，泡泡表面上的每一点的高度等于周围点高度的平均值。此外，表面的形状必须符合线形成的边界。 **解决这个问题的常规办法**是，用网格分格这个区域，使用网格上一点的周围点来计算这点的高度，然后迭代地进行。 边界上的点的高度和线上的那点一致，然后其他的点的高度都可以从临近网格的点的高度求平均得到。 这个过程不断的迭代，很像动态规划（DP）迭代策略评估。最终，这个不断迭代的过程会收敛到很接近真实的表面形状。   
**这个问题和最初设计蒙特卡洛（MC）所涉及的问题是类似的**。除了上述提到的迭代计算的方法，我们还可以想象在表面进行随机漫步。 在网格上的每一点以等概率向临近的点移动，直到到达边界。 结果是，这些边界点的高度求得的期望值即是我们随机漫步起始点的高度（事实上，它恰好等于之前的迭代方法计算得到的值）。 因此，我们能够很好地得到表面上任意一点的高度值。只需要从该点开始，进行许多次随机漫步，然后将所有得到的边界高度值求平均。 如果我们仅仅对某一点或者某一小块区域的高度感兴趣，这个蒙特卡洛（MC）方法要比之前的迭代方法高效的多。   
###5.1.2 将 backup diagram (备份图) 的想法推广到蒙特卡洛的算法中  
![8](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/8.png)   
这个图表的顶部根节点是我们需要更新的量， 树枝和叶节点分别表示这些转移状态的奖励以及下个状态的估计价值。   
具体的，对于蒙特卡洛估计 ![9](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/9.svg) ，如图中所示，根节点是我们的起始状态的价值，之后的轨迹表示一个特定回合的经历，最后以终止状态结束。   
**与动态规划（DP）的图表**（![9](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/9.svg) 备份图）**对比**:  
1. 首先，动态规划（DP）的图表展示了所有的转移可能，列出了所有可能的下一状态，而蒙特卡洛（MC）在一个回合里只有一种转移可能。   
2. 其次，动态规划（DP）只包含了单步的转移状态价值，而蒙特卡洛（MC）表示一个回合从开始到结束的所有状态价值。 这些图表所表现的不同精确地反应了这两种算法的根本性的差异。   
>1. **需要注意**的是，**蒙特卡洛（MC）方法对每个状态的估计是独立的**，即是说，对这个状态的估计并不取决于其他的状态，这点和动态规划（DP）是一样的。 换句话说，就像我们在前面的章节所提到的，**蒙特卡洛（MC）方法不使用 提升（bootstrap）** 。  
> 2. 特别地，注意到**我们估计每一个特定状态的价值所需要花费的计算开销都是独立于状态数量的**。 所以**当我们只需要一个或者一小部分状态信息时，蒙特卡洛（MC）方法就很有吸引力了**。 我们可以从我们关心的那个状态开始，生成很多回合的样本，然后求它们的回报的均值，而不用管其他的起始状态。 这是蒙特卡洛（MC）方法相对说DP方法的好处（继可以从真实经验和模拟经验中学习之后的第三个好处）。  
## 5.2 Monte Carlo Estimation of Action Values(动作价值的蒙特卡洛估计)  
**有 model 时**，state value就足够可以来决定一个 policy 了；只要向前看一步，选择导致最好的 reward 组合和下一 state 的 action 即可，就像前一章讲 DP 时做的那样.  
**没有 model 时**，只有 state-value function 是不够的。必须要确切地估计出每个 action 的 value，才能让这些 value 能够得出一个 policy。因此，我们一个首要的目标是估计 ![10](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/10.svg)   .  
**如果不能获取到模型， 则估计 action 的 value（state-action 对的 value）比估计 state 的 value 要有用**。  
因此，我们一个首要的目标是估计![13](/home/tenglong/0.png)   
  
**对于 action value 的 policy evaluation 问题**:   
也就是估计 ![11](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/11.svg) ，即估计从state s 开始，采取 action a，遵循 policy![12](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/12.svg) 下的期望的 return。  
Monte Carlo 方法对 action value 进行 policy evaluation与之前展示的对于 state value 进行policy evaluation 是一样的，除了我们现在讨论的是对于 state-action 对的 visits，而不是 state。如果 state s 被 visit，然后执行action a ，则称为这个 state-action 对 s,a 被 visit。  
`every-visit 的 MC 方法`对于所有 visit 后的 return 取均值来估计 state-action 对的 value。`first-visit 的 MC 方法`对在每个 episode 中第一次出现这个 state 并且选择这个 action之后的 return 取均值。这些方法跟之前一样以指数收敛，当对于每个 state-action 对 visit 无穷次后会收敛到真实的期望 value.   
  
**唯一的困难就是很多 state-action pair一次都不会 visit 到**。  
如果![12](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/12.svg) 是个 deterministic policy，那么遵循![12](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/12.svg) 的情况下每个 state 只能观察到（observe）一个 action 的 return。如果没有 return 值来取平均，则 MC 方法其他的 action 的评估并不能随着 experience 提升。   
这是一个很严肃的问题因为学习 action 的 value 是为了帮助在每个 state 时在所有可选择的 action 中进行选择。我们需要估计在每个 state 下所有 action 的 value，而不只是当前看中的那个。   
  
**exploring starts**:  
这是一个关于 maintaining exploration 的问题，就像第二章讨论的多臂老虎机问题一样。对于 policy evaluation 来说，持续的 exploration 是必须的。**一个常见的办法是在 episode 的开始时，每个 state-action pair 被选到的概率都不为 0。这就保证了在无限个 episodes 中，每个 state-action pair 都会被 visit 无数次。这种方法叫 exploring starts。**   
  
exploring starts 的假设有时是有用的，但是并不是一直很可靠，尤其是从与 environment 的直接 interaction 中学习的情况。**最常见的用以保证所有 state-action pair 都被遇到的替代方法是只考虑那些在每个 state 下所有 action 的选择概率都不为 0 的随机 policy**。我们在之后的部分讨论这种思路的两种方法。    
## 5.3 Monte Carlo Control (蒙特卡洛控制)  
我们现在考虑 Monte Carlo 估计如何应用到 control 问题中，即近似 optimal policy。  
总体思想跟DP那一章差不多，就是 GPI 的思想。在GPI 中需要有近似的 policy 和近似的 value function。如下图所示，value function 不停迭代近似当前 policy 下的 value function。 policy 也基于当前的 value function 不停提升。两种变化某种程度上可以说是对立的，一种为另一种创建一个目标，但是它们一起迭代得到最优。   
![13](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/13.jpg)    
  
  开始时，我们考虑经典 policy iteration 的Monte Carlo 版本。在这种方法中， 我们从一个任意的policy ![14](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/14.svg) 开始交替完成 policy evaluation 和 policy improvement，直到最终得到最优的 policy 和最优的 action-value function：  
  ![15](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/15.svg)   
  ![16](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/16.png)   
  
**policy improvement 是通过基于当前的 value function 采取 greedy 策略来得到的**。这种情况下我们得到的是 action-value function，因此并不需要 model 来帮助建立 greedy 的策略。（译者注，DP那一章计算的是 v ，要想知道每个action对应的 value 就需要有model 来辅助计算，这一章直接计算的是  q ，因此就不需要 model 了）。对于任意的 action-value function  q ，相应的 greedy policy 就是对于每个的 ![17](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/17.svg)  ，确定性地选择有最大 action-value 的 action：   
![18](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/18.svg)   
![19](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/19.png)   

**为了保证Monte Carlo 方法的收敛性，我们做出了两个不可能的假设**:  
一个是episodes 有着 exploring starts  
另一个是 policy evaluation 能够用无数个 episodes 完成。  
为了得到实际可行的算法，我们需要去除这些假设。我们将在本章后面讨论第一个假设。  

**首先:我们专注于讨论去除 policy evaluation 在无数个 episodes 上进行的假设**:  
这个假设很好去除。事实上，DP那一章也遇到了一样的问题，比如 iterative policy evaluation，也是渐进收敛到真实的 value function。在DP 和 Monte Carlo 中都有两种方法解决这一问题.  
**一种是设置一个误差 bound，经过足够的步骤后就能够保证每个 policy evaluation 的误差足够小**。这种方法在很多时候能够保证收敛到一定程度的对于真实 value function的近似，但是也有可能会在哪怕最小的问题中实际上需要太多的 episodes。   
**第二种避免需要无数个 episodes 的方法是我们放弃完整的 policy evaluation 过程，直接转向 policy improvement**. 每个 evaluation 步骤中，value function 都会向 ![20](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/20.svg) 逼近，但是我们并不需要去经过很多步变得足够近。我们用首先在 4.6 节介绍的 GPI 的思想。一个特殊情况就是 value iteration，就是在policy evaluation 中每两次 policy improvement 之间进行一次迭代。 in-place 版本的 value iteration 更为特殊，对单个的 state 交替进行 improvement 和 evaluation 操作。   
  
对于Monte Carlo 方法，很自然地基于 episode-by-episode 交替进行evaluation 和 improvement。在每个 episode 之后，观察到的 return 被用来 policy evaluation，然后 policy 可以在这个 episode 中 visit 过的 state 上进行提升。**完整的算法如图，这种算法叫 Monte Carlo ES，即 Monte Carlo with Exploring Starts**:  
![21](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/21.jpg)   
> 在 Monte Carlo ES 中，对于每个 state-action pair 的 return 进行累加求平均。**很容易看到 Monte Carlo ES 不能收敛到任何非最优的 policy。假设value function 会最终收敛到 非最优的 policy 的 value function，然后反过来会导致这个 policy的改变。只有在 policy 和 value function 都达到最优的时候才能达到稳定**。最终收敛到最优的不动点看上去是必然的，因为 action-value function的改变越来越小。但是还没有被证明。在我们来看，这是强化学习领域最基本的开放性理论问题。  
### Example 5.3：Solving Balckjack 
把 Monte Carlo ES 应用到 balckjack 非常直接。既然 episodes 都是可以仿真的，很容易用 exploring starts。  
庄家的牌，玩家牌的总和，玩家是否有usable 的 A，都是等概率随机的。  
初始的 policy 我们用前面 blackjack 例子中的 policy，只在20或者21 stick。  
初始的 action-value function 对于所有的 state-action pair 都设置为 0。  
![22](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/22.jpg)   
**图中展示了利用 Monte Carlo ES 方法找到的 optimal policy 和 state-value function。其中 state-value function 是利用 action-value function 计算的**。这个policy 与Thorp （1966）的基本策略一致，除了最左边有 usable 的A 情况下的凹凸部分，在 Thorp 的策略中没有体现。我们对于这种差异的原因也不确定，但是确信的是这里展示的确实是我们所描述的 blackjack 游戏的 optimal policy。  
## 5.4 Monte Carlo Control without Exploring Starts(不带有探索性初始化的蒙特卡洛控制)  
主要是为了解决第二个不可能的假设(即避免不可能的 exploring starts 的假设).  
**唯一保证所有的 actions 都被无限地选择的方法是让 agent 持续地选择这些 actions**。有两种方法可以保证，分别为 on-policy 方法和 off-policy 方法。  
**On-policy 方法**尝试 evaluate 或者 improve 那个做决策的 policy  
**off-policy 方法 **evaluate 或者 improve 的 policy 不同于产生数据的 policy。   
Monte Carlo ES 方法是一个 on-policy 方法的例子。  
**本节主要展示如何把一个 on-policy 的Monte Carlo control 方法设计为不需要 exploring starts 这一不现实的假设的**。off-policy在下一小节考虑。  
![23](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/23.png)   
on-policy 的 Monte Carlo control 的总体思想仍然跟 GPI 一样。在 Monte Carlo ES 中，我们用 first-visit MC 方法来估计当前 policy 下的 action-value function。不需要 exploring starts 的假设，但是我们不能简单通过基于当前的 value function 采取 greedy 策略来提升 policy，因为这会导致未来缺乏对 nongreedy actions 的 exploration。幸运的是，GPI 不需要 policy 一直是一个 greedy policy。  
![24](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/24.png)   
完整的算法如下:  
![25](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/25.jpg)   
**一个证明: 证明![27](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/27.png)**:   
![26](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/26.png)   
**一个证明: 证明![28](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/28.png) ![29](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/29.png)**   
![30](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/30.png)   
注:  
![31](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/31.png)   
## 5.5 通过重要性采样的离策略预测 Off-policy Prediction via Importance Sampling  
**所有的学习控制方法都会面临一个难题**：  
> 它们基于后来的 optimal behavior 来学习 action values，但是它们却需要 behave non-optimally 来 explore 所有的 actions （来找到最优的 actions）。   
  
**它们如何通过 behave 一个 exploratory policy 来学习一个最优的 policy**？  
> 前面小节的 **on-policy 其实是一个折中——其学习的不是最优 policy 的 value function，而是还能够进行 explore 的近似最优的 policy**。   
  
**off-policy learning方法**:   
> 一个更直观的方法是用两个 policies，一个是用来学习最优 policy 的，另一个更加 exploratory，并且用来产生 behavior。   
> 学习的那个 policy 称为 target policy，用来产生 behavior 的 policy 称为 behavior policy。   
> 这种方法称为 off-policy learning。   
  
**on-policy 和 off-policy 方法对比**:  
> 本书 on-policy 和 off-policy 方法我们都考虑。  
> on-policy 方法一般更简单，会优先考虑。  
> off-policy 方法需要一些额外的概念和记号，因为数据是来自于另一个 policy，off-policy 方法一般方差更大，收敛更慢。  
> 另一方面，off-policy 方法更强大更具一般性。其可以把on-policy 看作一种特殊情况，target 和 behavior policies 刚好一样。  
> off-policy learning 还可以看作是学习多步预测模型的关键所在。  

本节中，通过考虑 prediction（预测）问题来研究 off-policy 方法，其中 target 和 ![32](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/32.png)   

**覆盖（coverage） 假设**:  
![33](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/33.png)   
**importance sampling(重要性采样)方法**:  
 `importance sampling 方法`: 是利用一个分布的采样来估计另一个分布期望值的方法。几乎所有 off-policy 方法度采用了 importance sampling 的方法.    
 `importance sampling ratio`: 把 importance sampling 方法应用到 off-policy 学习中，根据在 target 和 behavior policy 下发生的 trajectories (trajectory就是state， action，reward的一连串序列![34](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/34.svg) ) 的相对概率来对 return 取权重。这种方法叫 importance sampling ratio。  
  `importance sampling ratio的计算方式:`  
  基于target 和 behavior policies 的 trajectory 的相对概率:  
![35](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/35.png)   
> 注意到上式中的trajectory的概率依赖于MDP的转移概率（常常是未知的），但是它们在分子和分母中都是相同的，能够被消掉。 即是说，**重要性采样率最终仅仅依赖于两个策略和序列，而与MDP无关**。   

`ordinary importance sampling`:  
![36](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/36.png)   
`weighted importance sampling`定义为:  
![37](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/37.svg)   
分母为 0 时，该式定义为 0。  
> 注: **上面两式中,和中的每个元素本身也是和：**
![64](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/64.jpg)   
  
**通俗的理解这两种形式的importance sampling的不同**:  
> 为了理解这两种形式的 importance sampling,考虑他们观测到**一个**return 后的估计值。  
> **weighted-average 估计中**，比例![38](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/38.svg) 对于**单次** return 来说分子分母同时消去了，所以估计值等于观察到的 return 值，而与比例无关（假设这个比例非 0）。假设这个 return 是唯一观测到的，则这个估计是合理的，但是其期望是![39](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_5/39.svg) 而不是![40](/home/tenglong/0.png) ，从统计上讲，这是有偏的。   相比之下:  
> **简单取平均的（5.4）期望**总是 ![40](/home/tenglong/0.png),这是无偏的），但是很容易走极端。假设比例是 10，表示观测到的 trajectory 是在 target policy 的可能性十倍于 behavior policy 的可能性。这种情况下 ordinary importance-sampling 估计会十倍于观测到的 return。也就是即使这个 episode 的 trajectory 被认为非常能代表 target policy，其估计也会与观测到的 return 相去甚远。   
  
**正式地，两种 importance sampling 的差异**:  
> 两种 importance sampling 的差异,**一般用其偏差（bias）和方差（variance）来描述**。  
> 1. **ordinary importance-sampling 是无偏的而 weighted importance-sampling 有偏的**（偏差渐进收敛到 0）。  
> 2. 另一方面，**ordinary importance-sampling 的方差一般来说是没有界的**，因为比例（ratio）的方差可以是无界的，而 **weighted importance-sampling 对于任意 return 其权重最大为1。事实上，假设 return 有界，即使 ratio 本身无穷，weighted importance-sampling 的方差也会收敛到 0**。  
> 3. 实际上，weighted importance-sampling 方差非常小，一般倾向于使用它。但是，我们也不会将 ordinary importance-sampling 遗弃，因为其能够更简单地扩展到利用 function approximation 的方法，这些会在本书第二部分讨论。   

  
**关于on-policy和off-policy的一些不同的简单归纳**(较为准确的表述最好参考原文):  
**在策略 on-policy**  
> 1. 行为策略和目标策略是统一策略  
2. 直接使用样本统计属性去估计总体  
3. 更简单，收敛性更好，但数据利用性差  
4. 限定了学习过程中的策略是随机性策略  
  
**离策略 off-policy**  
> 1. 行为策略和目标策略不是一个策略  
2. 一般行为策略 μ 采用随机性策略以获得探索，目标策略 π 选用确定性策略  
3. 需要结合重要性采样才能使用样本估计总体  
4. 方差更大，收敛性更差，数据利用性更好  
5. 行为策略需要比目标策略更具有探索性，即在每个状态下目标策略的可行动作是行为策略可行动作的子集：
### Example 5.4： off-policy Estimation of a Blackjack State Value  
我们将 ordinary 和 weighted importance-sampling 都用来从来自于 off-policy 的数据中估计一个 state 的 value。  
回顾一下 **Monte Carlo 方法的一个优势是**他们可以直接用于估计某一 state 的 value，而不需要对于其他 states 也进行估计。  
在这个例子中，我们要评估的state 是庄家两点，玩家点数和为13，有一个 usable A（也就是玩家有一张A一张2，或者三张A）。  
从此 state 开始，然后等概率随机选取 hit 或者 stick（behavior policy）。target policy 是只在和为 20 或者 21 时 stick，如 Example 5.1 提到的那样。在 target policy 下该 state 的 value近似为 -0.27726 （这是在 target policy 下产生一亿个 episodes 然后对 returns 取均值得到的）。  
两种 off-policy 方法都在采取随机 policy 产生 1000 次 off-policy episodes 后取得了很好的近似。为了保证实验的可靠性，我们跑了 100 次，每一次估计的初始值都为 0，并从 10000 个 episodes 中学习。图5.3展示了每种方法估计的误差随着 episodes 数变化的函数关系，从100次运行中取平均。两种方法误差都趋近于 0，但是 weighted importance-sampling 在开始时误差小得多。  
![41](/home/tenglong/0.png)   
### Example 5.5： Infinite Variance  
ordinary importance sampling 的估计一般有无穷的方差，所以当 return 也有着无穷的方差时，其收敛性不令人满意，缩放的回报都有无限的方差—当 trajectory 中包含有循环时，很容易在 off-policy 学习中发生。   
一个简单的例子时图5.4中那样。只有一个 nonterminal state s 和两个 actions，向左和向右(向左即图中的left,向右即图中的right)。  
**向右**会导致一定变为 termination，而**向左**则会使 0.9 的概率回到 s ，0.1的概率转为 termination。  
除了最后的转移的 reward 为 +1，其余均为 0。   
考虑 target policy 为一直向左，所有在此 policy 下产生的 episodes 会有一些转变回  s （有可能数目为 0）然后 termination， reward 和 return 都为 +1。  
因此 s 在这个 target policy 下，value 为 1（ ![42](/home/tenglong/0.png) ）。假设我们要从用 behavior policy（等概率选择向左或者向右）产生的 off-policy 数据中估计这个 value。  
![43](/home/tenglong/0.png)   
> 图5.4的下半部分展示了利用 ordinary importance sampling 的 first-visit MC 算法的 10次独立运行。即使由数百万的 episodes，估计值也不能收敛到正确的 value 1。相反地，weighted importance sampling 在以向左这个 action 结束的episode 后总是能给出准确的估计值 1.所有不是 1 的 return（即以向右做结尾）将会与 target policy 不一致，则其权重 ![44](/home/tenglong/0.png) 为 0，在（5.5）的分子或者分母上均不起作用。weighted importance sampling 算法产生与 target policy 一致的 return 的加权平均，这些都准确是 1。   
**证明这个例子中 importance-sampling-scaled returns 的方差是无限的**:  
> 所有随机变量的方差可以写为:  
> ![45](/home/tenglong/0.png)   
> 因此，如果均值是有限的，像这个例子中一样，当且仅当随机变量的平方的期望是无限时,方差是无限的。  
> ![46](/home/tenglong/0.png)   
> 为了计算这个期望，我们将其按照 episode 的长度和 termination 分开。  
> 对于以向右结束的 episode，其 importance sampling ratio 是 0，因为 target policy 不会采取向右这个 action。 这些 episodes 对于期望值没有影响，可以忽略。  
> 我们只需要考虑那些向左的 action 并且最终以向左结尾的 episodes，这些 episodes 的 return 都是 1，所以![47](/home/tenglong/0.png) 项可以忽略。  
为了得到平方的期望值，我们只需要考虑每个长度的 episode，将这个episode 发生的概率乘上其 importance-sampling ratio，然后加起来：得到:  
![48](/home/tenglong/0.png)   
## 5.6 增量实现 Incremental Implementation  
Monte Carlo prediction 方法可以用增量的方法处理，利用类似第二章介绍的方法（2.4节）。  
不同的是, 第二章我们是对 reward 取均值，在Monte Carlo 方法中对 return 取均值。其他方面都一样.  
第二章的方法可以用来处理 on-policy Monte Carlo 方法。  
对于 off-policy 方法，我们需要对 ordinary importance sampling 和 weighted importance sampling 分开考虑。  
> **在 ordinary importance sampling 中**，return 会乘上 importance sampling ratio  ，然后简单取均值。对于这些方法，我们可以再次利用第二章的增量实现方法，直接用放缩过的 return 值替代第二章之中的 reward 即可。但是不能直接用在那些用 weighted importance sampling 的 off-policy 方法。  
> **对于用 weighted importance sampling 的 off-policy 方法**:  
![49](/home/tenglong/0.png)   
![50](/home/tenglong/0.png)   
  
下图展示了 **Monte Carlo policy evaluation 的 incremental Implementation 算法**。  
![51](/home/tenglong/0.png)   
![52](/home/tenglong/0.png)   
## 5.7 Off-policy Monte Carlo Control (离策略蒙特卡洛控制)  
现在我们展示第二类本书中学习控制的方法：**off-policy 方法**。  
> on-policy 是估计一个 policy 的 value，同时也用这个来进行 control。  
>  off-policy 之中两者是分开的。用于产生 behavior 的是 behavior policy，可能与估计和提升的 target policy 无关。  
> 这种**分离的一个好处是**， target policy 可以是确定性的（deterministic）（e.g., greedy），而behavior policy 则可以一直对所有可能的 actions 采样。  

Off-policy Monte Carlo 控制（control）方法采用前两小节展示的技术。  
> 遵循 behavior policy 而学习提升 target policy。这种技术需要 behavior policy 在所有 target policy 可能采取的 actions 上面的概率都非 0（coverage）。为了探索所有的可能性，我们需要 behavior policy 是 soft 的（i.e.，soft 意味着所有 states 下的所有 actions 的选择概率都非 0）.  
  
  ![53](/home/tenglong/0.png)   
  ![55](/home/tenglong/0.png)   
  **一个隐含问题**:  
  > **这种方法只在 episode 结束时学习，如果 nongreedy actions 很常见，则学习会非常慢，尤其是对于那些出现在长 episode 早期的 states**。关于 off-policy Monte Carlo 方法中这个问题有多严重还没有足够的经验去验证。如果很严重，则最重要的解决方法是加上 temporal-difference learning，这种算法思想会在下一章介绍。作为替代，如果![54](/home/tenglong/0.png) ，则下一小节介绍的思想也很有用。  
## 5.8 *Discounting-aware Importance Sampling (折扣的重要性采样)  
**目前考虑的 off-policy 方法是**基于将 return 看作整体分配权重进行 importance-sampling 看作整体，而不是考虑 return 的内部结构，看作discounted rewards 的和。  
**这种思想的本质是**将 1-discounting 看作 termination 的概率，或者等价地，a degree of partial termination。  
我们现在简单介绍**利用结构信息来减少 off-policy 方差**的前沿思想。  
![56](/home/tenglong/0.png)   
![57](/home/tenglong/0.png)   
> 简单的理解, 可以把![58](/home/tenglong/0.png) 理解为 `此步` 不结束(termination)的概率, 把1-![58](/home/tenglong/0.png) 理解为 `此步` 结束的概率(度, degree)   
> flat partial returns 就是一种恒等变换,可以展开推导得到.  
  
![59](/home/tenglong/0.png)   
![60](/home/tenglong/0.png)   
相应的ordinary importance-sampling 定义为:  
![61](/home/tenglong/0.png)   
相应的weighted importance-sampling 定义为:  
![62](/home/tenglong/0.png)   
两式相比: 分子一样,分母变化.  
![63](/home/tenglong/0.png)   
## 5.9 *Per-reward Importance Sampling(*per-reward 重要性抽样)    
还有一种方法也考虑了 return 的内部架构，这种方法甚至在没有 discounting 情况下（ [公式] ），也能够减少方差。  
在（5.4）（5.5）中，分子上求和的每一项本身也是个求和：  
![64](/home/tenglong/0.png)   
off-policy 依赖于这些项的期望值。  
注意到（5.10）每一个子项都是一个随机 reward 与随机的 importance-sampling ratio 的乘积。例如第一项用（5.3）式可以写为  
![65](/home/tenglong/0.png)   
注意所有这些因式中，只有第一项和最后一项（reward）是相关的，其他的都是独立随机变量，期望值为 1：  
![66](/home/tenglong/0.png)   
由于独立随机变量乘积的期望等于它们期望的乘积，所有的这些比例值除了第一项其他都可以丢弃，因此可以变形为:  
![67](/home/tenglong/0.png)   
![68](/home/tenglong/0.png)   
有没有 weighted importance sampling 的 per-reward 版本？  
迄今为止我们所知道的已经提出的一些方法还不是很符合（它们不能在有无穷多的数据之后收敛到真实的 value）.  
##  5.10 Summary(总结)  
本章展示的**Monte Carlo 方法通过从 sample episodes 学习 value function 和 optimal policies**。  
**其相比较于 DP 的4个优势**:  
> 第一，它们能够直接从与 environment 的 interaction 中学习 optimal behavior，而不需要 environment 的模型。  
> 第二，它们能够利用 simulations 或者 sample models。在很多应用中，很容易仿真 episodes，即使很难建立DP 算法所需要的转移概率的明确模型。  
> 第三，用 Monte Carlo 方法专注于 states 的小的子集非常简单且高效。  
> 第四, Monte Carlo 方法在马尔可夫性不满足时受到的影响比较小。这是因为其不是基于successor states 的估计来更新当前值的。换句话说就是 Monte Carlo 方法不 bootstrap(不需要进行**`bootstrap`**，即不需要根据接下来的状态的value来更新当前状态的value)。  

 在设计 Monte Carlo control 方法时，我们遵循了 GPI 的总体模式。GPI 包括 policy evaluation 和 policy improvement 的交替过程。  
 Monte Carlo 方法`提供了一个替代的 policy evaluation 过程`。它们**简单地对于从 state 开始地 return 取均值，而不是用模型去算每个 state 的 value**。由于 state 的 value是return 的期望，这种均值是 value 的一个好的近似。  
 在 control 方法中，我们对近似的 action-value function 尤为感兴趣，因为其能够提升 policy 而不需要 environment 的模型。  
 **Monte Carlo 方法基于 episode-by-episode 混合了 policy evaluation 和 policy improvement，也能够用增量的方法实现**。   
**保持足够的 exploration 也是 Monte Carlo control 方法的一个问题**。  
> 只选择现在估计的最好的 action 时，exploration 不够。  
> 一种忽略这种问题的方法是假设 episode 从随机的 state-action pair 开始来包含所有的可能性。这种 `exploring starts`有时可以通过方针的 episodes 事先，但是在从实际 experience 中学习时则是不可行的。  
> 在 on-policy 方法中，agent 总是 explore，并且试着找到仍能 explore 的最好的 policy。  
> 在 off-policy 种，agent也 explore，但是学习一个与遵循的 policy 不相关的 deterministic optimal policy。  
  
**off-policy prediction 是从 behavior policy 产生的数据中，学习 target policy 的 value function**。  
> 这种学习方法**基于 importance sampling**，通过在两种 policy 下采取所观察到的 action 的概率来给 return 加上权重。  
> **Ordinary importance sampling** 简单对加权后的 return 取均值，而 **weighted importance sampling** 则使用加权平均。  
> **Ordinary importance sampling** 是无偏估计，但是方差更大，甚至可能是无限的. 而 **weighted importance sampling** 的方差有限，在实际应用中更被青睐。
> 尽管它们概念上十分简单，但是对于 prediction 和 control 问题的 off-policy Monte Carlo 方法仍然没有完全解决，仍是一个研究方向。  
  
  **Monte Carlo 方法区别于 DP 方法主要在两方面**。  
  > 第一，它们基于 sample experience，所以没有模型也可以学习。  
  > 第二，它们不 bootstrap。因为他们不基于别的 value 来更新。这两种区别并不紧密联系，可以被分开。  
  
  下一章，我们考虑像 Monte Carlo 方法一样从 experience 学习，但是也像 DP 一样 bootstrap。











[参考1](https://zhuanlan.zhihu.com/p/58870476)   
[参考2](https://zhuanlan.zhihu.com/p/53229050)   
[参考3](https://steemit.com/cn-stem/@hongtao/mc-21)   
[参考4](https://rl.qiwihui.com/zh_CN/latest/partI/chapter5/monte_carlo_methods.html)   
