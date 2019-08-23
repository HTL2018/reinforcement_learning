# Chapter_2 	多臂赌博机 (multi-armed bandit) 问题   
在数学领域，**多臂赌博机问题**（multi-armed bandit problem），也称为**顺序资源分配问题**（sequential resource allocation problem）。  
区别强化学习与其他学习方法最重要的特征是**使用训练信息来评估**所采取的action, 而不是直接给正确的action来训练。评定性反馈（evaluative）完全根据采取的actiion，而指导性反馈（instructive）和采取的action是独立的。  
## 1. 问题描述——k臂赌博机（k-armed bandit）  
![1](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_1/1.jpg)   
赌博机有k个摇臂，玩家投一个游戏币以后可以按下任意一个摇臂, 每个摇臂以一定的概率吐出硬币, 作为奖赏。 但这个概率玩家并不知道。 玩家的目标是通过一定的策略获得最大化的累积奖赏。  
![0](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/0.png)   
那么探索好还是利用好呢？  
> 取决于：**估计的精确度、不确定性和剩余操作次数**。  
### 1.1 动作值的估计方法:   
![2](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/2.png)   
## 2. 探索---利用问题解决办法:  
### 2.1 贪心算法: 只选择当前认为最好的,不进行探索;    
### 2.2 \epsilon-greedy方法:  
![3](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/3.png)   
为了系统的评估贪心算法和 \epsilon-greedy算法的有效性。以十臂赌博机进行实验，随机运行2000次。在每个赌博机问题中，动作值（action value）服从期望为0，方差为1的高斯分布。当一个具体的学习方法应用于在时间步 t 选择动作问题上时，实际奖赏Rt43服从期望 ![43](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/43.svg) 方差1的正态分布，如下图所示。  
![5](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_1/5.jpg)   
![6](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/6.png)   
![7](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_1/7.jpg)   
![8](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_1/8.jpg)   
> **结论**：短期内，贪婪算法显然更占优势，但从长远来看，适当的探索对我们更有利。  
  
**优缺点:**   
![9](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/9.png)   
**优化点：**  
![10](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/10.png)   
#### 2.2.1增量实现:  
样本平均值计算的增量实现: 目前我们所讨论的动作值方法中，所有估计的动作值都是所观察到的样本平均值。现在我们把问题焦点转为怎样使用更有效的计算方法计算出这些平均值，特别是利用连续记忆和持续时间步长计算。   
![11](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/11.png)   
显然，通过上式我们可以计算并记录任意时刻的奖赏估计值。但长期运作下，这对于计算机内存以及计算性能要求很高，并不明智。上式可化解如下：  
![12](/home/tenglong/0.png)   
![13](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/13.png)   
即: 新估计←旧估计+步长[目标−旧估计]  
表达式 [目标−旧估计] 是估计中的误差。通过向“目标”迈出一步来减少它。 目标被假定为指示移动的理想方向，尽管它可能是嘈杂的。例如，在上述情况下，目标是第n个奖励。  
> 请注意，增量方法中使用的步长参数（StepSize）会从时间步长到时间步长变化。  
   
简单的赌博机算法:  
![14](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/14.png)   
#### 2.2.2跟踪不稳定问题Tracking a Nonstationary Problem:  
上述讨论的平均法适用于稳定的赌博机问题，即在赌博机问题中的奖赏分布不会随时间的变化而变化。但强化学习问题又通常是不稳定的，在这种情况下，最近的奖赏相比于前面的奖赏通常占有更重的比值。 解决这类问题的通常方法是固定步长参数,即变形为:  
![14](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/14.png)   
推导出步长因子与奖赏之间的权重关系，做如下变化：  
![16](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/16.png)   
> 第一个条件用于保证执行次数足够多，最终克服任何初始条件或者随机的起伏；  
> 第二个条件保证最终收敛所用的次数足够少。  
  
注意在样本平均方法时两个收敛条件都成立。但对于固定步长因子的情况下却不是同时满足两个收敛条件。后者不满足第二个收敛条件这表明了估计值不会完全收敛于真实值但会不断变化影响最近的奖赏。正如我们刚所提到的，这种做法在不稳定的环境中是可信赖的，并且此类问题在强化学习中不稳定问题是很常见。如果想要使得固定步长因子的方法满足以上两个收敛条件,往往会收敛得比较慢或者需要适当的调整以达到一个满意的收敛率,因此只在理论上讨论,实际应用中很少关心.  
#### 2.2.3 优化初始值 Optimistic Initial Values
目前我们讨论过的所有方法都在某种程度上依赖于初始动作值的估计。在统计学中，这些方法在初值是估计的情况下结果是有偏差的。在样本平均方法中，只有在所有的动作都至少选择了一次之后这种偏差才会消失。但在使用到固定步长的方法中，虽然这种偏差在时间步长增加的过程中会减弱，但这种偏差永恒存在。在实践中，这种偏差通常不是一个值得考虑的问题甚至在某些时候这种偏差会很有帮助。坏处在于初始参数必须是由用户提供的一系列值集合，不然只能都初始化为0，而好处是它以一种简单的方式提供了一些奖赏可期望程度的先验知识。  
> encourage exploration: 初始值的设定：一般情况下设置为0，还可以设置为一个比较乐观的值如5，这样会鼓励探索。如下图:  
> ![17](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/17.png)   
### 2.3 置信上界动作选择 Upper-Confidence-Bound Action Selection  
我们总是需要探索，因为动作值估计准确度存在不确定性。贪心动作在目前的情况下看起来是最好的，但或许实际上一些其他的动作会更好。 \epsilon-greedy选择面临着非贪心动作的选取，但这种选取很弱智，它只会等概率的选择所有动作，而不会根据动作的潜力值（成为最优动作的潜力）来选取。一种有效的方式便是通过动作的潜力来选择动作：  
![18](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/18.png)   
> 置信上界选择核心思想是平方根是衡量动作值估计的准确度或方差。   
   
置信上界选择核心思想是:**平方根是衡量动作值估计的准确度或方差。**因此，数量上最大化可能是动作 a 值的真实值，而 c 决定了置信指数。  
一方面，每次选择 a 动作都会使得不确定性减少：![21](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/21.svg) 增加，分母增大，总函数值减少。  
另一方面，每次选择 a 以外的动作，时间 t 增大，![21](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/21.svg)不会增大，不确定性则增大。  
**使用自然对数的原因是为了随时间的增加让增长率变小一点，但不会有上界。**所有的动作都会被选择到，但是针对估计值低的动作，或者是被选择选择的动作，会降低其选择频率。  
![20](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/20.png)   
### 2.4 梯度赌博算法 Gradient Bandit Algorithms
目前我们所讨论的方法都是估计动作值并且用这些估计的数值来选择动作。这确实是一种好的方法，但并不是最佳的方法。现在我们考虑学习对每个动作 a 的数值偏好，用 ![22](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/22.svg) 表示，![22](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/22.svg) 值越大，动作被选择的机会越大。但这种偏好在奖赏方面并不会有什么影响，不同动作之间只有相对偏好才重要。如果我们把全部动作偏好都增加1000，那么对于选择动作的概率并没有什么影响，它是由Softmax分布决定：  
![23](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/23.svg)   
注: ![24](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/24.svg) 
基于随机梯度上升的思想，有一个自然学习算法，在每一步中，选择动作之后会收到一个奖赏，通过如下函数更新动作偏好：(这是定义的等式)  
![25](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/25.svg)   
其中:  
![26](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/26.png)   
> 十臂赌博机上运行梯度算法的结果图，如果基准线被省略那么性能将显着降低(图中 q∗(a) 被选择为接近+4而不是接近零时)  
> ![27](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/27.png)   
#### 2.4.1 梯度上升算法  
将梯度赌博算法理解为梯度上升的随机近似（stochastic approximation），每个动作偏好![22](/home/tenglong/0.png) 将随着对性能的影响成比例增加：  
![26](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/26.svg)   
其中:  
![27](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/27.svg)   
注:![28](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/28.svg)   
增量效应（increment’s effect）的度量是该性能相对于动作偏好的偏导数：  
![29](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/29.svg)   
加入 ![30](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/30.svg)  后上面等式依然能够成立是因为： ![31](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/31.svg) ，即梯度在所有动作上总和为零。虽然 ![32](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/32.svg) 在改变，引起了一些动作的概率上升而一些动作的概率下降，但是变化的总和必须为零，因为概率之和总是1。  
接下来，我们将每项乘以![33](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/33.svg):   
![34](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/34.svg)   
现在采用期望的形式，对随机变量 At（在时间步 t 时选择的动作）的所有可能值 x 求和，然后乘以取这些值的概率。从而得到：  
![35](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/35.svg)   
![36](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/36.png)   
此时![37](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/37.png) :  
![38](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/38.png)   
## 3. 关联搜索:上下文赌博机 Contextual Bandits(通俗理解,深入理解最好参照原文)  
  ![39](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/39.jpg)   
  **注意:至此,在上文讨论的多臂赌博机问题中，我们可以认为只有一个赌博机。**
agent可能的动作就是拉动赌博机中一个机臂，通过这种方式以不同的频率得到+1或者-1的奖励。在这个问题中，agent会永远选择同一个机械臂，该臂带来的回报最多。因此，我们设计的agent完全忽略环境状态，环境状态不会影响我们采取的动作和回报，所以对于所有的动作来说只有一种给定的状态。  
上下文赌博机问题中带来了**状态**的概念。状态包含agent能够利用的一系列环境的描述和信息。在这个例子中，有多个赌博机而不是一个赌博机，状态可以看做我们正在操作哪个赌博机。我们的目标不仅仅是学习单一赌博机的操作方法，而是很多赌博机。在每一个赌博机中，转动每一个机臂带来的回报都会不一样，我们的agent需要学习到在不同状态下（赌博机）执行动作所带来的回报。  
## 4.总结  
介绍了几种简单的平衡探索和利用的方法:  
![40](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/40.svg) 方法以 ![41](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/41.svg) 概率随机选择动作;  
UCB方法选择确定动作, 但同时UCB又巧妙地设计了在每一步倾向于选择那些搜索到次数更少的样本动作进行探索。  
梯度赌博算法估计的不是动作值，而是动作偏好，更加倾向于选择`更偏好`的动作(偏好本身对奖励没有影响,偏好本身没有意义,相对偏好才有意义)，而其分布概率方法则采用了Softmax概率分布方法。  
乐观初始值方法比贪心算法更加显著的进行探索。  
但很多人会产生一个疑问：这些方法中那个方法最好。我们在此通过parameter study图来给出解答：  
![42](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_2/42.png)   
在方法的选择中我们不仅应该评估其在最优参数下的表现，还要看其对参数的敏感度。所有的这些算法对参数都不敏感，参数在一个数量级内变化时这些算法表现都很好，但**总的来说，UCB表现最佳**.  
[主要参考](https://zhuanlan.zhihu.com/p/51680852) 
