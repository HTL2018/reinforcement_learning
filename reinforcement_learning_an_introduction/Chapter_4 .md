# Chapter_4:Dynamic Programming  
**动态规划（DP）指的是用来计算一类马尔可夫决策过程（MDP）的最优策略的算法集合**。这类MDP中关于environment的完美模型是给定的。   
经典的DP算法在强化学习问题中用处有限，因为其假设有完美的模型，而且需要很大的计算量。但是DP在理论上仍然很重要。  
DP为这本书剩余部分的方法提供了必要的基础。事实上这些方法可以看作以较小的计算代价且不假设有environment的完美模型，而努力达到与DP一样的效果。  
本章一开始，我们通常假设environment是有限MDP。  

**DP的核心思想，包括一般强化学习，是使用value function来搜索好的policy**。本章中我们展示了DP如何计算第三章中定义的value function。  
一旦有了最优的value function ，最优的policy就很容易得到。最优value function满足Bellman最优方程：  
![0](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_4/0.svg)   
或者  
![1](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_4/1.svg)   
DP算法就是将如上Bellman方程作为更新法则来获得value function的近似的。  
## 4.1 Policy Evaluation(Prediction)  
总结:  
![2](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_4/2.png)   
详细如下:  
首先我们考虑任一policy的state-value function。在DP中这叫做policy evaluation。也可以把它看作prediction problem。  
![3](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_4/3.png)   
iterative policy evaluation:  
![4](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_4/4.png)   
full backup:  
![5](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_4/5.png)   
sweep:  
![6](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_4/6.png)   
算法的termination:  
![7](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_4/7.png)   
例:  
![8](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_4/8.png)   
图： iterative policy evaluation在gridworld的收敛性。左边一列是随机policy下对于state-value function的近似的序列。右边一列是当前对于value function的估计下greedy policy的序列（箭头表示此action能获得最大value）。第三次迭代后就能达到最优的policy。  
## 4.2 Policy Improvement  
判断两个policy之间的优劣:  
![9](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_4/9.png)   
policy improvement theorem证明过程:  
![10](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_4/10.png)   
policy improvement:  
![11](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_4/11.png)   
依照原policy下得到的value function采取greedy策略可以提升原policy。这种过程叫policy improvement。Policy improvement一定会给我们一个严格的更好的policy，除非原policy就是最优的。  
注:(随机policy)  
![12](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_4/12.png)   
注:(如果最大值对应的action不唯一)  
另外，如果有一些action能同时达到最大值，在随机情况下，我们并不是只选其中的一个action。作为替代，每个对应最大value的action都被赋予一个被选择的正概率，可以是任意值，只要其他任意非最优的action的概率是0。  
## 4.3 Policy Iteration  
![13](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_4/13.png)   
策略迭代 (Policy Iteration) 的伪代码如下图:  
![14](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_4/14.png)   
例子:杰克租车:  
![15](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_4/15.png)   
## 4.4 Value Iteration  
value iteration 是通过把Bellman最优方程变为更新法则。value iteration backup和policy iteration backup是一样的，除了value iteration需要在所有action中最大化。   
policy iteration有个缺点就是每次迭代都要有一次policy evaluation，而这个本身可能就需要多轮在整个state集上的sweep的迭代计算。policy evaluation只能再极限处收敛。我们必须要等到确实收敛了？还是说提前终止？截短policy evaluation是可能的。  
有多种方法能截短policy iteration的policy evaluation步骤而不会影响policy iteration的收敛。**一个特殊的做法是当policy evaluation仅进行一次sweep就停止（每个state一次backup）。这种算法叫value iteration**。其可以用一个简单结合了policy improvement和截短policy evaluation的backup操作表示：  
![16](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_4/16.svg)   
另一种理解价值迭代的方法参考贝尔曼方程(4.1)。注意价值迭代仅仅是将贝尔曼最优方程转变为更新规则。   
value iteration算法:  
![17](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_4/17.jpg)   
value iteration在自己的一次sweep中结合了policy evaluation的sweep和policy improvement的一次sweep。  
一次policy improvement的sweep中插入多次policy evaluation的 sweep可以收敛的更快。  
赌徒问题:  
![18](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_4/18.png)   
![19](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_4/19.png)   
## 4.5 Asynchronous Dynamic Programming   
到目前为止我们所讨论的DP方法一个主要的缺点是他们涉及整个MDP状态集合，也就是说，需要对整个状态集合进行更新。 如果状态集非常大，即使一次更新(sweep)也会代价很大。  
Asynchronous DP算法都是in-place迭代DP算法，其不必要对于state集合进行系统地sweep。  
这些算法以任意顺序用其他任意可获得的state的value来back up 这些state的value。有一些state的value可能被back up若干次而其他state的value只被back up一次。为了正确收敛，**一个asynchronous算法必须继续back up所有state的value**：**不能忽略任意一个state**。Asynchronous DP算法在选择哪些state来进行backup操作上有一定灵活性。   
避免sweep并不一定就意味着能够有更少的计算量，这是能表示算法在能够提高policy之前不需要被一些没有希望的长时间的sweep束缚住。我们可以利用这种任意选择state来back up的自由性来提升算法的速率。我们也可以尝试将backup进行排序来使得value信息在state之间更有效地传播。一些state可能不需要像别的state一样频繁back up。我们甚至可能试着跳过一些state，如果它们对于最优的bahavior没有影响。   
Asynchronous算法也能更容易地让实时的interaction与计算混合起来。为了解决一个给定的MDP，我们可以运行一个迭代地DP算法，同时agent也在经历这个MDP。这个agent地经历可以用来决定哪些state用这个DP算法来backup。同时，来自于DP算法的最新的value和policy信息可以知道agent做决策。例如可以backup那些agent遇到的state，这让DP算法专注于那些与agent最相关的state成为可能。
## 4.6 Generalized Policy iteration  
policy iteration存在两个同时的相互作用的过程，一个使得value function与当前policy保持一致（policy evaluation），另一个使得policy关于当前的value function最优。（policy improvement）。   
在policy iteration中，这两个过程相互交替，一个完成了另一个再开始，但这其实是没有必要的。  
在value iteration中，例如两次policy improvement之间只进行了policy iteration的一次迭代。在asynchronous DP方法中，evaluation和improvement的过程是以相同频率交替进行的。   
只要两个过程一直更新所有的state，最终的结果都是一样的——收敛到最优的value function和最优的policy。   
我们用generalized policy iteration（GPI）来概括这种让policy evaluation和policy improvement交替进行的方法，而不考虑两个过程的间隔长度（每个执行多少次迭代这样）或者其他的细节。  
所有的都有可以识别的policies和value functions，其中policy是利用value function来提升的，而value function则是会向着提升后的policy的value function更新。  
Generalized policy iteration：policy和value function相互作用直到它们都是最优的并且相互一致。GPI的总体思想:  
![20](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_4/20.jpg)   
如果evaluation过程和improvement过程都稳定，那么不会再有步骤改变，然后value function和policy都肯定是最优的。  
value function只有与当前的policy一致时在稳定，policy只有在其是当前value function下的greedy时才稳定。因此，只有policy发现他是自身的value function的greedy的policy时，两个过程才都稳定。   
**GPI的evaluation和improvement过程可以看作是竞争和合作**的关系（competing和cooperating）。说他们是竞争关系因为它们互相向相反方向拉。**让policy参照value function 来greedy地采取action，使得value function对于改变过后的policy来说是不正确的。使value function与policy一致又使得policy变得不再greedy。****从长远来看，两者相互作用来寻找一个联合的解决方案：最优的value function和最优的policy。**  
还可以从两个约束或者目标的角度来看evaluation和improvement之间的相互作用——例如下图二维空间中的两条线：  
![21](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_4/21.jpg)   
## 4.7 Efficiency of Danamic Programming
DP对于非常大的问题可能并不奏效。但是相比较于其他解决MDP的方法，DP方法相当有效。  
如果我们忽略一些技术细节，DP方法寻找最优policy所耗费的时间是state和action数多项式关系。  
![22](https://github.com/HTL2018/reinforcement_learning/blob/master/reinforcement_learning_an_introduction/image/Chapter_4/22.png)   
从这个意义上讲，DP比直接在policy空间搜索要快得多（指数级地更快），因为直接搜索需要尽可能地检查每一个policy才能做到相同的效果  
**线性规划方法**也能被用来解决MDP问题。它们最坏的情况下，对于收敛性的保证也比DP方法好。但是线性规划方法在状态数稍微大一点的情况下就不现实了。   
**维度诅咒（curse of dimensionality）**，state数通常是随着表示state的变量数指数级增长的。大的state集合并不产生困难，但是这是问题固有的困难，而不是DP的问题。事实上，DP比起其竞争对手直接搜索和线性规划已经更为适合用来解决大的state集问题了。  
对于大state空间问题，asynchronous DP方法更被偏爱。哪怕只完整完成一次synchronous方法地sweep也需要对于每个state开一个内存，进行计算。对于一些问题，即使这么大的内存以及计算量是不现实的。Asynchronous方法和其他GPI的变体能够被用到这些情况中，并且找到好的或者最优的policy要比synchronous方法快得多。  
## 4.8 Summary
**policy evaluation**通常是指通过迭代来计算某个policy下的value function。  
**policy improvement**是指给定policy下的value function，来计算提高policy。  
将两种计算放在一起，我们得到**policy iteration和value iteration**两种方法，两种最流行的DP方法。  
**经典的DP算法，是在state集合上进行sweep，对于每个state进行一次full backup**。每个backup依照所有可能的successor state和他们发生的概率来更新一个state的value。完整的backup和Bellman方程的关系很近：**就是把方程变成了赋值语句**.   
事实上，**几乎所有的强化学习方法能够把它们进一步看作generalized policy iteration（GPI**）。GPI的总体思想是相互作用的在近似policy和近似value function之间切换的过程 。**一个过程**给定policy然后进行policy evaluation，改变value function让其更像这个policy真实的value function。**另一个过程**给定value function并且进行policy improvement，将policy变得更好。   
**总的来说它们共同协作来找寻联合的solution：一个policy和value function不随任何一个过程改变，最终成为最优的**。在一些情况下，GPI可以证明收敛，最典型的就是本章介绍的经典DP算法。在其他情况下，收敛性还没被证明，但是GPI的思想使我们对于这些方法的理解加深了。   
对于整个state集合进行完整的sweep是没有必要的。Asynchronous DP方法是in-place的迭代方法，其以任意的顺序对于state进行back up，也许是随机决定的或者利用一些过时的信息决定。   
**DP方法的最后一个特殊性质**:所有方法基于对于successor state的value的估计值更新对于state的value的估计值。也就是说，它们基于其他的估计来更新估计值。我们把这种思想称作 bootstrapping。  
很多强化学习方法都会用bootstrapping，哪怕是那些不需要DP算法所需要的完整而精确的对于environment的模型的算法。  
