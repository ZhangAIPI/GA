import numpy as np
import matplotlib.pyplot as plt
"""
遗传算法总体思想：
一. 产生一个初始种群
二. 根据问题的目标函数构造适值函数
三. 根据适应值的好坏不断的选择和繁殖
四. 若干代后得到的适应值最好的个体即为最优解

构成要素：
①种群和种群大小：
种群由染色体构成，每个个体也就是一个染色体，也同时对应着问题的一个解。
种群中个体的数量称为种群大小或种群规模，记作NP,一般情况下，种群规模通常设置为一个常量，这个值也一般越大越好，但一味地增大也会增大计算机运算的负担，一般设置为100~1000。有的时候根据实际情况的需要也会将种群规模设置为与遗传代数相关的变量，来获得更好的优化效果。
②编码方法：
编码方法也称为基因表达方法。
每个染色体可以表示为X=(x_1,x_2,x_3,……,x_n)，染色体的每一位都是一个基因，每一位的取值成为位值，n称为染色体的长度，原始GA采用固定长度的0.1字符串来表示一个染色体，例如X=(0110010)，这种方法称为0-1编码或者是二进制编码。
二进制编码适用于以下三种情况：背包问题，实优化问题(但当精度要求高时，编码就会很长，就不会采用这种编码方法，而是直接采用实数编码)，指派问题(一类特殊的线性规划问题。其中工作对资源的需求是一对一的，每样资源(雇员，机器，时间段)唯一的指派给一件工作(任务，位置，事件)。同时，资源i指派给工件j也会产生一个相应费用C_ij，问题的目标是如何指派可使总费用达到最小)。
此外，仍有
顺序编码：用1到n的自然数来编码，此种编码不允许重复，又称为自然数编码，例如n=7的染色体X=(2 3 1 5 4 6 7)，此类编码可以用于解决旅行商问题，指派问题等，i.e.上面7个序号可以理解为7个城市的游览路线
实数编码：染色体上的位置为实数，这种编码方式具有精度高，便于大空间搜索，运算简单的特定，特别适合实优化问题但是反映不出基因的特征。
整数编码：对于染色体X=(x_1,x_2,x_3,……,x_n),1<=x_i<=n_i,n_i为基因的最大取值。整数编码适用于新产品投入，时间优化，伙伴挑选的问题(每个项目都有挑选伙伴的数量的上限)。
③遗传算子：
交叉：单切点交叉，双切点交叉。单切点交叉随机选取一个切点，对于两个亲本，将该切点后的基因进行交叉互换。双切点则是选两个切点，将两个切点之间的基因进行交叉互换。这里会涉及一个交叉概率Pc。在遗传运算中，我们会通过选择策略进行选优进入遗传池，但并不是所有进入遗传池的个体都会发生交叉，于是引入了交叉概率Pc。Pc定义为各代交叉产生的后代数与种群个体数的比。(p.s.显然，较高的交叉率将达到更大的解空间，从而减少停留在非最优解的机会，但交叉率太高则会因过多搜索不必要的解空间而耗费大量的计算时间。一般设置为0.9。)
变异：染色体上单个基因发生变异，这里涉及一个变异概率Pm，一般设定为一个比较小的数，在5%以下。
这里就会涉及一个问题，如果在发生交叉变异后，染色体不合法了怎么办？
当然对于不合法状态有两种应对策略：拒绝或修复。拒绝策略需要保证不合法子代占比很小，修复策略则会导致父代基因丢失。
#顺序编码的合法性修复：
    交叉修复策略：
        ①部分映射交叉（PMX）：
            a:选切点X,Y
            b：交换中间部分
            c：确定映射关系
            d：将未交换部分按照映射关系恢复合法性
            #################################
            #         X       Y             #
            # P_1:2 1 | 3 4 5 | 6 7         #
            # P_2:4 3 | 1 2 5 | 7 6         #
            #-------------------------------#
            #映射关系：3-1 4—2 5-5            #
            #交叉互换：                       #
            # P_1:2 1 | 1 2 5 | 6 7         #
            # P_2:4 3 | 3 4 5 | 7 6         #
            #不满足顺序编码的不重复性           #
            #利用交换子串的映射关系进行修复      #
            # P_1:4 3 | 3 4 5 | 6 7         #
            # P_2:2 1 | 1 2 5 | 7 6         #
            #################################
            这样使得重复的位置被换掉，而交换的子串保持不变
        ②顺序交叉（OX）：
            可以看做是PMX的变式
            a:选择切点X，Y
            b：交换中间部分
            c：从第二个切点Y后第一个基因起列出（###）原顺序（###），去掉已有基因
            d：从第二个切点Y后第一个位置起，将获得的无重复顺序填入
            该方法较好的保留了相邻关系，先后关系，满足了TSP问题的需要，但是不保留位值特征。
        ③循环交叉（CX）：书上也不明白，用链接：https://www.cnblogs.com/gambler/p/9124862.html
    变异修复策略：
        ①换位变异：随机选取两个位置交换基因的位值
        ②移位变异：随意选择一位基因移位到最前面
#实数编码的合法性修复：
    交叉：
        ①单切点交叉
        ②双切点交叉
        ③凸组合交叉：简单的交叉操作（单切点交叉和双切点交叉）很容易造成解的不可行性
                    所以采用凸集合理论：
                        Z_1=a*X+(1-a)*Y
                        z_2=(1-a)*X+a*Y
                    若约束是个凸集，虽然可行性得到了保持，但是这样的操作将导致种群的分散性不好，基因的取值向中间汇集，染色体覆盖的区域越来越小。
    变异：
        ①位值变异：x=x+z，z为扰动
        ②向梯度方向变异：Z=X+/-grad(f(X))*a 我理解为降低损失函数
④选择策略：
    最常用的选择策略是正比选择策略，轮盘赌法。
    此外有，顺序选择。
⑤停止准则：一般是达到迭代次数，或者看是不是收敛（根据平均适应值是不是和max差不多）

#想了很久的一个地方是，轮盘赌选择好与父代一样数目的个体后，再进行交叉运算和变异运算，假设交叉概率Pc=0.6，则为每个个体产生一个0-1的随机数，所有小于0.6的个体与随机一个个体进行交配，交叉点随机确定。
"""
class person:
    def __init__(self,length=5,func=lambda x:x**3-60*x**2+900*x+100):
        self.code=np.array(np.random.normal(size=length)).astype(np.int32)
        self._code()
        self.func=func
        self.decode=.0
        self._decode()
        self.fitness=func(self.decode)

    def __call__(self):
        pass

    def _code(self):
        """
        编码
        """
        for i in range(len(self.code)):
            self.code[i]=np.random.choice([0,1],1,p=[0.5,0.5])

    def _decode(self):
        """
        解码
        """
        self.decode=.0
        for i in range(len(self.code)):
            self.decode+=self.code[i]*(2**(len(self.code)-i-1))

    def __len__(self):
        return len(self.code)

    def __lt__(self,other):
        #<
        if self.fitness<other.fitness:
            return 1
        return 0
    def __le__(self,other):
        #<=
        if self.fitness<=other.fitness:
            return 1
        return 0

    def __gt__(self,other):
        #>
        if self.fitness>other.fitness:
            return 1
        return 0
    def __ge__(self,other):
        #>=
        if self.fitness>=other.fitness:
            return 1
        return 0

    def mutated(self):
        """
        变异
        """
        index=np.random.choice(np.arange(len(self.code)),1)
        if self.code[index]==1:
            self.code[index]=0
        else:
            self.code[index]=1
        self.update()

    def copy(self):
        #深拷贝
        import copy
        return copy.deepcopy(self)

    def update(self):
        self._decode()
        self.fitness=self.func(self.decode)



class GA:
    def __init__(self,NP=800,Pc=0.6,Pm=0.02,length=5,func=lambda x:x**3-60*x**2+900*x+100,iters=2000,delta=1.0):
        self.village=[person(length,func) for i in range(NP)]
        self.func=func
        self.Pc=Pc
        self.Pm=Pm
        self.NP=NP
        self.mean_fitness=.0
        self._mean_fitness()
        self.max_fitness=.0
        self._max_fitness()
        self.circle_prob=.0
        self._circle_prob()
        self.x=.0
        self.length=length
        self.iters=iters
        self.delta=delta#收敛阈值
    def _mean_fitness(self):
        """
        求平均适应值
        """
        mean_fitness=.0
        for i in self.village:
            #print(i.fitness)
            mean_fitness+=i.fitness
        self.mean_fitness=mean_fitness/self.NP
    def _max_fitness(self):
        """
        求最大适应值
        """
        import copy
        for i in range(len(self.village)):
            #print(self.village[i].fitness,self.village[i].decode)
            if self.village[i].fitness>self.max_fitness:
                self.max_fitness=self.village[i].fitness
        for i in range(len(self.village)):
            if self.max_fitness==self.village[i].fitness:
                self.x=self.village[i].decode
    def _circle_prob(self):
        """
        轮盘赌概率
        """
        fitness=np.array([i.fitness for i in self.village])
        self.circle_prob=fitness/(self.NP*self.mean_fitness)
        self.circle_prob=self.circle_prob.cumsum(axis=0)#在axis=0上累加

    def update(self):
        """
        更新状态值
        """
        res=self.NP-len(self.village)
        if res:
            res=np.array([person() for i in range(res)])
            self.village=np.append(res,self.village)
        self._mean_fitness()
        self._max_fitness()
        self._circle_prob()
    def choose(self):
        """选择下一代个体"""
        self.update()
        new_generation=np.array([])
        prob=np.random.rand(self.NP)
        #print('prob len.{}'.format(len(prob)))
        prob.sort()
        for i in range(len(prob)):
            for j in range(len(self.circle_prob)):
                if prob[i]<=self.circle_prob[j]:
                    new_generation=np.append(new_generation,self.village[j])
                    break
                continue
        #print('choose/new_generation.len.{}'.format(len(new_generation)))
        self.village=new_generation
        self.update()
    def cross(self):
        """
        交配
        """
        self.update()
        N=int(self.Pc*self.NP)#交配个体数
        par=np.random.choice(np.arange(self.NP),replace=False,size=N)
        remain=np.array(list(set(np.arange(self.NP))-set(par)))
        par=self.village[par]
        remain=self.village[remain]
        np.random.shuffle(par)
        self.update()
        for i in np.arange(start=0,stop=N,step=2):
            if i+1<N:
                self.cross_bianary_gene(par[i],par[i+1])
        new_generation=np.append(remain,par)
        if len(new_generation)<self.NP:#如果因为精度损失造成的种群数量减少，则再随机加入新个体
            add=self.NP-len(new_generation)
            add=np.array([person(self.length,self.func) for i in range(add)])
            new_generation=np.append(new_generation,add)
        self.village=new_generation
        #print('cross/generation len.{}'.format(len(new_generation)))
        self.update()
    def cross_bianary_gene(self,Xiao_Ming,Xiao_Hong):
        index=np.random.choice(np.arange(self.length),replace=False,size=2)
        index.sort()
        #print(index)
        start,end=index
        temp=Xiao_Ming.code[start:end+1]
        Xiao_Ming.code[start:end+1]=Xiao_Hong.code[start:end+1]
        Xiao_Hong.code[start:end+1]=temp
        Xiao_Ming.update()
        #print(Xiao_Ming.code)
        Xiao_Hong.update()
        #print(Xiao_Hong.code)
        self.update()
    def mutated(self):
        mu=np.random.choice(np.arange(self.NP),replace=False,size=int(self.NP*self.Pm))
        remain=np.array(list(set(np.arange(self.NP))-set(mu)))
        mu=self.village[mu]
        remain=self.village[remain]
        for i in mu:
            i.mutated()
            i.update()
        self.village=np.append(mu,remain)
        self.update()

    def grow(self):
        diff=self.max_fitness-self.mean_fitness
        plt.axis([0,20,0,5000])#plt.axis([xmin,xmax,ymin,ymax])
        plt.ion()
        for iter in range(self.iters):
            if diff<=self.delta:
                break
            self.choose()
            self.cross()
            self.mutated()
            x=np.linspace(0,30,300)
            y=self.func(x)
            plt.clf()
            plt.plot(x,y,color='red',linewidth=1)
            x=np.array([self.village[i].decode for i in range(self.NP)])
            #x=x[np.where(x<=30)]
            y=self.func(x)
            plt.scatter(x,y,c='k')
            plt.draw()
            plt.pause(0.1)
            print('iter:{} while mean_fitness.{}'.format(iter,self.mean_fitness))
            diff=self.max_fitness-self.mean_fitness
        plt.ioff()



if __name__=='__main__':
    #a=person()
    #b=person()
    ga=GA()
    #print(ga.mean_fitness)
    #ga.choose()
    #print(ga.mean_fitness)
    #ga.cross()
    #print(ga.mean_fitness)
    #ga.mutated()
    #print(ga.mean_fitness)
    ga.grow()
    print('当前函数最优解x={} y={}'.format(ga.x,ga.max_fitness))