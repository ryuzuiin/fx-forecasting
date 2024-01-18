import numpy as np
import pandas as pd

class Node:

    def __init__(self, x, y, idxs, min_leaf = 5, depth = 10):
        #定义实例属性
        #self 类的当前实例，引用对象本身

        self.x, self.y = x, y
        self.idxs = idxs
        self.min_leaf = min_leaf
        self.row_count = len(idxs)
        self.col_count = x.shpae[1]
        self.avl = self.compute_gamma(y[self.idxs])

        self.score = float('inf')
        self.find_varsplit()


        '''
        扫描每一行每一列来计算最好的分裂点
        在这一点上分开（作为root）然后创造出两个新的点
        当我们想要增加树的深度时，深度是唯一可以改变的参数
        当没有分裂能比第一次分裂的分数更好时，分裂停止
        '''

        def find_varsplit(self):

            for c in range(self.col_count): self.find_greedy_split(c)
            if self.is_leaf : return
            x = self.split_col
            lhs = np.nonzero(x <= self.split)[0]
            rhs = np.nonzero(x > self.split)[0]
            """
            图或者树类结构经常使用的结构，实例中套实列，表特殊的数据结构
            """
            self.lhs = Node(self.x, self.y, self.idex[lhs], self.min_leaf, depth = self.depth-1)
            self.rhs = Node(self.x, self.y, self.idxs[rhs], self.min_leaf, depth = self.depth-1)

        def find_greedy_split(self, var_idx):
            """
            稷酸梅个可能的分割点带来的增益（信息增益，基尼不纯度减少量等等）
            在执行上述过程中找到了一个比当前已知更好的分割点,该函数或方法会更新一个全局的最佳分数
            """

            #构建了一个np二维数组，idxs也是数组或列表
            x = self.x.values[self.idxs, var_idx]

            #这里的self已经换了node了，是新的node，新的row_count = len(x)
            for r in range(self.row_count):
                lhs = x <= x[r]
                rhs = x > x[r]

                lhs_indices = np.nonzero(x <= x[r])[0]
                rhs_indices = np.nonzero(x > x[r])[0]
                if (rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf) : continue

                curr_score = self.gain(lhs, rhs)
                if curr_score > self.score:
                    self.var_idx = var_idx
                    self.score = curr_score
                    self.split = x[r]

        
        def gain(self, lhs, rhs):

            gradient = self.y[self.idex]

            lhs_gradient = gradient[lhs].sum()
            lhs_n_intances = len(gradient[lhs])

            rhs_gradient = gradient[rhs].sum()
            rhs_n_intances = len(gradient[rhs])

            gain = ((lhs_gradient**2/(lhs_n_intances)) + (rhs_gradient**2/(rhs_n_intances))
                    - ((lhs_gradient + rhs_gradient)**2/(lhs_n_intances + rhs_n_intances)))
            return(gain)
        
        '''
        静态方法装饰器，不接收隐式的第一个参数，因此它不能访问类或实例的属性，它是一个与类紧密相关，但不依赖于类实例的函数
        当你有一些与类相关的功能，但这些功能不需要访问或修改类的状态时，可以使用静态方法
        '''
        @staticmethod
        def compute_gamma(gradient):
            '''
            创建GBM分类器，最佳叶节点的值
            '''
            return(np.sum(gradient)/len(gradient))
        
        @property
        def split_col(self):
            return self.x.values[self.idexs, self.var_idx]
        
        @property
        def is_leaf(self):
            return self.score == float('inf') or self.depth <= 0
        
        def predict(self, x):
            return np.array([self.predict_row(xi) for xi in x])
        
        def predict_row(self, xi):
            node = self.lhs if xi[self.var_idx]<= self.split else self.rhs
            return node.predict_row(xi)
        

class DecisionTreeRegressor:
    '''
    为了与scikitlearn相兼容，一个类通常需要实现如’fit‘，’predict‘等方法，这样它就可以
    与scilearn的其他功能无缝集成。
    '''
    def fit(self, X, y, min_leaf = 5, depth = 5):
        self.dtree = Node(X, y, np.array(np.arange(len(y))), min_leaf, depth)
        return self
            
    def predict(self, X):
        return self.dtree.predict(X.values)

class GradientBoostingClassification:
    '''
    binary logistic loss function
    ''' 

    def __init__(self):
        self.estimators = []

    @staticmethod
    def sigmoid(x):
            return 1 /(1 + np.exp(-x))
            
    def negativeDerivitiveLogloss(self, y, log_odds):
        '''
        计算逻辑损失函数的负导数
        '''
        p = self.sigmoid(log_odds)
        return(y-p)
            
    @staticmethod
    def log_odds(column):

        if isinstance(column, pd.Series):
            binary_yes = np.count_nonzero(column.values == 1)
            binary_no == np.count_nonzero(column.values == 0)
        elif isinstance(column, list):
            column = np.array(column)
            binary_yes = np.count_nonzero(column == 1)
            binary_no = np.count_nonzero(column == 0)
        else:
            binary_yes = np.count_nonzero(column == 1)
            binary_no = np.count_nonzero(column == 0)
            
            value = np.log(binary_yes/binary_no)
            return(np.full((len(column), 1), value).flatten())
            
    def fit(self, X, y, depth = 5, min_leaf = 5, learning_rate = 0.1, boosting_rounds = 5):

        # use the log odds value of the target variable as our inital prediction
        self.learning_rate = learning_rate
        self.base_pred = self.log_odds(y)

        for booster in range(boosting_rounds):
            
            pseudo_residuals = self.negativeDerivitiveLogloss(y, self.base_pred)
            boosting_tree = DecisionTreeRegressor().fit(X = X, y = pseudo_residuals, depth=5, min_leaf=5)
            self.base_pred +=self.learning_rate * boosting_tree.predict(X)
            self.estimators.append(boosting_tree)

    def predict(self, X):

        pred = np.zeros(X.shape[0])
        for estimator in self.estimators:
            pred +=self.learning_rate*estimator.predict(X)

            return self.log_odds(y) + pred
        
class GradientBoostingRegressor:
    '''
    mean squared loss function to calculate the negative derivate
    '''

    def __init__(self, classification = False):
        self.estimators = []

    @staticmethod
    def MeanSquaredError(y, y_pred):
        return(np.mean((y-y_pred)**2))

    @staticmethod
    def negativeMeanSquaredErrorDerivitive(y, y_pred):
        return(2*(y-y_pred))
    
    def fit(self, X, y, depth = 5, min_leaf = 5, learning_rate = 0.1, boosting_rounds = 5):

        self.learning_rate = learning_rate
        self.base_pred = np.full((X.shape[0], 1),np.mean(y)).flatten()

        for booster in range(boosting_rounds):

            pseudo_residuals = self.negativeMeanSquaredErrorDerivitive(y, self.base_pred)
            boosting_tree = DecisionTreeRegressor().fit(X=x, y = pseudo_residuals, depth=5, min_leaf=5)
            self.base_pred += self.learning_rate*boosting_tree.predict(X)
            self.estimators.append(boosting_tree)

    def predict(self, X):

        pred = np.zeros(X.shape[0])
        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X)

        return np.full((X.shape[0], 1), np.mean(y)).flatten() + pred
                   

        









