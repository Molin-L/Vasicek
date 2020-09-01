import numpy as np

class Vasicek:
    def __init__(self, a=0.5, b=0.05, sigma=0.05, lamb=-1, r0=0.02, N=100, n_sim=100):
        '''
        theta: long term rate
        kappa: speed of convergence
        sigma: volatility

        d_rt = (k*(theta-r_t)-lamb*sigma)+sigma*d_WP
        '''
        self.theta = b
        self.kappa = a
        self.sigma = sigma
        self.lamb = lamb
        self.r0 = r0
        self.S = 1
        self.T = 0.5
        self.U = 2
        self.N = N
        self.n_sim = n_sim
        self.dt = self.T/self.N
        self.rate_P = np.full((1, self.n_sim), r0)
        self.rate_Q = np.full((1, self.n_sim), r0)
        self.qt = -1
        '''
        Initialize dW matrix
        '''
        self._initial_dW()

        # Calculate price under different measures
        self._cal_Tprice_Q()
        self._cal_Tprice_P()
        self._cal_Sprice_Q()
        self._cal_Sprice_P()

        self._cal_quant()
    

    def _initial_dW(self):
        '''
        Generate dW
        '''
        self.dW_Q = np.random.randn(self.N, self.n_sim)
        self.dW_P = self.lamb*self.dt+self.dW_Q
        self._generate_dr()
        return self.dW_Q, self.dW_P
        
    def _generate_dr(self):
        for i in range(self.N):
            self._forward_rate_Q()
            self._forward_rate_P()
    
    def _forward_rate_P(self):
        '''
        dr_t = (k*(theta-r_{t})-lambda*sigma)+sigma*dWP[t]
        '''
        dr = (self.kappa*(self.theta-self.rate_P[-1])-self.lamb*self.sigma)+self.sigma*self.dW_P[len(self.rate_P)-1,]
        rt = self.rate_P[-1]+dr
        self.rate_P = np.vstack((self.rate_P, rt))
        
    def _forward_rate_Q(self):
        '''
        Generate ideal dr from dWQ
        drt = (k*(theta-r_{t})+sigma*dWQ[t])
        '''
        dr = (self.kappa*(self.theta-self.rate_Q[-1]))+self.sigma*self.dW_Q[len(self.rate_Q)-1,]
        rt = self.rate_Q[-1]+dr
        self.rate_Q = np.vstack((self.rate_Q, rt))
    
    def _cal_Sprice_Q(self):
        '''
        Calculate the price of Sbond under Q measure.
        '''
        price_list = []
        for t in np.arange(0, self.T+self.dt, self.dt):
            price_t = self._A_S_Q(t)*np.exp(-self._B_S(t)*self.rate_Q[int(t/self.T*self.N)])
            price_list.append(price_t)
        self.Sprice_Q = np.array(price_list)
    
    def _cal_Sprice_P(self):
        '''
        Calculate the price of Sbond under P measure.
        '''
        price_list = []
        for t in np.arange(0, self.T+self.dt, self.dt):
            price_t = self._A_T_P(t)*np.exp(-self._B_S(t)*self.rate_P[int(t/self.T*self.N),])
            price_list.append(price_t)
        self.Sprice_P = np.array(price_list)
        
    def _cal_Tprice_Q(self):
        '''
        Calculate the price of Tbond under Q measure.
        '''
        
        price_list = []
        for t in np.arange(0, self.T+self.dt, self.dt):
            price_t = self._A_T_Q(t)*np.exp(-self._B_T(t)*self.rate_Q[int(t/self.T*self.N),])
            price_list.append(price_t)
        self.Tprice_Q = np.array(price_list)
        
    def _cal_Tprice_P(self):
        '''
        Calculate the price of Tbond under P measure.
        '''
        price_list = []
        for t in np.arange(0, self.T+self.dt, self.dt):
            price_t = self._A_T_P(t)*np.exp(-self._B_T(t)*self.rate_P[int(t/self.T*self.N)])
            price_list.append(price_t)
        self.Tprice_P = np.array(price_list)
    
    def cal_quant_delta(self):
        '''
        Calculate the quantity using derivative
        Delata hedging
        dPt/dPs
        '''
        self.quant_s_delta_Q = np.divide(np.gradient(self.Tprice_Q, axis=0), np.gradient(self.Sprice_Q, axis=0))
        self.quant_s_delta_P = np.divide(np.gradient(self.Tprice_P, axis=0), np.gradient(self.Sprice_P, axis=0))
    
    def cal_quant_gamma(self):
        '''
        Calculate gamma hedging quantity
        '''
        self.quant_u_gamma = np.divide((self.))