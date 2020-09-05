import numpy as np
import pandas as pd

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
        self.initial_dW()
        
        # Calculate price under different measures
        self.Tprice, self.dT, self.gamma_T = self._cal_price(self.T)
        self.Sprice, self.dS, self.gamma_S = self._cal_price(self.S)
        self.Uprice, self.dU, self.gamma_U = self._cal_price(self.U)
        self.ZCB_price = self._cal_zcbprice()

        # Calculat delta T and delta S
        '''
        self.dT = self._dx(self.Tprice)
        self.dS = self._dx(self.Sprice)
        self.dU = self._dx(self.Uprice)
        
        # Calculate gamma T, S, U
        self.gamma_T = self._gamma_x(self.Tprice)
        self.gamma_S = self._gamma_x(self.Sprice)
        self.gamma_U = self._gamma_x(self.Uprice)
        '''

        self.cal_quant_delta()
        self.cal_quant_gamma()
    
        self.portfolio()

        self.state()

    def get_result(self):
        finalday_pf_1 = self.pf_s1[-1]
        var_s1 = np.percentile(finalday_pf_1, 95)/np.sqrt(180)
        es_s1 = np.mean(var_s1)
        # Expected Profit & Loss
        pl_s1 = np.mean(finalday_pf_1)
        variance_s1 = np.mean(np.var(finalday_pf_1))

        var_s2 = np.percentile(self.pf_s2[-1], 95)/np.sqrt(180)
        es_s2 = np.mean(var_s2)
        pl_s2 = np.mean(self.pf_s2[-1])
        variance_s2 = np.mean(np.var(self.pf_s2[-1]))
        
        result = {}
        result['VaR'] = ["{:.4e}".format(var_s1), "{:.4e}".format(var_s2)]
        result['Expected Shortfall'] = ["{:.4e}".format(es_s1), "{:.4e}".format(es_s2)]
        result['Profit & loss'] = ["{:.4e}".format(pl_s1), "{:.4e}".format(pl_s2)]
        result['Variance'] = ["{:.4e}".format(variance_s1), "{:.4e}".format(variance_s2)]
        
        self.result = pd.DataFrame(result)

    def state(self):
        self.get_result()
        print(self.result)

    def initial_dW(self):
        '''
        Generate dW
        '''
        self.dW_Q = np.random.randn(self.N, self.n_sim)
        self.dW_P = np.random.randn(self.N, self.n_sim)
        #self.dW_P = self.sigma*self.lamb*self.dt+self.dW_Q
        self._generate_dr()
        #self.rate_P = self.rate_Qs
        return self.dW_Q, self.dW_P
        
    def _generate_dr(self):
        self.rate_P = np.zeros((self.N+1, self.n_sim))
        self.rate_P.fill(self.r0)
        for i in range(1, self.N+1):
            epsilon = np.random.randn(1, self.n_sim)
            self.rate_P[i, ] = self.rate_P[i-1, ]+(self.kappa*(self.theta-self.rate_P[i-1, ])-self.lamb*self.sigma)*self.dt+self.sigma*np.sqrt(self.dt)*epsilon
    
    def _dx(self, x):
        return np.gradient(x, axis=0)
        '''
        return np.divide(np.gradient(
                x, axis=0
            ),
                np.gradient(self.rate_P, axis=0)
            )
        '''

    def _gamma_x(self, x):
        '''
        Calculate the gamma of x.
        '''
        return self._dx(self._dx(x))
    def _cal_price(self, due_date):
        T = due_date
        price = []
        delta = []
        gamma = []
        for t in np.arange(0, self.T+self.dt, self.dt):
            A = self._A(t, T)
            B = self._B(t, T)
            price_t = A*np.exp(-B* self.rate_P[int(t/self.dt)])
            delta_t = -B*price_t
            gamma_t = B*B*price_t
            price.append(price_t)
            delta.append(delta_t)
            gamma.append(gamma_t)

        return np.array(price), np.array(delta), np.array(gamma)
    def _cal_zcbprice(self):
        B = (1/self.kappa)*(1-np.exp(-self.kappa*(self.dt)))
        A = np.exp(
            (
                (self.theta-(np.power(self.sigma, 2)/(2*np.power(self.kappa, 2)))
                -
                (self.lamb*self.sigma/self.kappa)
                )
                *
                (B-self.dt)
            )
            -
            (
                (
                    (np.power(self.sigma, 2)/(4*self.kappa))*np.power(B, 2)
                )
            )
        )
        price = []
        for t in np.arange(0, self.T+self.dt, self.dt):

            temp_price = A*np.exp(-B*self.rate_P[int(t/self.dt)])
            price.append(temp_price)
        return np.array(price)

    def depercated_cal_price(self, due_date, mode='P'):
        '''
        Calculate prices with different due date
        '''
        price_list = []
        T = due_date
        mode = mode.upper()
        if mode == 'P':
            for t in np.arange(0, self.T+self.dt, self.dt):
                price_t = self._A_P(t, T)*np.exp(-self._B(t, T)
                                                 * self.rate_P[int(t/self.T*self.N)])
                price_list.append(price_t)
        elif mode == 'Q':
            for t in np.arange(0, self.T+self.dt, self.dt):
                price_t = self._A_Q(t, T)*np.exp(-self._B(t, T) *
                                                 self.rate_Q[int(t/self.T*self.N), ])
                price_list.append(price_t)
        return np.array(price_list)
    
    def cal_quant_delta(self):
        '''
        Calculate the quantity using derivative
        Delata hedging
        dPt/dPs
        '''
        self.quant_s_delta = np.divide(self.dT, self.dS)
    
    def cal_quant_gamma(self):
        '''
        Calculate gamma hedging quantity
        '''
        '''
        u_gamma = list()
        for t in np.arange(0, self.T+self.dt, self.dt):
            temp = (np.power(-self._B(t, self.S), 2)-self.lamb*self.sigma/self.kappa)*(self._B(t, self.S)-self.S+t)-((self.sigma/(4*self.kappa))*np.power(self._B(t, self.S), 2))
            u_gamma.append(temp)
        '''
        #self.quant_u_gamma = np.array(u_gamma)
        self.quant_u_gamma = np.divide((self.gamma_S*self.dT-self.dS*self.gamma_T), (self.dU*self.gamma_S-self.gamma_U*self.dS))
        self.quant_s_gamma = np.divide((self.gamma_U*self.dT-self.dU*self.gamma_T), (self.dS*self.gamma_U-self.dU*self.gamma_S))

    def portfolio(self):
        self.portfolio_s1()
        self.portfolio_s2()
    
    def portfolio_s1(self):
        qz = np.zeros((self.N+1, self.n_sim))
        qz[0,] = np.divide(self.Tprice[0]-self.Sprice[0]*self.quant_s_delta[0], self.ZCB_price[0])

        for i in range(1, self.N+1):
            # Change here
            qz[i] = np.divide(self.Sprice[i]*self.quant_s_delta[i-1]-self.Sprice[i]*self.quant_s_delta[i]+qz[i-1], self.ZCB_price[i])
        #self.qz[1:,] = np.divide(self.Sprice[:-1, ]*self.quant_s_delta[:-1, ]-self.Sprice[1:, ]*self.quant_s_delta[1:, ]+self.ZCB_price[:-1, ], self.ZCB_price[1:, ])
        

        self.pf_s1 = -self.Tprice+self.quant_s_delta*self.Sprice+qz*self.ZCB_price
        self.qz_s1 = qz
    def portfolio_s2(self):
        qz = np.zeros((self.N+1, self.n_sim))
        qS_s2 = np.divide(self.dU*self.gamma_T-self.dT*self.gamma_U, self.dU*self.gamma_S-self.dS*self.gamma_U)
        qz[0] = np.divide(self.Tprice[0]-qS_s2[0]*self.Sprice[0]-self.quant_u_gamma[0]*self.Uprice[0], self.ZCB_price[0])
        for i in range(1, self.N+1):
            qz[i] = np.divide(qS_s2[i-1]*self.Sprice[i]+qz[i-1]+self.quant_u_gamma[i-1]*self.Uprice[i]-qS_s2[i]*self.Sprice[i]-self.quant_u_gamma[i]*self.Uprice[i], self.ZCB_price[i])
        '''
        Old version
        Left here in case the test version doesn't work
        qz[0] = np.divide(self.Tprice[0]-self.Sprice[0]*self.quant_s_gamma[0]-self.Uprice[0]*self.quant_u_gamma[0], self.ZCB_price[0])

        for i in range(1, self.N+1):
            qz[i] = np.divide(self.Sprice[i-1]*self.quant_s_gamma[i-1]-self.Sprice[i]*self.quant_s_gamma[i]+qz[i-1]+self.Uprice[i-1]*self.quant_u_gamma[i]-self.Uprice[i]*self.quant_u_gamma[i], self.ZCB_price[i])
        '''
        self.pf_s2 = -self.Tprice+qS_s2*self.Sprice+qz*self.ZCB_price+self.quant_u_gamma*self.Uprice
        #self.pf_s2 = -self.Tprice+self.quant_s_gamma*self.Sprice+self.Uprice*self.quant_u_gamma+qz*self.ZCB_price
        self.qz_s2 = qz
        self.qS_s2 = qS_s2

    def _A(self, t, T):
        result = np.exp(
            (
                (self.theta-(np.power(self.sigma, 2)/(2*np.power(self.kappa, 2)))
                -
                (self.lamb*self.sigma/self.kappa)
                )
                *
                (self._B(t, T)-T+t)
            )
            -
            (
                (
                    (np.power(self.sigma, 2)/(4*self.kappa))*np.power(self._B(t, T), 2)
                )
            )
        )
        return result

    def _B(self, t, T):
        '''
        B(t) for bonds
        '''
        return (1/self.kappa)*(1-np.exp(-self.kappa*(T-t)))


if __name__ == '__main__':
    model = Vasicek()