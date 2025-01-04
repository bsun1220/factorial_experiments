import unittest
import random
from src import *

rand_seed = 123123
class TestFactorialWeights(unittest.TestCase):
    def test_preprocess(self):
        random.seed(rand_seed)
        num_points = 10000
        df = pd.DataFrame()
        num_treatments = 4
        
        treat_cols = []
        for i in range(num_treatments):
            treat_name = 'z'+str(i)
            df[treat_name] = (2 * (np.random.randint(0, 2, num_points) - 0.5)).astype(int)
            treat_cols.append(treat_name)
        
        num_covs = 3
        cov_cols = []
        for i in range(num_covs):
            cov_name = 'x'+str(i)
            df[cov_name] = np.random.randint(0, 2, num_points)
            cov_cols.append(cov_name)
            
        df['y'] = np.random.randint(0, 2, num_points)

        fw = FactorialWeights()
        new_df = fw.preprocess_data(df, treat_cols, cov_cols, 'y')
        self.assertTrue(new_df.shape == df.shape)

    def test_hetero(self):
        random.seed(rand_seed)
        num_points = 10000
        df = pd.DataFrame()
        num_treatments = 4
        
        treat_cols = []
        for i in range(num_treatments):
            treat_name = 'z'+str(i)
            df[treat_name] = (2 * (np.random.randint(0, 2, num_points) - 0.5)).astype(int)
            treat_cols.append(treat_name)
        
        num_covs = 3
        cov_cols = []
        for i in range(num_covs):
            cov_name = 'x'+str(i)
            df[cov_name] = np.random.randint(0, 2, num_points)
            cov_cols.append(cov_name)
            
        df['y'] = np.random.randint(0, 2, num_points)

        fw = FactorialWeights()
        df = fw.preprocess_data(df, treat_cols, cov_cols, 'y')

        effects = ['z0']
        weights = fw.heterogeneous_treat_model(df, treat_cols, cov_cols, effects)
        self.assertTrue((weights >= 0).all())
        self.assertTrue(len(weights) == len(df['y']))

    def test_additive(self):
        random.seed(rand_seed)
        num_points = 10000
        df = pd.DataFrame()
        num_treatments = 4
        
        treat_cols = []
        for i in range(num_treatments):
            treat_name = 'z'+str(i)
            df[treat_name] = (2 * (np.random.randint(0, 2, num_points) - 0.5)).astype(int)
            treat_cols.append(treat_name)
        
        num_covs = 3
        cov_cols = []
        for i in range(num_covs):
            cov_name = 'x'+str(i)
            df[cov_name] = np.random.randint(0, 2, num_points)
            cov_cols.append(cov_name)
            
        df['y'] = np.random.randint(0, 2, num_points)

        fw = FactorialWeights()
        df = fw.preprocess_data(df, treat_cols, cov_cols, 'y')

        effects = ['z0']
        weights = fw.additive_treat_model(df, treat_cols, cov_cols, effects)
        self.assertTrue((weights >= 0).all())
        self.assertTrue(len(weights) == len(df['y']))
                
        
if __name__ == '__main__':
    unittest.main()
