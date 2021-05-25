from abstract_scaler import AbstractScaler

from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError

import numpy as np

class MinMax (AbstractScaler):

    def __init__(self, feature_range = (0, 1) ) -> None:
        super().__init__()
        self.scaler = MinMaxScaler(feature_range) 
        self.is_fit = False


    def learn(self,timeseries: list) -> None:
        """
        A method to calibrate the scaler ranges 
        """

        raw_data = np.array ( timeseries )
        raw_data = raw_data.reshape(raw_data.shape[0], 1) #reshape operation is not in place
        self.scaler.fit(raw_data)
        self.is_fit = True


    def work(self, timeseries: list, FWD = True) -> np.ndarray:
        """
        A method to scale and de_scale data
        city_name_MW is a pandas series
        TODO: make sure de_scale is not called before scale
        """   
        
        if self.is_fit:
            if FWD:
                # scaling
                raw_data = np.array ( timeseries )
                raw_data = raw_data.reshape(raw_data.shape[0], 1) #reshape operation is not in place
                scaled = self.scaler.transform(raw_data)
                return scaled
            else:
                # scalling back
                recons_data = np.array ( timeseries )
                recons_data = recons_data.reshape(recons_data.shape[0], 1) #reshape operation is not in place
                de_scaled = self.scaler.inverse_transform(recons_data)
                return de_scaled                    
        else:
            print("\n")
            print("Make sure to call the learn method before the work method")
            print("\n")
            return None
