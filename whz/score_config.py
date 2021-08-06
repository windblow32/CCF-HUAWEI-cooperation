class config:
    s_RMSE = 1
    s_R2 = 10
    s_Median_AE = 1
    s_Mean_AE = 1
    s_Explained_VS = 10
    def __init__(self):
        i = 0
    def load(self):

        return self.s_RMSE, self.s_R2, \
               self.s_Median_AE, self.s_Mean_AE,\
               self.s_Explained_VS


