import pymongo
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import joblib
from progress.bar import Bar

class Train:
    def __init__(self,config):
        self.config = config
        self.collection = pymongo.MongoClient(config['mongo_path'])[self.config['db_name_data']][self.config['collection_name_preprocessed']+'_'+self.config['method']]

    def get_df_reg(self):
        col_reg = list()
        bar = Bar('Fetching',max=self.collection.count_documents({}),fill='#')
        counter = 0
        for doc in self.collection.find():
            l = list()
            l.append(doc['Seconds(absolute)'])
            l.append(float(doc['Current'])) #Important for knr
            l.append(doc['Timestamp'])
            l.append(doc['Year'])
            l.append(doc['Month'])
            l.append(doc['Day'])
            l.append(doc['Hour'])
            l.append(doc['Minute'])
            l.append(doc['Second'])
            l.append(doc['Weekday'])
            col_reg.append(l)
            counter+=1
            if(counter%50000==0):
                bar.next(n=50000)
        bar.finish()
        self.df_reg = pd.DataFrame(col_reg,columns=['Seconds(absolute)','Current','Timestamp',\
            'Year','Month','Day','Hour','Minute','Second','Weekday'])

    def train(self):
        if self.config['method'] == 'regression':
            print('Building regression model')
            print('Fetching data')
            self.get_df_reg()
            print('Data Fetched')
            print('Splitting data')
            df_x = self.df_reg.iloc[:,3:]
            df_y = self.df_reg.iloc[:,1]
            x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.2,random_state=1)
            print('Data splitted')
            print('Size of x_train',x_train.shape)
            print('Size of y_train',y_train.shape)
            print('Size of x_test',x_test.shape)
            print('Size of y_test',y_test.shape)

            if self.config['model'] == 'svr':
                print('Support vector regressor')
                model = SVR(kernel=self.config['svr_kernel'])
            if self.config['model'] == 'knr':
                print('K-nearest neighbors regressor')
                model = KNeighborsRegressor(n_jobs=12)
            if self.config['model'] == 'dtr':
                print('Decision tree regressor')
                model = DecisionTreeRegressor()
            if self.config['model'] == 'rf':
                print('Random forest regressor')
                model = RandomForestRegressor(n_jobs=12)
            if self.config['model'] == 'et':
                print('Extra trees regressor')
                model = ExtraTreesRegressor(n_jobs=12)
            if self.config['model'] == 'gbr':
                print('Gradient boosting regressor')
                model = GradientBoostingRegressor()
            
            try:
                model
            except BaseException:
                print('Invalid model configuration. Check config.ini')
                return


            model.fit(x_train,y_train)
            pred = pd.Series(model.predict(df_x))
            self.df_reg.insert(2,'Predicted_current',pred)
            print('R^2 score',model.score(x_test,y_test))

            print('Converting to binary classification')
            y_test_list,y_pred_list,_,_ = self.to_bin_cl(x_test,y_test,model)
            _,_,bin_y,bin_y_pred = self.to_bin_cl(df_x,df_y,model)
            conf_mat = confusion_matrix(y_true=y_test_list,y_pred=y_pred_list)
            print('Converted to binary classification')
            
            self.df_reg.insert(3,'Actual_class',bin_y)
            self.df_reg.insert(4,'Predicted_class',bin_y_pred)

            print('Confusion matrix:\n',conf_mat)
            p=conf_mat[0][0]/(conf_mat[0][0]+conf_mat[1][0])
            r=conf_mat[0][0]/(conf_mat[0][0]+conf_mat[0][1])
            print('Accuracy is',np.sum(np.array(y_test_list) == y_pred_list)/len(y_pred_list) )
            print('Precision is',p)
            print('Recall is',r)
            print('F1-score is',self.get_f_score(p,r,1))
            print('F0.5-score is',self.get_f_score(p,r,0.5))
            print('F2-score is',self.get_f_score(p,r,2))


            # joblib.dump(model,'models/'+self.config['model']+'.model')
            self.save_result()
            # self.test_vis(model,x_test,y_test)


    def save_result(self):
        self.df_reg.to_csv(self.config['result_path']+'_'+self.config['model']+'.csv',index=False)

    def get_f_score(self,p,r,beta=1):
        return (1+beta**2)*p*r/(beta**2*p+r)
    
    def to_bin_cl(self,x,y,model):
        y_test_list,y_pred_list = [],[]
        y_pred = model.predict(x)
        for (yt,yp) in zip(y.tolist(),y_pred.tolist()):
            if float(yt) > float(self.config['reg_thres']):
                y_test_list.append(1)
            else:
                y_test_list.append(0)
            if float(yp) > float(self.config['reg_thres']):
                y_pred_list.append(1)
            else:
                y_pred_list.append(0)

        y_test_bin,y_pred_bin = [],[]
        for (yt,yp) in zip(y_test_list,y_pred_list):
            if int(yt)==1:
                y_test_bin.append('Functional')
            else:
                y_test_bin.append('Disfunctional')
            if int(yp)==1:
                y_pred_bin.append('Functional')
            else:
                y_pred_bin.append('Disfunctional')
        bin_test = pd.Series(y_test_bin)
        bin_pred = pd.Series(y_pred_bin)
        return y_test_list,y_pred_list,bin_test,bin_pred

    def test_vis(self,model,x,y):
        matplotlib.rcParams['agg.path.chunksize'] = 100000
        y_pred = model.predict(x).tolist()
        y = y.tolist()
        plt.scatter(y,y_pred)
        plt.xlabel("Real current")
        plt.ylabel("Predicted current")
        plt.show()