from os import listdir
from os.path import isfile, join
import pymongo
import gridfs
import time
from progress.bar import Bar

class Storage:
    def __init__(self,config):
        self.config = config

        self.client = pymongo.MongoClient(self.config['mongo_path'])
        self.db_gridfs = self.client[self.config['db_name_gridfs']]
        self.db_data = self.client[self.config['db_name_data']]
        self.col_raw = self.db_data[self.config['collection_name_raw']]
        self.col_prpr = self.db_data[self.config['collection_name_preprocessed']+'_'+self.config['method']]
        self.path_data = self.config['path_data']
        self.gfs = gridfs.GridFS(self.db_gridfs)
        
    def get_file_list(self):
        self.f_list = [join(self.path_data,f) for f in listdir(self.path_data) if isfile(join(self.path_data,f))]

    def get_raw(self):
        self.get_file_list()
        self.raw_list = list()
        for f in self.f_list:
            with open(f,'r') as file:
                for line in file.readlines():
                    head,_,_ = line.partition("#")
                    if len(head)!=0:
                        d = dict()
                        d["Seconds(absolute)"] = head.split()[0]
                        d["Current"] = head.split()[1]
                        d["Timestamp"] = head.split()[2]
                        self.raw_list.append(d)

    def get_pr_reg(self):
        self.get_raw()
        self.pr_reg_list = list()
        for doc in self.raw_list:
            new_doc = doc
            t = time.strptime(new_doc["Timestamp"].replace("/"," ").replace("T"," ").replace(":"," "),"%d %m %y %H %M %S")
            new_doc['Year'] = t[0]
            new_doc['Month'] = t[1]
            new_doc['Day'] = t[2]
            new_doc['Hour'] = t[3]
            new_doc['Minute'] = t[4]
            new_doc['Second'] = t[5]
            new_doc['Weekday'] = t[6]
            self.pr_reg_list.append(new_doc)

    def to_mongo(self):
        if self.config['pre_phase'] == 'file':
            self.file_to_gridfs()

        if self.config['pre_phase'] == 'raw':
            self.file_to_raw()
        
        if self.config['pre_phase'] == 'pre':
            if self.config['method'] == 'regression':
                self.file_to_preprocessed_regression()

    def file_to_gridfs(self):
        print('Generating file list')
        self.get_file_list()
        print('File list generated')
        print('Saving files to GridFS')
        # file --> GridFS
        for f in self.f_list:
            self.gfs.put( open(f,'rb') )
        print('Done')

    def file_to_raw(self):
        print('Loading files')
        self.get_raw()
        print('Files loaded')
        # raw data --> beam_data_raw
        if(self.col_raw.find_one() is not None):
            print('Deleting old documents')
            self.col_raw.drop()
        print('Inserting new documents')
        self.col_raw.insert_many(self.raw_list)
        print('Done')

    def file_to_preprocessed_regression(self):
        print('Pre-processing...')
        self.get_pr_reg()
        print('Pre-processed')
        # raw data --> preprocess --> beam_data_preprocessed
        if(self.col_prpr.find_one() is not None):
            print('Deleting old documents')
            self.col_prpr.drop()
        print('Inserting new documents')
        self.col_prpr.insert_many(self.pr_reg_list)
        print('Done')