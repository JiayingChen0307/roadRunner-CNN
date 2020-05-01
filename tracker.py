import pickle
import os

class Tracker():

  def __init__(self):
    self.info=[]
  
  #load experiment history
  def import_db(self,db_filename):
      if os.path.isfile(db_filename):
        dbfile = open(db_filename, 'rb')
        db = pickle.load(dbfile)
      else:
        db = []
      
      return db
  
  #write new experiment to history 
  def update_db(self,db,record):
    
      db.append(record)
      if os.path.isfile('Tina/history'):
         os.remove('Tina/history')
      dbfile = open('Tina/history', 'ab') 
      pickle.dump(db, dbfile) 
  
  #look up an experiment with given condition
  def find_record(self,db,info_key,expected_value):
   
      for d in db:
        if d is not None & d.get(info_key, "") == expected_value:
          return db.index(d)
        else:
          print('No matched experiment found.')
          
  