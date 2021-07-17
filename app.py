#Heart sound
from re import I
from keras.preprocessing import image
import random
import os
from keyword_spotting_service import Keyword_Spotting_Service
#ecg lib
import os
import urllib.request
import pandas as pd
import numpy as np
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import json
import plotly
import plotly.figure_factory as ff
import plotly.offline as py
import plotly.graph_objs as go
import configparser
#import cufflinks as cf
#cf.go_offline()
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import argparse
from keras import backend as K

JSON_PATH = "data.json"
UPLOAD_FOLDER = 'uploads'

app = Flask(__name__)


ALLOWED_EXTENSIONS = set(['csv', 'xlsx','xls'])
def allowed_file(filename):
	
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



#route
@app.route("/")
def main():
   return render_template('index1main.html')

#heart sound
@app.route('/index_heart',methods = ['GET'])
def get_audio():

    return render_template('index_heart.html')

@app.route('/predict_heart',methods=['GET','POST'])
def predictheart():
	if request.method == 'POST':

		# get file from POST request and save it
		audio_file = request.files['file']
		file_name = str(random.randint(0, 100000))
		audio_file.save(file_name)


		# instantiate keyword spotting service singleton and get prediction
		kss = Keyword_Spotting_Service()
		predicted_keyword,signal = kss.predict(file_name)

		# we don't need the audio file any more - let's delete it!
		os.remove(file_name)

		# dictionary where we'll store mapping, labels, MFCCs and filenames
		data_audio = {
			"result": [],
			"signal": []
			}

		data_audio["result"].append(predicted_keyword) 
		data_audio["signal"].append(signal.T.tolist()) 

		# save data in json file
		with open(JSON_PATH, "w") as fp:
		   json.dump(data_audio, fp, indent=4)
	
	#create table
	out_df1 = pd.DataFrame( columns =['Heart_Class'], data =np.array([predicted_keyword]).transpose())
	out_df1['User_id'] = out_df1.index+1
	out_df1 = out_df1[['User_id', 'Heart_Class']]

	colorscale = [[0, '#4d004c'],[.5, '#f2e5ff'],[1, '#ffffff']]
	table = ff.create_table(out_df1, colorscale=colorscale, height_constant = 20)
	table.to_html()
	pp_table = table.to_html()

	#create graph
	graphs = createHeartGraph(signal,predicted_keyword)

	return render_template('predict_heart.html',table = pp_table,graphplot = graphs)

def createHeartGraph(signal, predicted_keyword):

	df_index = len(signal)-1
	
	xvals = list(range(0,df_index ))
	yvals = list(signal)

	graphs = [
       			 {
            		'data': [
               	 	{
						"x": xvals,
						"y":yvals,
						"type": "scatter"
        	
					}
            				],

            		'layout': {
                		'title': f"ECG Readings for the record# {df_index}, ECG class = {predicted_keyword} <br>",
                	'yaxis': {
                    	'title': "Heart Sound"
                			},
                	'xaxis': {
                    	'title': "Time instance"
                			}
            					}
        		}
			]
	print("grap completed..")
	return graphs


#ECG
@app.route('/index_ecg')
def upload_form():

	return render_template('index_ecg.html')

@app.route('/uploader', methods=['GET','POST'])
def uploader():

	if request.method == 'POST':
        # check if the post request has the file part
		
		test_flag = ''
		if 'file' in request.files:
			test_flag = 'file'
		else:
			test_flag = 'demo'
			demo_data = request.form['samplevalue']
		
		if test_flag == 'demo' :
			demo_data = demo_data.split(',')
			demo_data = [ float(val) for val in demo_data]
			
			out_df, graphJSON = predictionHandler(demo_data = demo_data)
			#print("Show the shape of output DF")
			#print(out_df.shape)
			colorscale = [[0, '#4d004c'],[.5, '#f2e5ff'],[1, '#ffffff']]
			table = ff.create_table(out_df, colorscale=colorscale, height_constant = 20)
			
			table.to_html()
			pp_table = table.to_html()
			
			return render_template('response_ecg.html', table = pp_table, graphplot = graphJSON)

			
		else:
			file = request.files['file']
			if file.filename == '':
				flash('No file selected for uploading')
				return redirect(request.url)
			if file and allowed_file(file.filename):
				filename = secure_filename(file.filename)
				file.save(filename)
				flash('File successfully uploaded...call the handler now..')

				extension = file.filename.split('.')[1]
				plot_index = request.form['plot_sample']
				
				out_df, graphJSON = predictionHandler(file.filename,extension,plot_index= plot_index)
								
				colorscale = [[0, '#4d004c'],[.5, '#f2e5ff'],[1, '#ffffff']]
				table = ff.create_table(out_df, colorscale=colorscale, height_constant = 20)
				table.to_html()
				pp_table = table.to_html()
				
				return render_template('response_ecg.html', table = pp_table, graphplot = graphJSON)
			else:
				flash('Allowed file types are csv,xls,xlsx')
				return redirect(request.url)

def predictionHandler(test_file=False,extension='', plot_index=1, demo_data=[]):

	plot_index = int(plot_index)
	if test_file:
		if extension == "csv":
			df = pd.read_csv(test_file)
		elif (extension == "xls" or extension == "xlsx"):
			df = pd.read_excel(test_file)
		else:
			raise ValueError('Input file with unexpected extension, please use csv, xlsx,xls files')
		test_rec = df.values
		test_rec = test_rec.reshape(test_rec.shape[0], test_rec.shape[1],1)
		
	
	else:
		test_rec =  np.array(demo_data)
		test_rec = test_rec.reshape(1, test_rec.shape[0],1)

		df_data = np.array(demo_data)
		df_data = df_data.reshape(1,df_data.shape[0])
		
		df = pd.DataFrame(data=df_data)
	
	model_ECG_loaded = load_model('models/model_ECG_final.h5')
	model_MI_loaded = load_model('models/model_MI_final.h5')
	print("models loaded...")
	out_classes = model_ECG_loaded.predict(test_rec)
	print("prediction completed..")
	ECG_class = np.argmax(out_classes,axis=1)	

	out_classes = model_MI_loaded.predict(test_rec)
	MI_class = np.argmax(out_classes,axis=1)

	out_df = pd.DataFrame(columns =['ECG_Class', 'MI_Class'], data = np.array([ECG_class, MI_class]).transpose())
	out_df['User_id'] = out_df.index+1
	out_df = out_df[['User_id', 'ECG_Class','MI_Class']]
	ecg_clas_mapper = {0:'N', 1:'S', 2:'V', 3:'F',4:'Q'}
	MI_class_mapper = {0:'Normal', 1:'Abnormal'}

	out_df.ECG_Class = out_df.ECG_Class.map(ecg_clas_mapper)
	out_df.MI_Class = out_df.MI_Class.map(MI_class_mapper)

	ecg_class = out_df.iloc[plot_index-1].ECG_Class
	mi_class = out_df.iloc[plot_index-1].MI_Class
	if mi_class == 0:
		mi_class = 'Normal'
	else:
		mi_class = 'Abnormality'
	graphs = createECGGraph(df,plot_index,ecg_class,mi_class)
	graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
	return out_df,graphJSON


def createECGGraph(df, plot_index, ecg_class, mi_class):

	df_index = plot_index-1
	
	xvals = list(range(0, df.iloc[df_index].count()))
	yvals = list(df.iloc[df_index].values)

	graphs = [
       			 {
            		'data': [
               	 	{
						"x": xvals,
						"y":yvals,
						"type": "scatter"
        	
					}
            				],

            		'layout': {
                		'title': f"ECG Readings for the record# {plot_index}, ECG class = {ecg_class} <br> MI tests shows {mi_class}",
                	'yaxis': {
                    	'title': "ECG Readings"
                			},
                	'xaxis': {
                    	'title': "Time instance"
                			}
            					}
        		}
			]
	print("grap completed..")
	return graphs


if __name__ == '__main__':
    app.run(debug=True)
     
