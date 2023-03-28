from django.shortcuts import render , redirect
from django.http import HttpResponse,JsonResponse
import numpy as np
from myapp.models import MyArray
import re
import os  
from sklearn.preprocessing import StandardScaler 
import pandas as pd    
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout,Bidirectional
from random import randint
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from keras.models import model_from_json
import random
from django.views.decorators.csrf import csrf_exempt
from keras.models import model_from_json
from django.db.models import Q
from django.contrib import messages
import os
import ast
from sklearn.model_selection import train_test_split
from keras.layers import Flatten
from collections import Counter

count_sequence =1   

path = os.path.abspath("database/Data_based_Sequences.txt")
with open(path, 'r') as file:
    content = file.read()
arrays = []
lines = content.split('\n')
for line in lines:  
    # Join the string values with a space to form a single string
    arr = ' '.join(line.split())
    # Find all integer values in the string using regex and convert them to integers
    integer_array = [int(x) for x in re.findall(r'\d+', arr)]
    arrays.append(integer_array)
del arrays[-1]


def SaveDAtabase(request):
    for arr in arrays: 
        MyArray.objects.create(data=arr)   
    return HttpResponse(request,'Random arrays saved!')
# def home(request):
#     return render(request, "index.html")
       
# @csrf_exempt
# def add_new_sequence(request):
#     global count_sequence       
#     if request.method=='POST':
#         NewAddedArray= request.POST.get('array')
#         integer_array = [int(x) for x in re.findall(r'\d+', NewAddedArray)]  
#         filtered_lst = [item for item in integer_array if item <= 25]
#         New=[i for i in filtered_lst if filtered_lst.count(i)==1]
#         if len(New)==10:
#             latest_id = MyArray.objects.latest('id').id if MyArray.objects.exists() else 0
#             arr_obj = MyArray(id=latest_id+1, data=New,status=1)
#             # arr_obj = MyArray(id=latest_id+1, data=New)   
#             arr_obj.save()
#             num_status_1 = MyArray.objects.filter(status=1).count()
#             if num_status_1 == count_sequence:
#                 MyArray.objects.filter(status=1).update(status=2)
#                 data = MyArray.objects.all().values()
#                 # prepare_data(request ,data)
#                 arrays=[]
#                 for list_values in data:
#                     for key, value in list_values.items():
#                         if key == 'data':
#                             arrays.append(value)
#                 sequence=[ast.literal_eval(arr) for arr in arrays]     
#                 df=pd.DataFrame(sequence,columns =[f'seq_{i}' for i in range(1,11)])
                
#                 scaler=StandardScaler().fit(df.values)
#                 transformed_dataset=scaler.transform(df.values)
#                 transformed_df=pd.DataFrame(data=transformed_dataset,index=df.index)
                
#                 ## Define gyper params of model
#                 number_of_rows=df.shape[0]
#                 window_length=4                                   # time steps
#                 n_feature = 10                           # steps tp predict

#                 input=np.zeros([number_of_rows-window_length ,window_length,10], dtype=float)
#                 output=np.zeros([number_of_rows-window_length,n_feature],dtype=float)
#                 for i in range(0,number_of_rows - window_length):
#                     input_window = transformed_df.iloc[i:i+window_length, :10].values
#                     input[i] = input_window
#                     output_window = transformed_df.iloc[i+window_length:i+window_length+1, :n_feature].values
#                     output[i] = output_window.squeeze()
                    
#                 train_x,test_x,train_y,test_y = train_test_split(input,output,test_size=0.25,random_state=42)
                
#                 model = Sequential()
#                 model.add(Bidirectional(LSTM(240,return_sequences=True,input_shape=(window_length,n_feature))))
#                 model.add(Dropout(0.2))
#                 model.add(Bidirectional(LSTM(240,return_sequences=True, input_shape=(window_length,n_feature))))
#                 model.add(Dropout(0.2))
#                 model.add(Bidirectional(LSTM(240,return_sequences=True, input_shape=(window_length,n_feature))))
#                 model.add(Dropout(0.2))
#                 model.add(Bidirectional(LSTM(240,return_sequences=True, input_shape=(window_length,n_feature))))
#                 model.add(Dropout(0.2))
#                 model.add(Bidirectional(LSTM(240,return_sequences=True, input_shape=(window_length,n_feature))))
#                 model.add(Dropout(0.2)  )
#                 model.add(Bidirectional(LSTM(240,return_sequences=False, input_shape=(window_length,n_feature))))
#                 model.add(Dense(n_feature))
                
#                 model.compile(optimizer='adam',loss = 'mse',metrics=['accuracy'])
#                 model.fit(train_x, train_y, epochs=500, batch_size=100, verbose=1,shuffle=False)
                               
#                 # model_json = model.to_json()
#                 # with open("saved_model/AIGuessModel.json", "w") as json_file:
#                 #     json_file.write(model_json)   

#                 # model.save_weights("saved_model/modelweights.h5") 
#                 last_sequence=transformed_df.iloc[-window_length:,:]
#                 scaled_input = scaler.transform(last_sequence.values)
#                 predicted_output = model.predict(scaled_input.reshape(1,window_length,n_feature),batch_size=100,verbose=1)
#                 output=scaler.inverse_transform(predicted_output).astype(int)
#                 print(output)  
#                 input_sequence1= df.iloc[-1:].values
#                 input_sequence2= df.iloc[-2:].values
#                 input_sequence3= df.iloc[-3:].values
#                 input_sequence4= df.iloc[-5:].values 
#                 input_sequence5= df.iloc[-7:].values 

#                 # scl_input=scaler.inverse_transform(input_sequence).astype(int)
#                 input1=[integer for x in input_sequence1 for integer in x]
#                 input2=[integer for x in input_sequence2 for integer in x]
#                 input3=[integer for x in input_sequence3 for integer in x]
#                 input4=[integer for x in input_sequence4 for integer in x]
#                 input5=[integer for x in input_sequence5 for integer in x]
                
#                 all_words1=np.concatenate([input1,output.flatten()])
#                 all_words2=np.concatenate([input2,output.flatten()])
#                 all_words3=np.concatenate([input3,output.flatten()])
#                 all_words4=np.concatenate([input4,output.flatten()])
#                 all_words5=np.concatenate([input5,output.flatten()])

#                 freq1=Counter(all_words1)
#                 freq2=Counter(all_words2)
#                 freq3=Counter(all_words3)
#                 freq4=Counter(all_words4)
#                 freq5=Counter(all_words5)

#                 most_repeated_prob1=[x[0] for x in freq1.most_common(3)]
#                 most_repeated_prob2=[x[0] for x in freq2.most_common(3)]
#                 most_repeated_prob3=[x[0] for x in freq3.most_common(3)]
#                 most_repeated_prob4=[x[0] for x in freq4.most_common(3)]
#                 most_repeated_prob5=[x[0] for x in freq5.most_common(3)]
#                 random_value=[most_repeated_prob1,
#                         most_repeated_prob2,
#                         most_repeated_prob3,
#                         most_repeated_prob4,
#                         most_repeated_prob5]
#                 return render(request , "genrate.html" ,    {"random_values":random_value})
#         else:
#             return redirect("/")
#     else:
#         return redirect("/")
    
# def array_history(request):
#     recent_sequences = MyArray.objects.filter(Q(status=1) | Q(status=2))
#     print(recent_sequences)
#     return render(request, 'history.html' , {'recent_sequences':recent_sequences})  

# @csrf_exempt
# def Reset_history(request):
#     recent_sequences = MyArray.objects.filter(Q(status=1) | Q(status=2))
#     val = request.POST.get('input')
#     if request.method=='POST':
#         val = request.POST.getlist('input')
#         if val:
#             val=list(map(int,val))
#             for i in val:
#                 MyArray.objects.filter(id=i).delete()
#                 messages.success(request, "Selected sequences have been deleted.")
#             return  redirect('/')
#         else:
#             messages.warning(request, "No sequences were selected.")
        
#     return render(request, 'reset.html' , {'recent_sequences':recent_sequences})  


  
 