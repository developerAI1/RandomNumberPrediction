from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
import numpy as np
from myapp.models import MyArray
import re
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout,Bidirectional
from random import randint
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Create your views here.

num_samples = 3000
num_inputs = 10
num_outputs =7

   
# def SaveArray(request):
#     arrays = [np.random.choice(np.arange(1, 26), size=10, replace=False) for i in range(num_samples)]
#     print(len(arrays))
#     for arr in arrays:
#         print(arr)
#         MyArray.objects.create(data=arr)    
#     return HttpResponse(request,'Random arrays saved!')

def prepare_data(request,data):
# def prepare_data(request):
    num_inputs = 10
    num_outputs =7
    
    arrays=[]
    input=[]
    output=[]
    data=MyArray.objects.all().values()
    for list_values in data:
        for key , value in list_values.items():
            if key=='data':
                arrays.append(value)
                
    for arr in arrays:
        integer_array = [int(x) for x in re.findall(r'\d+', arr)]  
        input_data = np.zeros((len(arrays),num_inputs),dtype=int)
        output_data = np.zeros((len(arrays),num_outputs),dtype=int)
        arr2 = np.array(integer_array)
        
        input_data=integer_array
        input.append(input_data)  
          
        indices = np.random.choice(10, size=7, replace=False)
        output_data=arr2[indices]
        output.append(output_data)
        
    input=np.array(input)
    output=np.array(output)
    print('Length of data ------------------->>>>>',len(arrays))
    from sklearn.model_selection import train_test_split
    train_x,test_x,train_y,test_y = train_test_split(input,output,test_size=0.2,random_state=42)

    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()

    train_x=scaler.fit_transform(train_x)
    test_x=scaler.transform(test_x)

    train_X=train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
    test_X=test_x.reshape(test_x.shape[0], 1, test_x.shape[1])

    print(train_X.shape)
    print(test_X.shape)
    
    model = Sequential()
    model.add(Bidirectional(LSTM(240,return_sequences=True,input_shape=(1,10))))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(240,return_sequences=True,input_shape=(1,10))))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(240,return_sequences=True, input_shape=(1,10))))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(240,return_sequences=True, input_shape=(1,10))))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(240,return_sequences=True, input_shape=(1,10))))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(240,return_sequences=False, input_shape=(1,10))))
    model.add(Dense(7))

    model.compile(optimizer='adam',loss = 'mse',metrics=['accuracy'])
    model.fit(train_X, train_y, epochs=400, batch_size=100, verbose=1)
    
    y_pred=model.predict(test_X[[0]]).astype(int)
    prediction=(set(y_pred.flatten()))
    print(prediction)   
    new_sequence = np.random.choice(np.arange(1,26), size=10, replace=False)
    AI_guess_number=[]
    count=0
    for i in list(prediction):
        for j in list(new_sequence):
            if i== j:
                AI_guess_number.append(i)
                count += 1
            if count == 3:
                break
    print('AI_guess_three-number',AI_guess_number)
    print('new_sequence',new_sequence)
    
    
def add_new_sequence(request):
    input= [1,8,11,3,15,20,21,22,14,17]
    new_input=np.array(input)
    latest_id = MyArray.objects.latest('id').id if MyArray.objects.exists() else 0
    arr_obj = MyArray(id=latest_id+1, data=new_input)
    print('----------->>>>>>',arr_obj)
    arr_obj.save()
    data = MyArray.objects.all().values()
    prepare_data(request , data)

